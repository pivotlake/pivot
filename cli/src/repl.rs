//! The interactive shell: psql's look and feel, straight onto the engine.
//!
//! Statements run through [`server::Session`], the same embedded path the
//! server's own console uses, so there is no wire protocol and no socket in
//! the loop. Two modes share the same line handling: interactive (rustyline
//! prompt, history, Ctrl-C cancels the running query) and script (statements
//! streamed from a non-tty stdin).

use std::time::Instant;

use arrow_array::{Array, RecordBatch, cast::AsArray};
use arrow_schema::DataType;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use session::{Session, StatementOutcome};
use tokio::signal::unix::Signal;

use crate::sql_split::split_statements;
use crate::{Result, render};

/// Why the shell ended.
pub enum ExitReason {
    /// `\q`, end of input, or end of script.
    Quit,
    /// SIGTERM or SIGHUP; the caller exits nonzero.
    Terminated,
}

/// The shell: an embedded session to run statements on, and the signal streams
/// that give Ctrl-C its cancel meaning.
pub struct Shell<'a> {
    session: &'a Session,
    sigint: &'a mut Signal,
    sigterm: &'a mut Signal,
    sighup: &'a mut Signal,
    timing: bool,
    buffer: String,
}

enum LineOutcome {
    Continue,
    Exit(ExitReason),
}

impl<'a> Shell<'a> {
    pub fn new(
        session: &'a Session,
        sigint: &'a mut Signal,
        sigterm: &'a mut Signal,
        sighup: &'a mut Signal,
    ) -> Self {
        Self {
            session,
            sigint,
            sigterm,
            sighup,
            timing: false,
            buffer: String::new(),
        }
    }

    /// Run the interactive prompt until `\q`, end of input, or a terminating
    /// signal.
    pub async fn run_interactive(&mut self) -> Result<ExitReason> {
        let mut editor = Some(DefaultEditor::new()?);
        let history_path = std::env::var_os("HOME")
            .map(|home| std::path::PathBuf::from(home).join(".pivot_history"));
        if let (Some(editor), Some(path)) = (editor.as_mut(), &history_path) {
            let _ = editor.load_history(path);
        }

        let reason = loop {
            let prompt = if self.buffer.is_empty() {
                "pivot=> "
            } else {
                "pivot-> "
            };
            // rustyline blocks the thread it runs on, so it reads on the
            // blocking pool while this task stays free to see signals.
            let mut owned = editor.take().expect("editor is returned every iteration");
            let mut reading = tokio::task::spawn_blocking(move || {
                let line = owned.readline(prompt);
                (owned, line)
            });
            let line = tokio::select! {
                joined = &mut reading => {
                    let (owned, line) = joined?;
                    editor = Some(owned);
                    line
                }
                _ = self.sigterm.recv() => break ExitReason::Terminated,
                _ = self.sighup.recv() => break ExitReason::Terminated,
            };
            match line {
                Ok(line) => {
                    if !line.trim().is_empty() {
                        let _ = editor
                            .as_mut()
                            .expect("returned above")
                            .add_history_entry(line.as_str());
                    }
                    match self.feed_line(&line).await {
                        LineOutcome::Continue => {}
                        LineOutcome::Exit(reason) => break reason,
                    }
                }
                // Ctrl-C at the prompt abandons the statement being typed,
                // as in psql.
                Err(ReadlineError::Interrupted) => self.buffer.clear(),
                // Ctrl-D.
                Err(ReadlineError::Eof) => break ExitReason::Quit,
                Err(e) => return Err(e.into()),
            }
        };
        if let (Some(editor), Some(path)) = (editor.as_mut(), &history_path) {
            let _ = editor.save_history(path);
        }
        Ok(reason)
    }

    /// Run `script` (stdin's whole content) statement by statement. Errors
    /// print and the script continues, psql's default. A final statement
    /// without a terminating semicolon still runs, as at psql's end of input.
    pub async fn run_script(&mut self, script: &str) -> ExitReason {
        for line in script.lines() {
            match self.feed_line(line).await {
                LineOutcome::Continue => {}
                LineOutcome::Exit(reason) => return reason,
            }
        }
        let tail = std::mem::take(&mut self.buffer);
        if !tail.trim().is_empty() {
            self.run_statement(tail.trim()).await;
        }
        ExitReason::Quit
    }

    /// Consume one input line: a backslash command or a bare `quit`/`exit`/
    /// `help` when one starts a fresh statement, otherwise SQL accumulated
    /// until its terminating semicolon.
    async fn feed_line(&mut self, line: &str) -> LineOutcome {
        let trimmed = line.trim();
        // As in psql, the bare words may carry one trailing semicolon:
        // `quit;` quits rather than parsing as SQL.
        let word = trimmed
            .strip_suffix(';')
            .map(str::trim_end)
            .unwrap_or(trimmed);
        let asks_to_quit = word.eq_ignore_ascii_case("quit") || word.eq_ignore_ascii_case("exit");
        if self.buffer.is_empty() {
            if asks_to_quit {
                return LineOutcome::Exit(ExitReason::Quit);
            }
            if word.eq_ignore_ascii_case("help") {
                print_help();
                return LineOutcome::Continue;
            }
        } else if asks_to_quit {
            // Mid-statement, as in psql: point at \q rather than feed a bare
            // `quit` into the SQL being typed.
            eprintln!("Use \\q to quit.");
            return LineOutcome::Continue;
        }
        if self.buffer.is_empty() && trimmed.starts_with('\\') {
            return self.run_meta_command(trimmed);
        }
        if !self.buffer.is_empty() {
            self.buffer.push('\n');
        }
        self.buffer.push_str(line);
        let (statements, rest) = split_statements(&self.buffer);
        self.buffer = rest;
        for statement in statements {
            self.run_statement(&statement).await;
        }
        LineOutcome::Continue
    }

    fn run_meta_command(&mut self, command: &str) -> LineOutcome {
        let mut words = command.split_whitespace();
        let name = words.next().unwrap_or_default();
        let argument = words.next();
        match name {
            "\\q" => return LineOutcome::Exit(ExitReason::Quit),
            "\\h" | "\\help" | "\\?" => print_help(),
            "\\timing" => {
                self.timing = match argument {
                    None => !self.timing,
                    Some(value) => value.eq_ignore_ascii_case("on"),
                };
                println!("Timing is {}.", if self.timing { "on" } else { "off" });
            }
            _ => {
                eprintln!("invalid command {name}");
                eprintln!("Try \\h for help.");
            }
        }
        LineOutcome::Continue
    }

    /// Run one statement and print its outcome. Errors print as psql's
    /// `ERROR:` line and the shell continues.
    async fn run_statement(&mut self, statement: &str) {
        let started = Instant::now();
        let run = self.session.run(statement);
        tokio::pin!(run);
        let result = tokio::select! {
            result = &mut run => Some(result),
            // Ctrl-C mid-query: dropping the run future fires the session's
            // cancel guard, which stops the dataflow on the workers.
            _ = self.sigint.recv() => None,
        };
        let elapsed = started.elapsed();
        match result {
            None => eprintln!("ERROR:  canceling statement due to user request"),
            Some(Ok(outcome)) => {
                if let Err(message) = print_outcome(outcome) {
                    eprintln!("ERROR:  {message}");
                    return;
                }
                if self.timing {
                    println!("Time: {:.3} ms", elapsed.as_secs_f64() * 1e3);
                }
            }
            Some(Err(message)) => eprintln!("ERROR:  {message}"),
        }
    }
}

fn print_help() {
    println!(
        "Statements end with a semicolon and may span lines.\n\
         \n\
         \\h or \\help       show this help (a bare `help` works too)\n\
         \\q                quit (`quit`, `exit`, or Ctrl-D work too)\n\
         \\timing [on|off]  report each statement's time; toggles without an argument\n\
         \n\
         Ctrl-C cancels the running statement, or clears the line being typed."
    );
}

/// Print one statement's outcome: a result set as an aligned table, a row-less
/// completion as its command tag.
fn print_outcome(outcome: StatementOutcome) -> Result<(), String> {
    match outcome {
        StatementOutcome::Affected(count) => println!("INSERT 0 {count}"),
        StatementOutcome::Command(tag) => println!("{tag}"),
        StatementOutcome::Rows {
            column_names,
            batches,
        } => {
            let mut rows = Vec::new();
            for batch in &batches {
                append_text_rows(&mut rows, batch)?;
            }
            print!("{}", render::format_table(&column_names, &rows));
        }
    }
    Ok(())
}

/// Append `batch`'s rows as text, the shape the renderer takes. Every column
/// casts to Utf8; NULLs stay `None`.
fn append_text_rows(
    rows: &mut Vec<Vec<Option<String>>>,
    batch: &RecordBatch,
) -> Result<(), String> {
    let text_columns = batch
        .columns()
        .iter()
        .map(|column| {
            arrow_cast::cast(column, &DataType::Utf8)
                .map_err(|e| format!("cannot render a {} column as text: {e}", column.data_type()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    for i in 0..batch.num_rows() {
        rows.push(
            text_columns
                .iter()
                .map(|column| {
                    let values = column.as_string::<i32>();
                    (!values.is_null(i)).then(|| values.value(i).to_string())
                })
                .collect(),
        );
    }
    Ok(())
}
