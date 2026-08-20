use std::io::{self, IsTerminal, Write};
use std::time::Instant;

use engine::{ExecuteOptions, StatementOutput};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::ShellInstance;
use crate::parser::split_complete;
use crate::render::{TextBatch, render_table};

const HELP: &str = "\
Shell commands:
  \\q                 quit
  quit;              quit
  \\timing            toggle statement timing
  \\timing on|off     set statement timing
  \\h, \\help         show this help
";

pub(crate) async fn run_shell(
    datastore_location: String,
) -> Result<(), Box<dyn std::error::Error>> {
    if !io::stdin().is_terminal() {
        return Err("an interactive terminal is required".into());
    }

    let instance = ShellInstance::open(&datastore_location)?;
    let editor = DefaultEditor::new()?;

    println!("pivot shell ({})", env!("CARGO_PKG_VERSION"));
    println!("Type \"\\help\" for help.");

    run_repl(instance.engine(), editor).await
}

async fn run_repl(
    engine: &engine::Engine,
    mut editor: DefaultEditor,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer = String::new();
    let mut timing = false;
    loop {
        let prompt = if buffer.is_empty() {
            "pivot=> "
        } else {
            "pivot-> "
        };
        let (returned_editor, readline) = read_line(editor, prompt).await?;
        editor = returned_editor;
        let line = match readline {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                buffer.clear();
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(error) => return Err(error.into()),
        };

        if let Some(command) = parse_meta_command(&line) {
            let _ = editor.add_history_entry(line.clone());
            if handle_meta_command(command, &mut timing) {
                break;
            }
            continue;
        }

        if !buffer.is_empty() {
            buffer.push('\n');
        }
        buffer.push_str(&line);
        let ParsedBuffer {
            statements,
            remainder,
            trailing_meta_command,
        } = parse_buffer(&buffer);
        buffer = remainder;

        let mut failed = false;
        for statement in statements {
            let _ = editor.add_history_entry(statement.clone());
            if is_quit_statement(&statement) {
                return Ok(());
            }
            if !execute_statement(engine, statement, timing).await {
                failed = true;
                break;
            }
        }
        if failed {
            buffer.clear();
        }
        if let Some((input, command)) = trailing_meta_command {
            let _ = editor.add_history_entry(input);
            if handle_meta_command(command, &mut timing) {
                break;
            }
        }
    }
    Ok(())
}

async fn read_line(
    mut editor: DefaultEditor,
    prompt: &'static str,
) -> Result<(DefaultEditor, rustyline::Result<String>), tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || {
        let result = editor.readline(prompt);
        (editor, result)
    })
    .await
}

#[derive(Debug, PartialEq, Eq)]
enum MetaCommand {
    Quit,
    Help,
    Timing(Option<bool>),
    Invalid(String),
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedBuffer {
    statements: Vec<String>,
    remainder: String,
    trailing_meta_command: Option<(String, MetaCommand)>,
}

fn parse_buffer(buffer: &str) -> ParsedBuffer {
    let split = split_complete(buffer);
    let mut remainder = split.remainder;
    let trailing_meta_command = parse_meta_command(&remainder).map(|command| {
        let input = std::mem::take(&mut remainder);
        (input, command)
    });
    ParsedBuffer {
        statements: split.complete,
        remainder,
        trailing_meta_command,
    }
}

fn parse_meta_command(line: &str) -> Option<MetaCommand> {
    let trimmed = line.trim();
    if !trimmed.starts_with('\\') {
        return None;
    }
    let mut words = trimmed.split_whitespace();
    let command = words.next().unwrap();
    let arguments = words.collect::<Vec<_>>();
    Some(match (command, arguments.as_slice()) {
        ("\\q", []) => MetaCommand::Quit,
        ("\\h" | "\\help", []) => MetaCommand::Help,
        ("\\timing", []) => MetaCommand::Timing(None),
        ("\\timing", [value]) if is_on(value) => MetaCommand::Timing(Some(true)),
        ("\\timing", [value]) if is_off(value) => MetaCommand::Timing(Some(false)),
        _ => MetaCommand::Invalid(trimmed.to_string()),
    })
}

/// Execute a meta-command and return whether the REPL should exit.
fn handle_meta_command(command: MetaCommand, timing: &mut bool) -> bool {
    match command {
        MetaCommand::Quit => true,
        MetaCommand::Help => {
            print!("{HELP}");
            false
        }
        MetaCommand::Timing(value) => {
            *timing = value.unwrap_or(!*timing);
            println!("Timing is {}.", if *timing { "on" } else { "off" });
            false
        }
        MetaCommand::Invalid(command) => {
            eprintln!("invalid command {command}");
            false
        }
    }
}

fn is_on(value: &str) -> bool {
    matches!(value.to_ascii_lowercase().as_str(), "on" | "true" | "1")
}

fn is_off(value: &str) -> bool {
    matches!(value.to_ascii_lowercase().as_str(), "off" | "false" | "0")
}

fn is_quit_statement(statement: &str) -> bool {
    statement
        .strip_suffix(';')
        .is_some_and(|command| command.trim().eq_ignore_ascii_case("quit"))
}

async fn execute_statement(engine: &engine::Engine, sql: String, timing: bool) -> bool {
    let started = Instant::now();
    let succeeded = tokio::select! {
        result = engine.execute_with_output::<TextBatch>(sql, ExecuteOptions::default()) => match result {
            Ok(execution) => {
                if let Err(error) = display_output(execution.output) {
                    eprintln!("pivot: writing output: {error}");
                    false
                } else {
                    true
                }
            }
            Err(error) => {
                eprintln!("ERROR:  {error}");
                false
            }
        },
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => {
                    eprintln!("Cancel request sent");
                    eprintln!("ERROR:  query canceled");
                }
                Err(error) => eprintln!("pivot: failed to listen for Ctrl-C: {error}"),
            }
            false
        }
    };

    if timing {
        println!("Time: {:.3} ms", started.elapsed().as_secs_f64() * 1e3);
    }
    succeeded
}

fn display_output(output: StatementOutput<TextBatch>) -> io::Result<()> {
    let rendered = match output {
        StatementOutput::Rows { columns, batches } => render_table(&columns, batches),
        StatementOutput::Command(command) => format!("{command}\n"),
        StatementOutput::Set { value, .. } => {
            if value.is_some() { "SET\n" } else { "RESET\n" }.to_string()
        }
        StatementOutput::CopyFromStdin(ingest) => {
            // The shell has no copy-in channel; dropping the running ingest
            // aborts it and rolls the statement's transaction back.
            drop(ingest);
            "COPY FROM STDIN is not supported in the shell\n".to_string()
        }
    };
    let mut stdout = io::stdout().lock();
    stdout.write_all(rendered.as_bytes())?;
    stdout.flush()
}

#[cfg(test)]
mod tests {
    use super::{MetaCommand, ParsedBuffer, is_quit_statement, parse_buffer, parse_meta_command};

    #[test]
    fn parses_supported_meta_commands() {
        let quit = parse_meta_command("  \\q ");
        let short_help = parse_meta_command("\\h");
        let help = parse_meta_command("\\help");
        let toggle = parse_meta_command("\\timing");
        let off = parse_meta_command("\\timing OFF");

        assert_eq!(quit, Some(MetaCommand::Quit));
        assert_eq!(short_help, Some(MetaCommand::Help));
        assert_eq!(help, Some(MetaCommand::Help));
        assert_eq!(toggle, Some(MetaCommand::Timing(None)));
        assert_eq!(off, Some(MetaCommand::Timing(Some(false))));
    }

    #[test]
    fn rejects_meta_command_arguments_and_unknown_commands() {
        let quit_with_argument = parse_meta_command("\\q now");
        let unknown = parse_meta_command("\\dt");

        assert!(matches!(quit_with_argument, Some(MetaCommand::Invalid(_))));
        assert!(matches!(unknown, Some(MetaCommand::Invalid(_))));
    }

    #[test]
    fn parses_a_meta_command_after_complete_sql_on_the_same_line() {
        assert_eq!(
            parse_buffer("SELECT 1; \\timing"),
            ParsedBuffer {
                statements: vec!["SELECT 1;".to_string()],
                remainder: String::new(),
                trailing_meta_command: Some(("\\timing".to_string(), MetaCommand::Timing(None),)),
            }
        );
    }

    #[test]
    fn recognizes_quit_as_a_complete_statement() {
        assert!(is_quit_statement("quit;"));
        assert!(is_quit_statement("QUIT ;"));
        assert!(!is_quit_statement("quit"));
        assert!(!is_quit_statement("SELECT 'quit;';"));
    }
}
