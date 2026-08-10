//! psql-style rendering of query results.
//!
//! The aligned format psql calls `aligned`: one space of padding around each
//! cell, headers centered, a `----+----` rule under them, and a `(N rows)`
//! footer. The simple query protocol carries every value as text with no type
//! information, so a column is right-aligned when all of its values read as
//! numbers, which matches psql's numeric alignment in practice.

/// Render one result set. `rows` hold text values; `None` is a SQL NULL and
/// renders empty, as psql does by default.
pub fn format_table(columns: &[String], rows: &[Vec<Option<String>>]) -> String {
    let widths: Vec<usize> = columns
        .iter()
        .enumerate()
        .map(|(i, name)| {
            rows.iter()
                .map(|row| cell_text(row, i).chars().count())
                .chain([name.chars().count()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let right_aligned: Vec<bool> = (0..columns.len())
        .map(|i| {
            let mut values = rows
                .iter()
                .map(|row| cell_text(row, i))
                .filter(|v| !v.is_empty());
            let mut any = false;
            let all_numeric = values.all(|v| {
                any = true;
                v.parse::<f64>().is_ok()
            });
            any && all_numeric
        })
        .collect();

    let mut out = String::new();
    let header: Vec<String> = columns
        .iter()
        .zip(&widths)
        .map(|(name, width)| center(name, *width))
        .collect();
    push_row(&mut out, &header);
    out.push_str(
        &widths
            .iter()
            .map(|w| "-".repeat(w + 2))
            .collect::<Vec<_>>()
            .join("+"),
    );
    out.push('\n');
    for row in rows {
        let cells: Vec<String> = widths
            .iter()
            .enumerate()
            .map(|(i, width)| {
                let value = cell_text(row, i);
                if right_aligned[i] {
                    format!("{value:>width$}")
                } else {
                    format!("{value:<width$}")
                }
            })
            .collect();
        push_row(&mut out, &cells);
    }
    let count = rows.len();
    let noun = if count == 1 { "row" } else { "rows" };
    out.push_str(&format!("({count} {noun})\n"));
    out
}

fn cell_text(row: &[Option<String>], i: usize) -> &str {
    row.get(i).and_then(|v| v.as_deref()).unwrap_or("")
}

/// One ` cell | cell ` line, without the trailing padding psql also omits.
fn push_row(out: &mut String, cells: &[String]) {
    out.push(' ');
    out.push_str(&cells.join(" | "));
    while out.ends_with(' ') {
        out.pop();
    }
    out.push('\n');
}

fn center(text: &str, width: usize) -> String {
    let len = text.chars().count();
    let left = width.saturating_sub(len) / 2;
    let right = width.saturating_sub(len) - left;
    format!("{}{text}{}", " ".repeat(left), " ".repeat(right))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[Option<&str>]) -> Vec<Option<String>> {
        values.iter().map(|v| v.map(str::to_string)).collect()
    }

    #[test]
    fn output_matches_psql_aligned_format() {
        let columns = vec!["id".to_string(), "name".to_string()];
        let rows = vec![row(&[Some("7"), Some("first")])];

        let table = format_table(&columns, &rows);

        assert_eq!(table, " id | name\n----+-------\n  7 | first\n(1 row)\n");
    }

    #[test]
    fn text_columns_left_align_and_the_footer_pluralises() {
        let columns = vec!["name".to_string()];
        let rows = vec![row(&[Some("a")]), row(&[Some("longer")])];

        let table = format_table(&columns, &rows);

        assert_eq!(table, "  name\n--------\n a\n longer\n(2 rows)\n");
    }

    #[test]
    fn a_null_renders_empty_without_breaking_numeric_alignment() {
        let columns = vec!["n".to_string()];
        let rows = vec![row(&[Some("10")]), row(&[None])];

        let table = format_table(&columns, &rows);

        assert_eq!(table, " n\n----\n 10\n\n(2 rows)\n");
    }

    #[test]
    fn zero_rows_still_print_the_header_and_footer() {
        let columns = vec!["id".to_string()];

        let table = format_table(&columns, &[]);

        assert_eq!(table, " id\n----\n(0 rows)\n");
    }
}
