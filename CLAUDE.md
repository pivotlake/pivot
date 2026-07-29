If you're in a worktree (check!) make sure you're editing files IN YOUR WORKTREE and not the main repo.

All tests should be blackbox and hopefully SHORT, styled Setup/Execute/Assert (each section separated by a new line, no need to literally write "setup" etc)

Never mention benchmarks (such as ClickBench) within library code/tests (also comments)

Use descriptive variable names and function names; avoid ad-hoc abbreviations like `g_ty` for `group_type`.
Established short names are fine: `i`/`j` for loop indices, `ctx`, `idx`, `len`.

Function names should usually begin with a verb (e.g. `parse_header`, not `header`).

When adding comments, don't reference old code (the user reading the new code has no idea what you're talking about) or
things that are extremely mission-specific (for example, if changing GROUP-BY to optimize a query in a benchmark, do NOT mention the benchmark in comments- your code is generic!)

Don't return None or have fallbacks when it's not absolutely necessary. We don't want to have silent 
failures or have unexpected flows.

Never push unformatted code. Before every `git push`, run `cargo fmt` and ensure `cargo fmt --check` passes.

## Never use pgrep to decide whether a job is alive

`pgrep -f PATTERN` scans full command lines, and the shell running it has
PATTERN in its own command line, so it matches itself. The check returns a hit
even when nothing is running, and it fails in the direction that looks healthy:
"still going" when the job died.