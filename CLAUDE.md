If you're in a worktree (check!) make sure you're editing files IN YOUR WORKTREE and not the main repo.

All tests should be blackbox and hopefully SHORT, styled Setup/Execute/Assert (each section separated by a new line, no need to literally write "setup" etc)

Don't use em dash ANYWHERE

Never mention benchmarks (such as ClickBench) within library code/tests (also comments)

Use descriptive variable names and function names; avoid ad-hoc abbreviations like `g_ty` for `group_type`.
Established short names are fine: `i`/`j` for loop indices, `ctx`, `idx`, `len`.

Function names should usually begin with a verb (e.g. `parse_header`, not `header`).
