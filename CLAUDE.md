If you're in a worktree (check!) make sure you're editing files IN YOUR WORKTREE and not the main repo.

All tests should be blackbox and hopefully SHORT, styled Setup/Execute/Assert (each section separated by a new line, no need to literally write "setup" etc)

Never mention benchmarks (such as ClickBench) within library code/tests (also comments)