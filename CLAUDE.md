If you're in a worktree (check!) make sure you're editing files IN YOUR WORKTREE and not the main repo.

 (replies, code, comments, commit messages).

## Talking to me

Keep replies short. Answer what I asked, and don't restate the plan, list options I
didn't ask for, or re-explain something you already explained.

When I say "explain to human" or "explain simply", it means the last answer was too
low level. Drop the jargon, the type names and the code excerpts, and explain the
mechanism in plain prose.

When I ask "why did you add X?", the honest answer is often "it isn't needed". Say
that instead of defending it.

## Git

NEVER put AI attribution in a commit: no "Generated with Claude", no
"Co-Authored-By: Claude", no "Claude-Session:" trailer, nothing anywhere in the
message or its body that says a model wrote it.

Commit subject is `crate, crate: lowercase imperative summary`, for example
`datastore-delta: configure the Delta Kernel engine from the datastore's store`.
Default to a subject and NO body. Write a body only for a change big enough that a
reviewer needs the reasoning, never as a changelog of what you did.

One logical change per commit. When I ask for a follow-up fix to code introduced
earlier on this same branch, fix up the commit that introduced it so the history
stays clean; when I ask for something genuinely new, add a new commit. Say which
one you're doing.

Only push when I say push. Push over git from this machine, not through the GitHub
API, so the author and committer are both me.

Before every push, run `cargo fmt` and make sure `cargo fmt --check` passes.

I review in VS Code by reading the working tree, so "unstage the last commit so I
can see it" means `git reset HEAD~1`: keep every change, just take it out of the
commit.

## Verifying

`./ci.sh` is the single source of truth for checks (fmt, clippy, test, doc over
every crate) and CI runs the same cells, so `./ci.sh <check> <crate>` locally is
exactly what CI will run. Clippy runs with `-D warnings`, so a warning fails CI.

Run the checks for the crates you touched before saying something is done, and
never report a build, test or push as finished on output you didn't actually see.
If it failed, say it failed and show the output.

## The build cache

Local builds are wrapped by kache (`rustc-wrapper` in `.cargo/config.toml`), and
its key covers the Rust inputs. After changing C++ under `duckdb-planner`, the
final link can come back from cache, so the change appears to do nothing at
runtime. When a C++ change seems to have no effect, rebuild that crate with
`RUSTC_WRAPPER=` to bypass the cache before hunting for a logic bug.

## Code

Use descriptive variable names and function names; avoid ad-hoc abbreviations like
`g_ty` for `group_type`. Established short names are fine: `i`/`j` for loop indices,
`ctx`, `idx`, `len`.

Function names should usually begin with a verb (e.g. `parse_header`, not `header`).

A name has to describe what the thing does now. When behaviour moves, rename with
it: something that encodes and then uploads isn't `upload_spec`. Never leave two
names in the tree that differ only by word order, like `cas_commit` next to
`commit_cas`.

Don't return None or have fallbacks when it's not absolutely necessary. We don't
want to have silent failures or have unexpected flows.

Configuration is explicit or it's an error. No default region, no default datastore
name, no falling back to environment variables for credentials.

Prefer one generic path over two parallel ones. Two structs, enum variants or
functions that differ only by a field or a verb should be unified, with the
difference carried as data.

Delete a parameter or field instead of passing it and ignoring it, and delete code
whose only callers are tests. If a constructor or accessor has no production caller,
remove it and let the tests go through the real path.

No `as_any` or downcasting escape hatches on traits. If a caller needs the concrete
type, fix the trait or the ownership.

Errors are `thiserror` variants carrying their source, not `format!`-ed strings.

Fix at the right depth. A special case bolted onto shared infrastructure usually
means the underlying mechanism should be generalized instead.

## Tests

All tests should be blackbox and hopefully SHORT, styled Setup/Execute/Assert (each
section separated by a new line, no need to literally write "setup" etc)

Put a new test where the tests for that same kind of thing already live.

## Comments

When adding comments, don't reference old code (the user reading the new code has no
idea what you're talking about) or things that are extremely mission-specific (for
example, if changing GROUP-BY to optimize a query in a benchmark, do NOT mention the
benchmark in comments, your code is generic!)

Never mention benchmarks (such as ClickBench) within library code/tests (also
comments)

Put a comment directly above the line it explains. Don't hoist it into the
function's doc comment, where the reader can't tell what it refers to.

A comment carries the why. Don't restate what the code already says, don't justify a
branch that shouldn't exist, and don't narrate what changed.

I'm picky about comment wording. When a comment matters and is hard to phrase, use
the `ask-codex` skill rather than shipping a first draft.

## Never use pgrep to decide whether a job is alive

`pgrep -f PATTERN` scans full command lines, and the shell running it has
PATTERN in its own command line, so it matches itself. The check returns a hit
even when nothing is running, and it fails in the direction that looks healthy:
"still going" when the job died.
