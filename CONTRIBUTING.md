# Contributing to Pivot

Thanks for your interest in Pivot! This guide covers how to report problems,
propose changes, and get a pull request merged.

## Questions

Ask usage and configuration questions in
[GitHub Discussions](https://github.com/pivotlake/pivot/discussions) rather
than in issues.

## Reporting bugs

- Search the [existing issues](https://github.com/pivotlake/pivot/issues)
  first; someone may already have reported it.
- Otherwise, [open a new one](https://github.com/pivotlake/pivot/issues/new/choose)
  with the smallest reproduction you can find: the SQL, the table definition,
  the configuration, and what you expected instead of what happened.

## Before you write code

- For anything beyond a small bug fix, open an issue or a discussion first so
  we can agree on the approach before you invest time in it. A pull request for
  a feature nobody has discussed may be closed.

## Use of AI tools

You may use AI assistants and agents to help write, refactor or debug your
contribution. However, **you must read and fully understand every line of code
you submit**. By opening a pull request you confirm that you can explain each
change, why it is correct, and how it was tested, and that you stand behind it
as your own work. Pull requests that look generated without that
understanding will be closed without review.

## Testing and checks

`./ci.sh` runs exactly what CI runs: `cargo fmt --check`, `clippy`, the
tests and the docs build, on every crate.

```sh
./ci.sh                  # every check on every crate
./ci.sh test planner     # one check on one crate
```

Every pull request must pass CI. If you change behavior, add tests.

## Code style

- Run `cargo fmt` before every push.
- Use descriptive names and avoid ad-hoc abbreviations (`group_type`, not
  `g_ty`); established short names such as `i`, `ctx`, `idx` and `len` are
  fine. Function names usually start with a verb (`parse_header`, not
  `header`).
- Don't hide failures: avoid fallbacks and `None`s that silently swallow an
  error or an unexpected state.

## Commits and pull requests

- Commit messages are a single line, prefixed with the area they touch, for
  example `planner: push LIMIT below a projection` or
  `ci: run pull requests from forks on GitHub-hosted runners`.
- Describe in the pull request what the change does, why, and how you tested
  it, and link the issue it addresses.

## Documentation

User-facing documentation lives in [`docs/`](docs/). If your change adds or
alters user-visible behavior, update the relevant page in the same pull
request. See [`docs/README.md`](docs/README.md) for how to preview the site.

## License

Pivot is licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE)
or the [MIT license](LICENSE-MIT), at your option. Unless you explicitly state
otherwise, any contribution you intentionally submit for inclusion in Pivot,
as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
