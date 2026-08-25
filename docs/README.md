# docs

The pivotdb documentation site: markdown files rendered by
[Astro Starlight](https://starlight.astro.build/) into a static bundle.

Writing a page means adding a markdown file. Everything else (sidebar, table of
contents, search index, dark/light mode, mobile nav) is generated.

## Add a page

Create a `.md` file under `src/content/docs/` with frontmatter:

```md
---
title: Compaction
description: How small files are merged.
sidebar:
  order: 3
---

Body text.
```

A directory under `src/content/docs/` becomes a sidebar group once it is
listed in the `sidebar` array of `astro.config.mjs`. Within a group, pages are
ordered by `sidebar.order`, then alphabetically.

## Develop

Requires Node 22.12 or newer.

```sh
npm --prefix docs install
npm --prefix docs run dev      # http://localhost:4321/docs
```

The dev server hot-reloads on every markdown save.

## Build

```sh
npm --prefix docs run build    # writes docs/dist/
npm --prefix docs run preview  # serve the built bundle
```

`dist/` is a plain static directory. It can be uploaded to any static host, or
served by `pivot server` the same way `web/frontend/dist/` is embedded today.

## After changing `astro.config.mjs`

Rendered markdown is cached in `.astro/`, `node_modules/.astro` and
`node_modules/.vite`, and a change to the config does not always invalidate it.
A config edit that appears to do nothing usually just needs those cleared:

```sh
rm -rf docs/.astro docs/node_modules/.astro docs/node_modules/.vite docs/dist
npm --prefix docs run build
```

Editing markdown or CSS does not need this.

## Theme

`src/styles/theme.css` holds the whole visual design as overrides of
Starlight's CSS variables: JetBrains Mono for body text, Space Grotesk for
headings, a slate palette, hairline borders and square corners. Nothing else
in the site is styled by hand.

## Serving path

`astro.config.mjs` sets `base: "/docs"`, so the site expects to live under
`example.com/docs/`. Drop that line to serve it from the domain root.
