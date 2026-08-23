import { defineRouteMiddleware } from "@astrojs/starlight/route-data";

// Starlight only reads a banner from a page's own frontmatter. Setting it here
// puts the same one on every page, so a reader who arrives from a search
// result rather than from the introduction still sees it. A page that sets its
// own banner keeps it.
export const onRequest = defineRouteMiddleware((context) => {
  const { entry } = context.locals.starlightRoute;
  entry.data.banner ??= {
    content:
      'Pivot is in early development and is <strong>not production ready</strong>. See <a href="/docs/#project-status">Project status</a>.',
  };
});
