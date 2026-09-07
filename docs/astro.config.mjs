// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { codeThemeDark, codeThemeLight } from "./src/styles/code-theme.mjs";

export default defineConfig({
  // Used for canonical URLs and the sitemap.
  site: "https://pivotlake.io",
  base: "/docs",
  integrations: [
    starlight({
      title: "pivotdb",
      description: "Documentation for pivotdb, a columnar analytics engine.",
      favicon: "/pivot-favicon.png",
      customCss: ["./src/styles/theme.css"],
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 4 },
      // Adds the shared pre-release banner and nests the table of contents
      // beneath the page title.
      routeMiddleware: "./src/routeData.ts",
      expressiveCode: {
        themes: [codeThemeDark, codeThemeLight],
        // Pointed at CSS variables rather than literal colours so the code
        // block surfaces follow the site theme. Literals here apply to both
        // themes at once, which leaks the dark palette into light mode.
        styleOverrides: {
          borderColor: "var(--docs-code-border)",
          borderRadius: "2px",
          codeBackground: "var(--docs-code-bg)",
          frames: {
            terminalBackground: "var(--docs-code-bg)",
            terminalTitlebarBackground: "var(--docs-code-titlebar-bg)",
            terminalTitlebarBorderBottomColor: "var(--docs-code-border)",
            terminalTitlebarForeground: "var(--docs-code-titlebar-fg)",
            editorTabBarBackground: "var(--docs-code-titlebar-bg)",
            editorActiveTabBackground: "var(--docs-code-bg)",
            editorActiveTabIndicatorTopColor: "var(--sl-color-text-accent)",
            editorTabBarBorderBottomColor: "var(--docs-code-border)",
            inlineButtonForeground: "var(--sl-color-gray-2)",
          },
        },
      },
      components: {
        // Opens light for a first-time reader instead of following the OS.
        ThemeProvider: "./src/components/ThemeProvider.astro",
        // Draws the wordmark with its cursor as a separate element.
        SiteTitle: "./src/components/SiteTitle.astro",
        // Leaves the light default alone until the reader picks a theme.
        ThemeSelect: "./src/components/ThemeSelect.astro",
      },
      head: [
        {
          tag: "link",
          attrs: { rel: "preconnect", href: "https://fonts.googleapis.com" },
        },
        {
          tag: "link",
          attrs: {
            rel: "preconnect",
            href: "https://fonts.gstatic.com",
            crossorigin: true,
          },
        },
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap",
          },
        },
        // Privacy-friendly analytics by Plausible.
        {
          tag: "script",
          attrs: {
            async: true,
            src: "https://plausible.io/js/pa-mzBvHvq0jpw3-SL6-9gRy.js",
          },
        },
        {
          tag: "script",
          content:
            "window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)},plausible.init=plausible.init||function(i){plausible.o=i||{}};plausible.init()",
        },
      ],
      // Every entry is a directory under src/content/docs/. Pages order
      // themselves by the `sidebar.order` field in their frontmatter.
      sidebar: [
        { label: "Introduction", slug: "index" },
        { label: "Quickstart", slug: "quickstart" },
        {
          label: "Database",
          items: [{ autogenerate: { directory: "database" } }],
        },
        { label: "Roadmap", slug: "roadmap" },
        {
          label: "Use cases",
          items: [{ autogenerate: { directory: "use-cases" } }],
        },
        {
          label: "Reference",
          items: [{ autogenerate: { directory: "reference" } }],
        },
      ],
      pagination: true,
      lastUpdated: true,
    }),
  ],
});
