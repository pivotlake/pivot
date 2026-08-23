// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { codeThemeDark, codeThemeLight } from "./src/styles/code-theme.mjs";

export default defineConfig({
  // Used for canonical URLs and the sitemap. Change once the domain is fixed.
  site: "https://pivotdb.dev",
  base: "/docs",
  integrations: [
    starlight({
      title: "pivotdb",
      description: "Documentation for pivotdb, a columnar analytics engine.",
      customCss: ["./src/styles/theme.css"],
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
