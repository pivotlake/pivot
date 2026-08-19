/**
 * Syntax colours for code blocks. Expressive Code drives highlighting through
 * a Shiki theme, so these cannot be set as CSS variables the way the rest of
 * the palette is: they have to be a theme object handed to the integration.
 *
 * Both a dark and a light theme are required. Given one theme, Starlight
 * scopes the generated CSS to `[data-theme='<theme name>']`, a selector the
 * page never sets, and none of these colours reach the reader.
 */

const KEYWORD = "#e5a0e0";
const STRING = "#c8d97a";
const NUMBER = "#e0b070";
const COMMENT = "#6a6a6a";

const KEYWORD_SCOPES = [
  "keyword",
  "keyword.control",
  "keyword.operator.new",
  "keyword.other",
  "storage",
  "storage.type",
  "storage.modifier",
  "variable.language",
  "entity.name.tag",
];

const STRING_SCOPES = [
  "string",
  "string.quoted",
  "string.template",
  "punctuation.definition.string",
  "meta.attribute string",
];

const NUMBER_SCOPES = [
  "constant.numeric",
  "constant.language",
  "constant.character.escape",
  "constant.other",
];

const COMMENT_SCOPES = ["comment", "punctuation.definition.comment"];

export const codeThemeDark = {
  name: "pivotdb-dark",
  type: "dark",
  colors: {
    "editor.background": "#151515",
    "editor.foreground": "#b9b9b9",
  },
  tokenColors: [
    { scope: KEYWORD_SCOPES, settings: { foreground: KEYWORD } },
    { scope: STRING_SCOPES, settings: { foreground: STRING } },
    { scope: NUMBER_SCOPES, settings: { foreground: NUMBER } },
    { scope: COMMENT_SCOPES, settings: { foreground: COMMENT } },
  ],
};

/** The same hues, darkened to stay legible on a light background. */
export const codeThemeLight = {
  name: "pivotdb-light",
  type: "light",
  colors: {
    "editor.background": "#eae8e3",
    "editor.foreground": "#4a4b52",
  },
  tokenColors: [
    { scope: KEYWORD_SCOPES, settings: { foreground: "#8c3186" } },
    { scope: STRING_SCOPES, settings: { foreground: "#5c6b1c" } },
    { scope: NUMBER_SCOPES, settings: { foreground: "#8f5f14" } },
    { scope: COMMENT_SCOPES, settings: { foreground: "#8a877f" } },
  ],
};

