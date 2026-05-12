// Small SVG spinner — centered in its container, with an optional label
// below. Animation uses the global `perfy-spin` @keyframes declared in
// `styles.css` so we don't need a CSS-in-JS dependency.

interface Props {
  size?: number;
  label?: string;
  /// Stroke colour. Defaults to a soft amber so loaders read as
  /// "in progress" without competing with the rest of the UI.
  color?: string;
  /// When `true` the spinner expands to fill its parent and centres itself.
  /// Defaults to `true`; pass `false` for inline usage.
  fill?: boolean;
}

export function Spinner({
  size = 28,
  label,
  color = "oklch(0.7 0.13 70)",
  fill = true,
}: Props) {
  const stroke = Math.max(2, Math.round(size / 12));
  const r = (size - stroke) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;
  // Show ~25% of the circumference as a moving arc.
  const dashLen = circumference * 0.25;
  const gapLen = circumference - dashLen;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        ...(fill ? { width: "100%", height: "100%", padding: 24 } : {}),
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{
          animation: "perfy-spin 0.9s linear infinite",
          transformOrigin: "50% 50%",
        }}
        aria-label="Loading"
        role="img"
      >
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke="oklch(0.94 0.005 250)"
          strokeWidth={stroke}
        />
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dashLen} ${gapLen}`}
        />
      </svg>
      {label && (
        <span
          style={{
            fontSize: 11.5,
            color: "oklch(0.5 0.01 250)",
            fontStyle: "italic",
            fontFamily:
              '"Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif',
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}
