/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // ── PalmVerse Design System ──────────────────────────────────
      colors: {
        // Core background palette — deep dark charcoal/navy
        void:    "#050508",
        abyss:   "#0a0a12",
        deep:    "#0f0f1a",
        surface: "#141420",
        // Card / elevated surfaces
        card:    "#1a1a2e",
        border:  "#252540",
        // Purple mystical accent palette
        mystic: {
          50:  "#f3e8ff",
          100: "#e9d5ff",
          200: "#d8b4fe",
          300: "#c084fc",
          400: "#a855f7",
          500: "#9333ea",
          600: "#7c3aed",
          700: "#6d28d9",
          800: "#5b21b6",
          900: "#4c1d95",
        },
        // Warm gold accent for highlights
        gold: {
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b",
        },
        // Text hierarchy
        text: {
          primary:   "#f0eeff",
          secondary: "#a89bc2",
          muted:     "#6b6080",
        },
      },

      // ── Typography ────────────────────────────────────────────────
      fontFamily: {
        // Serif for headings — mystical, premium feel
        serif:  ["Cormorant Garamond", "Georgia", "serif"],
        // Sans for body — clean, readable
        sans:   ["Inter", "system-ui", "sans-serif"],
        // Monospace (for code/debug only)
        mono:   ["JetBrains Mono", "monospace"],
      },

      // ── Glow effects via box-shadow ───────────────────────────────
      boxShadow: {
        "glow-sm":  "0 0 12px 2px rgba(147, 51, 234, 0.25)",
        "glow-md":  "0 0 24px 4px rgba(147, 51, 234, 0.35)",
        "glow-lg":  "0 0 48px 8px rgba(147, 51, 234, 0.30)",
        "glow-gold":"0 0 20px 4px rgba(251, 191, 36, 0.25)",
        "card":     "0 4px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05)",
      },

      // ── Background radial gradients ───────────────────────────────
      backgroundImage: {
        "mystic-radial": "radial-gradient(ellipse at 50% 0%, rgba(147,51,234,0.15) 0%, transparent 70%)",
        "mystic-card":   "linear-gradient(135deg, rgba(26,26,46,0.9) 0%, rgba(20,20,32,0.95) 100%)",
        "shimmer":       "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%)",
      },

      // ── Animation durations ───────────────────────────────────────
      transitionDuration: {
        "400": "400ms",
        "600": "600ms",
        "800": "800ms",
      },

      // ── Border radius ─────────────────────────────────────────────
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
        "4xl": "2rem",
      },
    },
  },
  plugins: [],
};
