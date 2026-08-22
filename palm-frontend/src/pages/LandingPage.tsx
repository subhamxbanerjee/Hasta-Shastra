import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Sparkles } from "lucide-react";
import { useEffect, useRef } from "react";

// ── Particle type ─────────────────────────────────────────────────────────────
interface Particle {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  speed: number;
  drift: number;
}

// ── Generate static particles once ───────────────────────────────────────────
function generateParticles(count: number): Particle[] {
  return Array.from({ length: count }, (_, i) => ({
    id:      i,
    x:       Math.random() * 100,
    y:       Math.random() * 100,
    size:    Math.random() * 2.5 + 0.5,
    opacity: Math.random() * 0.5 + 0.15,
    speed:   Math.random() * 20 + 15,
    drift:   (Math.random() - 0.5) * 30,
  }));
}

const PARTICLES = generateParticles(60);

// ── Animated palm SVG ─────────────────────────────────────────────────────────
function PalmGlyph() {
  return (
    <motion.div
      className="relative w-40 h-40 md:w-56 md:h-56 mx-auto mb-10"
      initial={{ opacity: 0, scale: 0.7 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
    >
      {/* Outer glow ring */}
      <motion.div
        className="absolute inset-0 rounded-full"
        style={{ background: "radial-gradient(circle, rgba(147,51,234,0.18) 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.08, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />
      {/* SVG palm hand */}
      <svg
        viewBox="0 0 200 200"
        className="w-full h-full"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Palm silhouette */}
        <path
          d="M100 175 C65 175 40 155 35 120 L30 75 C29 65 36 58 45 60 L47 61 L50 30 C51 20 62 18 66 27 L70 55 L72 25 C73 15 84 14 87 23 L90 58 L92 28 C93 18 104 18 106 28 L108 58 L112 35 C114 25 124 26 124 37 L122 80 C135 72 148 76 148 90 L148 110 C148 145 130 175 100 175Z"
          fill="rgba(147,51,234,0.12)"
          stroke="rgba(192,132,252,0.5)"
          strokeWidth="1.5"
        />
        {/* Life line */}
        <motion.path
          d="M60 145 C65 120 68 95 75 75 C80 60 90 52 100 50"
          stroke="rgba(192,132,252,0.8)"
          strokeWidth="1.8"
          strokeLinecap="round"
          fill="none"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 2, delay: 0.8, ease: "easeInOut" }}
        />
        {/* Head line */}
        <motion.path
          d="M62 110 C75 108 90 107 105 106 C118 105 130 100 138 95"
          stroke="rgba(216,180,254,0.7)"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 2, delay: 1.2, ease: "easeInOut" }}
        />
        {/* Heart line */}
        <motion.path
          d="M58 88 C72 82 90 80 108 78 C122 76 134 72 140 68"
          stroke="rgba(251,191,36,0.6)"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 2, delay: 1.6, ease: "easeInOut" }}
        />
        {/* Landmark dots */}
        {[
          { cx: 100, cy: 50 },
          { cx: 138, cy: 95 },
          { cx: 140, cy: 68 },
        ].map((dot, i) => (
          <motion.circle
            key={i}
            cx={dot.cx}
            cy={dot.cy}
            r="3"
            fill="rgba(192,132,252,0.9)"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: [0, 1.4, 1], opacity: 1 }}
            transition={{ delay: 2 + i * 0.2, duration: 0.5 }}
          />
        ))}
      </svg>
    </motion.div>
  );
}

// ── Main Landing Page ─────────────────────────────────────────────────────────
export default function LandingPage() {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement>(null);

  // Subtle mouse parallax on the aura
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handleMove = (e: MouseEvent) => {
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      const x = (clientX / innerWidth - 0.5) * 20;
      const y = (clientY / innerHeight - 0.5) * 20;
      el.style.setProperty("--mx", `${x}px`);
      el.style.setProperty("--my", `${y}px`);
    };
    window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative min-h-dvh flex flex-col items-center justify-center overflow-hidden bg-void"
    >
      {/* ── Animated background particles ── */}
      <div className="absolute inset-0 pointer-events-none">
        {PARTICLES.map((p) => (
          <motion.div
            key={p.id}
            className="absolute rounded-full bg-mystic-400"
            style={{
              left:   `${p.x}%`,
              top:    `${p.y}%`,
              width:  `${p.size}px`,
              height: `${p.size}px`,
              opacity: p.opacity,
            }}
            animate={{
              y:       [0, -p.speed, 0],
              x:       [0, p.drift, 0],
              opacity: [p.opacity, p.opacity * 1.8, p.opacity],
            }}
            transition={{
              duration:   p.speed * 0.8 + 8,
              repeat:     Infinity,
              ease:       "easeInOut",
              delay:      p.id * 0.07,
            }}
          />
        ))}
      </div>

      {/* ── Background mystic aura ── */}
      <div className="absolute inset-0 bg-mystic-aura pointer-events-none" />
      <div
        className="absolute inset-0 pointer-events-none transition-transform duration-300"
        style={{
          background:  "radial-gradient(ellipse 60% 50% at 50% 0%, rgba(109,40,217,0.08) 0%, transparent 70%)",
          transform:   "translate(var(--mx, 0px), var(--my, 0px))",
        }}
      />

      {/* ── Main content ── */}
      <AnimatePresence>
        <motion.div
          className="relative z-10 flex flex-col items-center text-center px-6 max-w-2xl"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: "easeOut" }}
        >
          {/* Brand badge */}
          <motion.div
            className="flex items-center gap-2 mb-8 px-4 py-2 rounded-full border border-mystic-700/40 bg-mystic-900/20 backdrop-blur-sm"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Sparkles className="w-3.5 h-3.5 text-mystic-400" />
            <span className="text-mystic-300 text-xs tracking-[0.2em] uppercase font-medium">
              AI Palmistry Experience
            </span>
          </motion.div>

          {/* Palm glyph */}
          <PalmGlyph />

          {/* Wordmark */}
          <motion.h1
            className="font-serif text-6xl md:text-8xl font-light tracking-tight mb-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
          >
            <span className="text-gradient-mystic">PALM</span>
            <span className="text-text-primary">VERSE</span>
          </motion.h1>

          {/* Tagline */}
          <motion.p
            className="font-serif text-xl md:text-2xl text-text-secondary italic mb-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.8 }}
          >
            Discover the story written in your palm.
          </motion.p>

          {/* Description */}
          <motion.p
            className="text-text-muted text-sm md:text-base leading-relaxed mb-12 max-w-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            Explore an AI-powered palmistry experience that analyses palm patterns
            and transforms them into a personalised reading.
          </motion.p>

          {/* CTA button */}
          <motion.button
            className="btn-primary flex items-center gap-3 text-base group"
            onClick={() => navigate("/details")}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            <span>Begin Your Reading</span>
            <ArrowRight className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-1" />
          </motion.button>

          {/* Disclaimer */}
          <motion.p
            className="mt-6 text-text-muted text-xs"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
          >
            For entertainment and exploration.
          </motion.p>
        </motion.div>
      </AnimatePresence>

      {/* ── Bottom gradient fade ── */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-void to-transparent pointer-events-none" />
    </div>
  );
}
