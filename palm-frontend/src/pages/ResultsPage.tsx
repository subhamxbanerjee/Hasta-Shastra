import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { RotateCcw, Share2, ChevronDown, ChevronUp, Star } from "lucide-react";
import { usePalmReading } from "../context/PalmReadingContext";
import type { ReadingCategory } from "../types/palm";

// ── Score arc visualisation ────────────────────────────────────────────────
function ScoreArc({ score }: { score: number }) {
  const r    = 20;
  const circ = 2 * Math.PI * r;
  const dash = circ * (1 - score / 100);
  return (
    <svg width="50" height="50" className="rotate-[-90deg]">
      <circle cx="25" cy="25" r={r} fill="none" stroke="rgba(37,37,64,0.8)" strokeWidth="4" />
      <circle
        cx="25" cy="25" r={r}
        fill="none"
        stroke="url(#arcGrad)"
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={dash}
      />
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#7c3aed" />
          <stop offset="100%" stopColor="#c084fc" />
        </linearGradient>
      </defs>
    </svg>
  );
}

// ── Individual category card ───────────────────────────────────────────────
function CategoryCard({ cat, index }: { cat: ReadingCategory; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      className="glass-card overflow-hidden"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 + index * 0.08, duration: 0.5 }}
    >
      {/* Card header — always visible */}
      <button
        className="w-full flex items-center gap-4 p-5 text-left"
        onClick={() => setExpanded((e) => !e)}
      >
        {/* Score arc + icon */}
        <div className="relative flex-shrink-0">
          <ScoreArc score={cat.score} />
          <span className="absolute inset-0 flex items-center justify-center text-sm rotate-90">
            {cat.icon}
          </span>
        </div>

        {/* Label + summary */}
        <div className="flex-1 min-w-0">
          <p className="text-text-muted text-xs uppercase tracking-widest mb-0.5">{cat.label}</p>
          <p className="text-text-secondary text-sm leading-snug">{cat.summary}</p>
        </div>

        {/* Score + toggle */}
        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          <span className="text-mystic-300 font-serif text-lg">{cat.score}</span>
          {expanded
            ? <ChevronUp   className="w-4 h-4 text-text-muted" />
            : <ChevronDown className="w-4 h-4 text-text-muted" />
          }
        </div>
      </button>

      {/* Expandable detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            key="detail"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-0 border-t border-border/40">
              <p className="text-text-secondary text-sm leading-relaxed mt-4">{cat.details}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ── Main Results Page ──────────────────────────────────────────────────────
export default function ResultsPage() {
  const navigate          = useNavigate();
  const { state, dispatch } = usePalmReading();
  const result            = state.result;
  const user              = state.userDetails;

  // Guard: if no result, send back
  if (!result || !user) {
    navigate("/");
    return null;
  }

  const categories = Object.values(result.categories);

  // ── Share ────────────────────────────────────────────────────────────────
  const handleShare = async () => {
    const text = `✨ ${user.name}'s PalmVerse Reading\n\n${result.overallSummary}\n\nGet your own reading at PalmVerse.`;
    if (navigator.share) {
      await navigator.share({ title: "My PalmVerse Reading", text });
    } else {
      await navigator.clipboard.writeText(text);
      alert("Reading copied to clipboard!");
    }
  };

  // ── Read Again ───────────────────────────────────────────────────────────
  const handleReadAgain = () => {
    dispatch({ type: "RESET" });
    navigate("/");
  };

  return (
    <div className="min-h-dvh bg-void bg-mystic-aura pb-20">
      {/* ── Hero header ── */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-mystic-900/20 to-transparent pointer-events-none" />
        <div className="relative max-w-lg mx-auto px-6 pt-16 pb-10 text-center">
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="flex items-center justify-center gap-1.5 mb-6">
              {[...Array(5)].map((_, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 + i * 0.07 }}
                >
                  <Star className="w-3.5 h-3.5 text-gold-400 fill-gold-400" />
                </motion.div>
              ))}
            </div>

            <h1 className="font-serif text-4xl md:text-5xl text-gradient-mystic mb-1">
              {user.name}'s
            </h1>
            <h2 className="font-serif text-2xl md:text-3xl text-text-primary mb-4">
              Palm Reading
            </h2>

            {/* User details pill */}
            <div className="flex items-center justify-center gap-4 flex-wrap mb-6">
              {user.dateOfBirth && (
                <span className="text-text-muted text-xs px-3 py-1 rounded-full border border-border/50 bg-surface/40">
                  Born {new Date(user.dateOfBirth + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })}
                </span>
              )}
              {user.placeOfBirth && (
                <span className="text-text-muted text-xs px-3 py-1 rounded-full border border-border/50 bg-surface/40">
                  {user.placeOfBirth}
                </span>
              )}
            </div>
          </motion.div>
        </div>
      </div>

      <div className="max-w-lg mx-auto px-6 space-y-5">
        {/* ── Overall summary card ── */}
        <motion.div
          className="glass-card p-6 shimmer-line"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
        >
          <p className="text-text-muted text-xs uppercase tracking-widest mb-3">Overall Reading</p>
          <p className="font-serif text-lg text-text-primary leading-relaxed italic">
            "{result.overallSummary}"
          </p>
        </motion.div>

        {/* ── Category cards ── */}
        <div className="space-y-3">
          {categories.map((cat, i) => (
            <CategoryCard key={cat.label} cat={cat} index={i} />
          ))}
        </div>

        {/* ── Disclaimer ── */}
        <motion.div
          className="p-4 rounded-xl border border-border/30 bg-surface/20"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
        >
          <p className="text-text-muted text-xs text-center leading-relaxed">
            PalmVerse is designed for entertainment and self-reflection.
            Palm readings are not scientific predictions and should not replace
            professional advice.
          </p>
        </motion.div>

        {/* ── Action buttons ── */}
        <motion.div
          className="flex gap-3 pt-2"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
        >
          <button
            onClick={handleReadAgain}
            className="btn-ghost flex items-center gap-2 flex-1 justify-center text-sm"
          >
            <RotateCcw className="w-4 h-4" />
            Read Again
          </button>
          <button
            onClick={handleShare}
            className="btn-primary flex items-center gap-2 flex-1 justify-center text-sm"
          >
            <Share2 className="w-4 h-4" />
            Share
          </button>
        </motion.div>
      </div>
    </div>
  );
}
