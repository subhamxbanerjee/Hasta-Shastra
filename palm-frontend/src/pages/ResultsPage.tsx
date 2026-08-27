import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { RotateCcw, Share2, ChevronDown, ChevronUp, Star, Activity, Maximize, TrendingUp, Layers } from "lucide-react";
import { usePalmReading } from "../context/PalmReadingContext";
import type { LineFeatures } from "../types/palm";

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

// ── Individual Line Feature Card ───────────────────────────────────────────────
function LineCard({ lineName, features, index }: { lineName: string; features: LineFeatures; index: number }) {
  const [expanded, setExpanded] = useState(false);

  // Derive score from confidence (0-1) to percentage (0-100)
  const score = features.max_confidence ? Math.round(features.max_confidence * 100) : 0;

  return (
    <motion.div
      className="glass-card overflow-hidden"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 + index * 0.08, duration: 0.5 }}
    >
      <button
        className="w-full flex items-center gap-4 p-5 text-left"
        onClick={() => setExpanded((e) => !e)}
      >
        <div className="relative flex-shrink-0">
          <ScoreArc score={score} />
          <span className="absolute inset-0 flex items-center justify-center text-sm rotate-90">
             {features.detected ? "✅" : "❌"}
          </span>
        </div>

        <div className="flex-1 min-w-0">
          <p className="text-text-muted text-xs uppercase tracking-widest mb-0.5">{lineName}</p>
          <p className="text-text-secondary text-sm leading-snug">
             {features.detected ? `Detected with ${score}% confidence` : "Not prominently detected"}
          </p>
        </div>

        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          {expanded
            ? <ChevronUp   className="w-4 h-4 text-text-muted" />
            : <ChevronDown className="w-4 h-4 text-text-muted" />
          }
        </div>
      </button>

      <AnimatePresence>
        {expanded && features.detected && (
          <motion.div
            key="detail"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-0 border-t border-border/40">
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="flex items-center gap-2">
                   <Maximize className="w-4 h-4 text-mystic-400" />
                   <div>
                     <p className="text-text-muted text-xs">Total Length</p>
                     <p className="text-mystic-300 text-sm font-medium">{Math.round(features.total_length)} px</p>
                   </div>
                </div>
                <div className="flex items-center gap-2">
                   <Layers className="w-4 h-4 text-mystic-400" />
                   <div>
                     <p className="text-text-muted text-xs">Fragments</p>
                     <p className="text-mystic-300 text-sm font-medium">{features.num_fragments}</p>
                   </div>
                </div>
                <div className="flex items-center gap-2">
                   <TrendingUp className="w-4 h-4 text-mystic-400" />
                   <div>
                     <p className="text-text-muted text-xs">Orientation</p>
                     <p className="text-mystic-300 text-sm font-medium">{features.weighted_orientation ? `${Math.round(features.weighted_orientation)}°` : 'N/A'}</p>
                   </div>
                </div>
                <div className="flex items-center gap-2">
                   <Activity className="w-4 h-4 text-mystic-400" />
                   <div>
                     <p className="text-text-muted text-xs">Curvature</p>
                     <p className="text-mystic-300 text-sm font-medium">{features.weighted_curvature ? features.weighted_curvature.toFixed(2) : 'N/A'}</p>
                   </div>
                </div>
              </div>
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

  const handleReadAgain = () => {
    dispatch({ type: "RESET" });
    navigate("/");
  };

  const lines = Object.entries(result.lines);

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
              Palm Structure
            </h1>
            <h2 className="font-serif text-2xl md:text-3xl text-text-primary mb-4">
              Analysis Complete
            </h2>

            {/* User details pill */}
            <div className="flex items-center justify-center gap-4 flex-wrap mb-6">
              <span className="text-text-muted text-xs px-3 py-1 rounded-full border border-border/50 bg-surface/40">
                Image: {result.image_id}
              </span>
              <span className="text-text-muted text-xs px-3 py-1 rounded-full border border-border/50 bg-surface/40">
                Res: {result.image_metadata.normalized_width}x{result.image_metadata.normalized_height}
              </span>
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
          <p className="text-text-muted text-xs uppercase tracking-widest mb-3">CV Pipeline Metrics</p>
          <div className="flex justify-between items-center border-b border-border/30 pb-3 mb-3">
             <span className="text-text-secondary text-sm">Major Lines Detected</span>
             <span className="text-mystic-300 font-medium">{result.summary.detected_major_lines}</span>
          </div>
          <div className="flex justify-between items-center border-b border-border/30 pb-3 mb-3">
             <span className="text-text-secondary text-sm">Line Groupings</span>
             <span className="text-mystic-300 font-medium">{result.summary.total_major_line_groups}</span>
          </div>
          <div className="flex justify-between items-center">
             <span className="text-text-secondary text-sm">Average Confidence</span>
             <span className="text-mystic-300 font-medium">{Math.round(result.summary.average_confidence * 100)}%</span>
          </div>
        </motion.div>

        {/* ── Line Feature cards ── */}
        <div className="space-y-3">
          {lines.map(([lineName, features], i) => (
            <LineCard key={lineName} lineName={lineName} features={features} index={i} />
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
            This data is derived directly from the Phase 6.0 Computer Vision architecture.
            No palmistry interpretation or predictive logic is applied in this step.
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
            className="btn-primary flex items-center gap-2 flex-1 justify-center text-sm"
          >
            <RotateCcw className="w-4 h-4" />
            Analyse Another Image
          </button>
        </motion.div>
      </div>
    </div>
  );
}
