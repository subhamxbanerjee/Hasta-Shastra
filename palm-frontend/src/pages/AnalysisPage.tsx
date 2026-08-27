import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { usePalmReading } from "../context/PalmReadingContext";
import { analyzePalm } from "../services/apiClient";

const STAGES = [
  { pct: 0,  label: "Preparing your palm image…"       },
  { pct: 20, label: "Mapping palm structure…"           },
  { pct: 40, label: "Tracing major palm patterns…"      },
  { pct: 62, label: "Analysing line characteristics…"   },
  { pct: 82, label: "Creating your personalised reading…" },
  { pct: 100, label: "Complete."                         },
];

// ── Animated progress ring ─────────────────────────────────────────────────
function ProgressRing({ progress }: { progress: number }) {
  const r   = 52;
  const circ = 2 * Math.PI * r;
  const dash = circ * (1 - progress / 100);

  return (
    <svg width="130" height="130" className="rotate-[-90deg]">
      {/* Track */}
      <circle cx="65" cy="65" r={r} fill="none" stroke="rgba(37,37,64,0.8)" strokeWidth="6" />
      {/* Progress arc */}
      <circle
        cx="65" cy="65" r={r}
        fill="none"
        stroke="url(#ringGrad)"
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={dash}
        style={{ transition: "stroke-dashoffset 0.6s ease" }}
      />
      <defs>
        <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stopColor="#7c3aed" />
          <stop offset="100%" stopColor="#c084fc" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export default function AnalysisPage() {
  const navigate              = useNavigate();
  const { state, dispatch }   = usePalmReading();
  const [progress, setProgress] = useState(0);
  const [stageIdx, setStageIdx] = useState(0);
  const [scanY, setScanY]       = useState(0);
  const calledRef               = useRef(false);

  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Guard: redirect if no image or user
  useEffect(() => {
    if (!state.palmImage || !state.userDetails) {
      navigate("/capture");
    }
  }, []);

  // ── Animated progress driver ─────────────────────────────────────────────
  useEffect(() => {
    if (errorMsg) return; // stop progress if error

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) { clearInterval(interval); return 100; }
        // Slightly random increments for realistic feel
        const step = Math.random() * 2.5 + 0.8;
        return Math.min(prev + step, 100);
      });
    }, 80);
    return () => clearInterval(interval);
  }, [errorMsg]);

  // Update stage label based on progress
  useEffect(() => {
    for (let i = STAGES.length - 1; i >= 0; i--) {
      if (progress >= STAGES[i].pct) { setStageIdx(i); break; }
    }
  }, [progress]);

  // ── Scanning line animation ───────────────────────────────────────────────
  useEffect(() => {
    let frame: number;
    let dir = 1;
    let y   = 0;
    const tick = () => {
      y += dir * 0.8;
      if (y >= 100) dir = -1;
      if (y <= 0)   dir = 1;
      setScanY(y);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  // ── Trigger API analysis ─────────────────────────────────────────
  useEffect(() => {
    if (calledRef.current) return;
    calledRef.current = true;

    if (!state.userDetails || !state.palmImage) return;

    analyzePalm(state.palmImage)
      .then((result) => {
        dispatch({ type: "SET_RESULT", payload: result });
        // wait for progress to visually reach 100% before redirecting
        const checkProgress = setInterval(() => {
          setProgress((p) => {
            if (p >= 100) {
              clearInterval(checkProgress);
              navigate("/results");
            }
            return p;
          });
        }, 100);
      })
      .catch((err) => {
        setErrorMsg(err.message);
      });
  }, []);

  const currentLabel = STAGES[stageIdx]?.label ?? "";

  return (
    <div className="min-h-dvh bg-void bg-mystic-aura flex flex-col items-center justify-center px-4">
      <motion.div
        className="w-full max-w-sm flex flex-col items-center gap-8"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
      >
        {/* Palm image with scan overlay */}
        {state.palmImage && (
          <div className="relative w-56 h-56 rounded-2xl overflow-hidden border border-mystic-700/40"
               style={{ boxShadow: "0 0 40px rgba(147,51,234,0.25)" }}>
            <img
              src={state.palmImage}
              alt="Palm being analysed"
              className="w-full h-full object-cover"
            />
            {/* Scan line */}
            <motion.div
              className="absolute left-0 right-0 h-0.5 pointer-events-none"
              style={{
                top:        `${scanY}%`,
                background: "linear-gradient(90deg, transparent 0%, rgba(192,132,252,0.8) 50%, transparent 100%)",
                boxShadow:  "0 0 12px 3px rgba(192,132,252,0.4)",
              }}
            />
            {/* Purple tint overlay */}
            <div className="absolute inset-0 bg-gradient-to-b from-mystic-900/20 to-mystic-900/40" />
          </div>
        )}

        {/* Progress ring + percentage */}
        <div className="relative flex items-center justify-center">
          <ProgressRing progress={progress} />
          <div className="absolute flex flex-col items-center">
            <span className="text-2xl font-serif text-text-primary">{Math.round(progress)}%</span>
          </div>
        </div>

        {/* Stage label */}
        <AnimatePresence mode="wait">
          <motion.p
            key={stageIdx}
            className="text-text-secondary text-sm text-center"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.4 }}
          >
            {currentLabel}
          </motion.p>
        </AnimatePresence>

        {/* Progress bar */}
        <div className="w-full bg-surface rounded-full h-1.5 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-mystic-700 to-mystic-400 rounded-full"
            style={{ width: `${progress}%`, transition: "width 0.3s ease" }}
          />
        </div>

        <p className="text-text-muted text-xs text-center">
          Analysing palm line patterns and characteristics…
        </p>
        {/* Error State */}
        {errorMsg && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center gap-4 mt-4"
          >
            <div className="p-3 bg-red-950/40 border border-red-500/50 rounded-xl text-red-200 text-sm text-center">
              {errorMsg}
            </div>
            <button
              onClick={() => navigate("/capture")}
              className="btn-ghost"
            >
              Go Back and Try Again
            </button>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
