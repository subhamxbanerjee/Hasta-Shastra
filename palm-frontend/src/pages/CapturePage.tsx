import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Upload, Camera, X, CheckCircle, AlertCircle, Sparkles, RotateCcw } from "lucide-react";
import { usePalmReading } from "../context/PalmReadingContext";

const ACCEPTED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

const GUIDANCE = [
  { text: "Use good, even lighting" },
  { text: "Keep your palm fully visible" },
  { text: "Avoid heavy shadows across the palm" },
  { text: "Hold the camera steady and level" },
];

function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-2 mb-8">
      {Array.from({ length: total }, (_, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full transition-all duration-400 ${
            i < current ? "bg-mystic-400" : i === current ? "bg-mystic-500 w-5" : "bg-border"
          }`} />
        </div>
      ))}
      <span className="text-text-muted text-xs ml-2">Step {current + 1} of {total}</span>
    </div>
  );
}

export default function CapturePage() {
  const navigate = useNavigate();
  const { state, dispatch } = usePalmReading();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef     = useRef<HTMLVideoElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);

  const [preview,     setPreview]     = useState<string | null>(state.palmImage);
  const [cameraMode,  setCameraMode]  = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [fileError,   setFileError]   = useState<string | null>(null);
  const [streaming,   setStreaming]   = useState(false);
  const streamRef = useRef<MediaStream | null>(null);

  // ── Handle file upload ───────────────────────────────────────────────────
  const handleFile = (file: File) => {
    setFileError(null);
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setFileError("Please upload a JPG or PNG image.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setFileError("File is too large. Please use an image under 15MB.");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    setCameraMode(false);
    stopCamera();
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, []);

  // ── Camera ───────────────────────────────────────────────────────────────
  const startCamera = async () => {
    setCameraError(null);
    setCameraMode(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStreaming(true);
      }
    } catch {
      setCameraError("Camera access was denied or is not available. Please upload an image instead.");
      setCameraMode(false);
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStreaming(false);
  };

  const captureFrame = () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    setPreview(dataUrl);
    stopCamera();
    setCameraMode(false);
  };

  const clearPreview = () => {
    setPreview(null);
    stopCamera();
    setCameraMode(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Submit ───────────────────────────────────────────────────────────────
  const handleAnalyze = () => {
    if (!preview) return;
    dispatch({ type: "SET_IMAGE", payload: preview });
    navigate("/analyzing");
  };

  return (
    <div className="min-h-dvh bg-void bg-mystic-aura flex items-center justify-center px-4 py-16">
      <motion.div
        className="w-full max-w-lg"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        {/* Header */}
        <div className="mb-8">
          <StepIndicator current={1} total={3} />
          <h1 className="font-serif text-4xl text-text-primary mb-2">Your Palm</h1>
          <p className="text-text-secondary text-sm">
            Place your palm clearly in the frame, fully open with all fingers visible.
          </p>
        </div>

        {/* Guidance checklist */}
        <div className="glass-card p-5 mb-6">
          <p className="text-text-muted text-xs uppercase tracking-widest mb-3">Tips for best results</p>
          <div className="grid grid-cols-2 gap-2">
            {GUIDANCE.map((g, i) => (
              <div key={i} className="flex items-start gap-2">
                <CheckCircle className="w-3.5 h-3.5 text-mystic-400 mt-0.5 flex-shrink-0" />
                <span className="text-text-secondary text-xs">{g.text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Main interaction area */}
        <AnimatePresence mode="wait">
          {preview && !cameraMode ? (
            /* ── Image preview ── */
            <motion.div
              key="preview"
              className="glass-card overflow-hidden mb-5"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
            >
              <div className="relative aspect-video bg-abyss">
                <img
                  src={preview}
                  alt="Palm preview"
                  className="w-full h-full object-contain"
                />
                <button
                  onClick={clearPreview}
                  className="absolute top-3 right-3 p-1.5 rounded-full bg-black/60 border border-border/60 text-text-secondary hover:text-white hover:border-red-500/50 transition-all"
                >
                  <X className="w-4 h-4" />
                </button>
                {/* Success badge */}
                <div className="absolute bottom-3 left-3 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-mystic-900/80 border border-mystic-700/50 backdrop-blur-sm">
                  <CheckCircle className="w-3.5 h-3.5 text-mystic-400" />
                  <span className="text-mystic-300 text-xs font-medium">Image ready</span>
                </div>
              </div>
              <div className="p-4 flex gap-3">
                <button onClick={clearPreview} className="btn-ghost flex items-center gap-2 text-sm flex-1 justify-center">
                  <RotateCcw className="w-3.5 h-3.5" />
                  Replace
                </button>
              </div>
            </motion.div>
          ) : cameraMode ? (
            /* ── Camera view ── */
            <motion.div
              key="camera"
              className="glass-card overflow-hidden mb-5"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="relative aspect-video bg-abyss">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  playsInline
                  muted
                />
                {/* Framing guide overlay */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-2/3 h-4/5 border-2 border-dashed border-mystic-500/40 rounded-2xl" />
                </div>
              </div>
              <div className="p-4 flex gap-3">
                <button
                  onClick={() => { stopCamera(); setCameraMode(false); }}
                  className="btn-ghost flex-1 text-sm justify-center"
                >
                  Cancel
                </button>
                <button
                  onClick={captureFrame}
                  disabled={!streaming}
                  className="btn-primary flex-1 text-sm justify-center disabled:opacity-50"
                >
                  Capture
                </button>
              </div>
            </motion.div>
          ) : (
            /* ── Upload / camera choice ── */
            <motion.div
              key="choose"
              className="space-y-4 mb-5"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {/* Drop zone */}
              <label
                htmlFor="palm-upload"
                className="glass-card glow-border flex flex-col items-center justify-center p-10 cursor-pointer hover:border-mystic-600/50 transition-all"
                onDrop={onDrop}
                onDragOver={(e) => e.preventDefault()}
              >
                <Upload className="w-8 h-8 text-mystic-500 mb-3" />
                <span className="text-text-primary font-medium text-sm mb-1">Upload Palm Image</span>
                <span className="text-text-muted text-xs">JPG or PNG · drag & drop or click</span>
                <input
                  id="palm-upload"
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png"
                  className="hidden"
                  onChange={onFileChange}
                />
              </label>

              <div className="flex items-center gap-3">
                <div className="h-px flex-1 bg-border/50" />
                <span className="text-text-muted text-xs">or</span>
                <div className="h-px flex-1 bg-border/50" />
              </div>

              <button
                onClick={startCamera}
                className="w-full glass-card glow-border flex items-center justify-center gap-3 p-5 cursor-pointer hover:border-mystic-600/50 transition-all"
              >
                <Camera className="w-5 h-5 text-mystic-500" />
                <span className="text-text-primary font-medium text-sm">Use Camera</span>
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error messages */}
        {(fileError || cameraError) && (
          <div className="flex items-start gap-2 p-3 rounded-xl bg-red-950/30 border border-red-800/40 mb-4">
            <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
            <p className="text-red-300 text-xs">{fileError || cameraError}</p>
          </div>
        )}

        {/* Hidden canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Analyze button */}
        <motion.button
          className="btn-primary w-full flex items-center justify-center gap-3 text-base group disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
          onClick={handleAnalyze}
          disabled={!preview}
          whileHover={preview ? { scale: 1.02 } : {}}
          whileTap={preview ? { scale: 0.98 } : {}}
        >
          <Sparkles className="w-4 h-4" />
          <span>Analyse My Palm</span>
        </motion.button>
      </motion.div>
    </div>
  );
}
