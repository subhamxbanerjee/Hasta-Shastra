import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import { PalmReadingProvider, usePalmReading } from "./context/PalmReadingContext";

import LandingPage  from "./pages/LandingPage";
import DetailsPage  from "./pages/DetailsPage";
import CapturePage  from "./pages/CapturePage";
import AnalysisPage from "./pages/AnalysisPage";
import ResultsPage  from "./pages/ResultsPage";

// ── Route guard ───────────────────────────────────────────────────────────────
// Prevents users jumping to a later step without completing earlier ones.
// e.g. accessing /results directly redirects to /
function GuardedRoute({
  children,
  requireUser   = false,
  requireImage  = false,
  requireResult = false,
}: {
  children:       React.ReactNode;
  requireUser?:   boolean;
  requireImage?:  boolean;
  requireResult?: boolean;
}) {
  const { state } = usePalmReading();
  if (requireResult && !state.result)      return <Navigate to="/"        replace />;
  if (requireImage  && !state.palmImage)   return <Navigate to="/capture" replace />;
  if (requireUser   && !state.userDetails) return <Navigate to="/details" replace />;
  return <>{children}</>;
}

// ── Animated route wrapper ────────────────────────────────────────────────────
function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/"          element={<LandingPage />} />
        <Route path="/details"   element={<DetailsPage />} />
        <Route path="/capture"   element={
          <GuardedRoute requireUser>
            <CapturePage />
          </GuardedRoute>
        } />
        <Route path="/analyzing" element={
          <GuardedRoute requireUser requireImage>
            <AnalysisPage />
          </GuardedRoute>
        } />
        <Route path="/results"   element={
          <GuardedRoute requireUser requireImage requireResult>
            <ResultsPage />
          </GuardedRoute>
        } />
        {/* Catch-all → home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AnimatePresence>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <BrowserRouter>
      <PalmReadingProvider>
        <AnimatedRoutes />
      </PalmReadingProvider>
    </BrowserRouter>
  );
}
