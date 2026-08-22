/**
 * PalmReadingContext.tsx — Global state for the PalmVerse flow
 *
 * WHY CONTEXT:
 *   The user's name, palm image, and reading result need to be accessed
 *   from multiple pages (Details → Capture → Analysis → Results).
 *   Instead of prop-drilling through every component, we use React Context —
 *   a built-in way to share state across the component tree.
 *
 * HOW TO USE:
 *   1. Wrap <App> with <PalmReadingProvider>
 *   2. In any component: const { state, dispatch } = usePalmReading()
 *   3. Dispatch actions like: dispatch({ type: 'SET_USER', payload: { ... } })
 */

import React, { createContext, useContext, useReducer, ReactNode } from "react";
import type { PalmReadingState, UserDetails, PalmReadingResult } from "../types/palm";

// ── Initial state ─────────────────────────────────────────────────────────────
const initialState: PalmReadingState = {
  userDetails: null,
  palmImage:   null,
  result:      null,
  isComplete:  false,
};

// ── Action types ──────────────────────────────────────────────────────────────
// Using discriminated union types — TypeScript ensures each action has
// the correct payload shape at compile time.
type Action =
  | { type: "SET_USER";    payload: UserDetails }
  | { type: "SET_IMAGE";   payload: string }
  | { type: "SET_RESULT";  payload: PalmReadingResult }
  | { type: "RESET" };

// ── Reducer ───────────────────────────────────────────────────────────────────
// A reducer is a pure function: (currentState, action) → newState.
// It never mutates state directly — it returns a new state object.
// This is the same pattern Redux uses, but simpler (no library needed).
function reducer(state: PalmReadingState, action: Action): PalmReadingState {
  switch (action.type) {
    case "SET_USER":
      return { ...state, userDetails: action.payload };

    case "SET_IMAGE":
      return { ...state, palmImage: action.payload };

    case "SET_RESULT":
      return { ...state, result: action.payload, isComplete: true };

    case "RESET":
      return initialState;

    default:
      return state;
  }
}

// ── Context definition ────────────────────────────────────────────────────────
interface PalmReadingContextValue {
  state:    PalmReadingState;
  dispatch: React.Dispatch<Action>;
}

const PalmReadingContext = createContext<PalmReadingContextValue | null>(null);

// ── Provider component ────────────────────────────────────────────────────────
export function PalmReadingProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <PalmReadingContext.Provider value={{ state, dispatch }}>
      {children}
    </PalmReadingContext.Provider>
  );
}

// ── Custom hook ───────────────────────────────────────────────────────────────
// Instead of importing useContext(PalmReadingContext) in every component,
// we export a hook that includes the null-safety check.
export function usePalmReading(): PalmReadingContextValue {
  const ctx = useContext(PalmReadingContext);
  if (!ctx) {
    throw new Error("usePalmReading must be used inside <PalmReadingProvider>");
  }
  return ctx;
}
