/**
 * mockPalmAnalysis.ts — Temporary mock analysis service
 *
 * PURPOSE:
 *   Generates a deterministic palm reading result from user details
 *   so the UI has something to render before the real backend exists.
 *
 * FUTURE REPLACEMENT:
 *   When the FastAPI backend is ready (Phase 9), this entire file will
 *   be replaced by an API call:
 *
 *     POST /api/analyze-palm
 *       body: { userDetails, palmImage (base64) }
 *       returns: PalmReadingResult
 *
 *   The component that calls generateReading() will not need to change —
 *   only this service file will be swapped out. That is the value of
 *   isolating this logic here instead of in the page component.
 *
 * DETERMINISM:
 *   We derive a seed number from the user's name and date of birth so
 *   the same person always gets the same reading on repeated runs.
 *   This avoids jarring result changes on re-render.
 */

import type { PalmReadingResult, UserDetails, ReadingCategory } from "../types/palm";

// ── Simple string → deterministic number seed ─────────────────────────────────
function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0; // Convert to 32-bit integer
  }
  return Math.abs(hash);
}

// ── Seeded pseudo-random number generator ─────────────────────────────────────
// Returns a float in [0, 1) deterministically from a seed and an index.
function seededRandom(seed: number, index: number): number {
  const x = Math.sin(seed + index) * 10000;
  return x - Math.floor(x);
}

// ── Score generator: returns 0–100 from a seeded random ──────────────────────
function scoreFrom(seed: number, idx: number, min = 55, max = 96): number {
  return Math.round(min + seededRandom(seed, idx) * (max - min));
}

// ── Reading pools — varied phrases for each category ─────────────────────────
const LOVE_SUMMARIES = [
  "A heart generous with feeling, yet selective with trust.",
  "Depth runs beneath a composed surface — connections are earned, not given.",
  "Your path of love is winding but ultimately rich with meaning.",
];
const LOVE_DETAILS = [
  "Your palm suggests a romantic nature that values depth over breadth. You tend to invest fully in relationships once trust is established. A period of meaningful connection may be drawing closer.",
  "The lines suggest a cautious but warm emotional nature. You give loyalty slowly and expect it in return. Patience in love tends to reward you with lasting bonds.",
  "You carry an intuitive understanding of others' needs. This emotional attunement makes you a valued partner — but remember to honour your own needs equally.",
];

const CAREER_SUMMARIES = [
  "Ambition tempered by purpose — work means more than reward.",
  "A path of steady building; recognition comes through consistency.",
  "Creative thinking opens doors that determination alone cannot.",
];
const CAREER_DETAILS = [
  "Your palm patterns reflect an individual who works with intention. You are drawn to roles where your ideas can take tangible form. A transition or expansion of responsibility may be on the horizon.",
  "Discipline and methodical thinking are your professional strengths. The slow arc of your career bends toward increasing influence — trust the process.",
  "Adaptability and creative problem-solving are your career hallmarks. Environments that reward unconventional thinking bring out the best in you.",
];

const MONEY_SUMMARIES = [
  "Financial intuition is strong; impulsive decisions are the main risk.",
  "A natural builder of stability — slow gains accumulate with discipline.",
  "Unexpected opportunities favour those who remain prepared.",
];
const MONEY_DETAILS = [
  "Your palm suggests a practical attitude toward resources combined with occasional bursts of inspired risk-taking. The key is distinguishing between the two in the moment.",
  "Consistent, methodical financial habits serve you far better than speculation. Your financial path is one of gradual, sustainable accumulation.",
  "You have a keen eye for opportunity but sometimes act on impulse. Building a small reserve of caution alongside your instincts will serve you well.",
];

const WELLBEING_SUMMARIES = [
  "Vitality runs high when rest is honoured alongside effort.",
  "The body signals its needs clearly — the practice is listening.",
  "Energy is your natural gift; the challenge is channelling it wisely.",
];
const WELLBEING_DETAILS = [
  "Your palm reflects an active constitution with strong natural energy. The greatest wellbeing gains come from balancing this output with genuine recovery — sleep, stillness, and time in nature.",
  "You are sensitive to your environment in ways others may not be. Creating physical and mental space protects your wellbeing more than any external remedy.",
  "A dynamic and expressive nature tends toward high output. Learning to pause before depletion — rather than after — will sustain your energy through long-term endeavours.",
];

const MIND_SUMMARIES = [
  "An analytical mind with a flair for seeing patterns others miss.",
  "Curiosity is the engine; depth of focus is the fuel.",
  "Intuition and logic are equally present — a rare and useful balance.",
];
const MIND_DETAILS = [
  "Your palm suggests a mind that moves quickly between ideas, seeking connections. This associative thinking is a creative asset. Allowing ideas to settle before acting on them strengthens outcomes.",
  "You have a natural inclination toward deep inquiry. Once engaged, your focus is formidable. The challenge is choosing your focus points wisely — not every problem deserves equal attention.",
  "You operate from both feeling and reason, which gives you a nuanced perspective. When these faculties align, your decisions tend to be both sound and inspired.",
];

const PATH_SUMMARIES = [
  "A life of deliberate choices leading toward something genuinely your own.",
  "The path is less a straight line than a series of meaningful turning points.",
  "Growth is the constant — the destination reveals itself through the journey.",
];
const PATH_DETAILS = [
  "The patterns in your palm suggest a life in which your most meaningful chapters are still being written. The decisions you are making now are forming the foundation of a future that reflects your deepest values.",
  "Your life path has involved significant turning points that, in retrospect, redirected you more accurately toward your true direction. Trust that current uncertainties serve a similar purpose.",
  "Your overall life pattern reflects a person who grows through experience rather than avoiding it. The breadth of what you have encountered — and will encounter — becomes the source of your genuine wisdom.",
];

// ── Category builder ──────────────────────────────────────────────────────────
function buildCategory(
  seed: number,
  scoreIdx: number,
  summaries: string[],
  details: string[],
  summaryIdx: number,
  icon: string,
  label: string
): ReadingCategory {
  return {
    score:   scoreFrom(seed, scoreIdx),
    summary: summaries[summaryIdx % summaries.length],
    details: details[summaryIdx % details.length],
    icon,
    label,
  };
}

// ── Main export ───────────────────────────────────────────────────────────────
/**
 * Generate a deterministic palm reading result.
 *
 * FUTURE: Replace the body of this function with:
 *   const response = await fetch('/api/analyze-palm', {
 *     method: 'POST',
 *     body: JSON.stringify({ userDetails, palmImage }),
 *   });
 *   return response.json() as PalmReadingResult;
 */
export async function generateReading(
  userDetails: UserDetails,
  _palmImage: string  // will be sent to API in the real implementation
): Promise<PalmReadingResult> {
  // Simulate network/processing delay
  await new Promise((r) => setTimeout(r, 3500));

  const seed = hashString(userDetails.name + userDetails.dateOfBirth);
  const i = (n: number) => Math.floor(seededRandom(seed, n) * 3);

  const OVERALL_SUMMARIES = [
    `${userDetails.name}, your palm reveals a life of intention — shaped by both sensitivity and quiet strength. The lines suggest a nature that grows most fully through authentic connection and purposeful work.`,
    `${userDetails.name}, the story written in your palm speaks of a thoughtful spirit navigating a rich, complex inner world. You carry both the capacity for deep feeling and the resilience to move through difficulty.`,
    `${userDetails.name}, your palm reflects a journey of continuous self-discovery. The patterns suggest someone whose greatest chapters are shaped by the courage to pursue what is truly meaningful — not merely what is convenient.`,
  ];

  return {
    overallSummary: OVERALL_SUMMARIES[i(0)],
    categories: {
      love:      buildCategory(seed, 1, LOVE_SUMMARIES,      LOVE_DETAILS,      i(1), "❤️",  "Love & Relationships"),
      career:    buildCategory(seed, 2, CAREER_SUMMARIES,    CAREER_DETAILS,    i(2), "💼",  "Career"),
      money:     buildCategory(seed, 3, MONEY_SUMMARIES,     MONEY_DETAILS,     i(3), "💰",  "Money & Opportunities"),
      wellbeing: buildCategory(seed, 4, WELLBEING_SUMMARIES, WELLBEING_DETAILS, i(4), "🌿", "Wellbeing"),
      mind:      buildCategory(seed, 5, MIND_SUMMARIES,      MIND_DETAILS,      i(5), "🧠",  "Mind & Personality"),
      path:      buildCategory(seed, 6, PATH_SUMMARIES,      PATH_DETAILS,      i(6), "✨",  "Overall Path"),
    },
    generatedAt: new Date().toISOString(),
  };
}
