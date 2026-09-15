import { Verification } from "../lib/types";

// Phase 5 verifier agent (backend/verifier.py) surfaced as a live badge —
// "gives the presentation a live demo moment" per derma_guide_plan.md.
export function VerificationBadge({ verification }: { verification: Verification }) {
  if (verification.error) {
    return (
      <span className="text-xs text-slate-400" title={verification.error}>
        verification unavailable
      </span>
    );
  }

  if (verification.is_abstention) {
    return (
      <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
        ⊘ abstained
      </span>
    );
  }

  const score = verification.groundedness_score;
  if (score === null) {
    // No chunks were retrieved at all — nothing for the verifier to check
    // citations against (e.g. a closed-book generator).
    return null;
  }

  const pct = Math.round(score * 100);
  const tone =
    score >= 0.8
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : score >= 0.5
      ? "bg-amber-50 text-amber-700 border-amber-200"
      : "bg-red-50 text-red-700 border-red-200";

  return (
    <span
      className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${tone}`}
      title={`${verification.n_grounded}/${verification.n_claims} claims grounded in the cited excerpts`}
    >
      ✓ {pct}% grounded
    </span>
  );
}
