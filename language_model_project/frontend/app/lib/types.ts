export interface Source {
  chapter: string;
  section: string;
  subsection: string;
  page_start: number | null;
  page_end: number | null;
  excerpt: string;
  score: number;
}

// Phase 5's verifier agent (backend/verifier.py) — one groundedness check
// per turn, re-checking the answer's citations against the same chunks the
// generator saw. groundedness_score is null when there's nothing to check
// (an abstention, or the verifier call itself failed — see `error`).
export interface Verification {
  is_abstention: boolean;
  n_claims: number;
  n_grounded: number;
  groundedness_score: number | null;
  error?: string | null;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  verification?: Verification;
}
