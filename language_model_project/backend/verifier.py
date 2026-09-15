"""
Phase 5 verifier agent (derma_guide_plan.md: "the one agentic component
kept... One prompt plus one function").

After a generator produces an answer, this re-reads it against the SAME
retrieved excerpts the generator was shown and checks each claim's citation:
  1. is the citation real — does "[Chapter: X, p.Y]" actually match one of
     the numbered excerpts the model was given, or is the chapter/page
     fabricated (the model hallucinated a plausible-looking reference)?
  2. is the claim grounded — does that excerpt's text actually support the
     claim, or does the citation point at a real excerpt that doesn't
     actually say this?

Both failure modes matter and are reported separately: a fabricated
citation is a worse failure than a real-but-mismatched one, and Phase 5's
analysis explicitly wants "does the system abstain or confidently guess" as
a distinct question from "are its citations accurate."

This is deliberately one Claude call per verification (one prompt, one
function, per the plan) rather than a multi-step claim-extraction pipeline
— simple enough to run on every live chat turn (surfaced as a UI badge) and
in a batch for eval_generation.py's citation-accuracy metric.
"""

from __future__ import annotations

import json
import re

import anthropic

import config

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

VERIFIER_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a fact-checking verifier for a medical RAG system. You will be given:
1. A set of numbered book excerpts (the ONLY source material the answering model was allowed to use).
2. An answer that model produced, which should cite those excerpts inline like "[Chapter: X, p.Y]" or "[Excerpt N]".

Break the answer into its individual factual claims (skip pure transitions/headings). For each claim:
- Find any citation attached to it (a chapter/page reference, or an excerpt number).
- "citation_valid": true only if that citation genuinely corresponds to one of the numbered excerpts
  provided (match by chapter name and page range) — false if the model invented a chapter/page that
  isn't among the excerpts, or attached no citation at all to a claim that needed one.
- "excerpt_index": the excerpt number (1-based) the citation points to, if citation_valid, else null.
- "supported": true only if that excerpt's text actually contains or directly implies this claim
  (not just topically related) — false otherwise, including when citation_valid is false.

Also record whether the answer is an explicit abstention (states the excerpts don't contain enough
information) — in that case "claims" should be empty and "is_abstention" should be true.

Respond with ONLY a single JSON object, no other text, no markdown fences:
{"is_abstention": false,
 "claims": [{"claim": "...", "citation_valid": true, "excerpt_index": 2, "supported": true}, ...]}"""


def _format_excerpts(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        page_ref = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] \
                   else f"pp.{c['page_start']}–{c['page_end']}"
        parts.append(f"[Excerpt {i} | {c['chapter']} | {page_ref}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _parse_response(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "claims" not in obj or not isinstance(obj["claims"], list):
        return None
    obj.setdefault("is_abstention", False)
    return obj


def _empty_result(reason: str) -> dict:
    return {
        "is_abstention": False,
        "claims": [],
        "n_claims": 0,
        "n_grounded": 0,
        "groundedness_score": None,
        "error": reason,
    }


def verify(answer: str, chunks: list[dict], model: str = VERIFIER_MODEL) -> dict:
    """Returns:
        {is_abstention, claims: [...], n_claims, n_grounded,
         groundedness_score: fraction of claims with citation_valid AND
         supported (None if there are no claims to check — an abstention,
         or an unparseable verifier response, which is reported honestly
         via "error" rather than silently scored as 0 or 1), "error"?: str}
    """
    if not chunks:
        # Nothing was retrieved at all (arm A / closed-book) — there is no
        # excerpt a citation could validly point to, so every claim is
        # automatically uncited-by-construction rather than "checked."
        return _empty_result("no chunks were retrieved — nothing to verify citations against")

    excerpts_text = _format_excerpts(chunks)
    user_message = f"Excerpts:\n\n{excerpts_text}\n\n---\n\nAnswer to check:\n\n{answer}"

    try:
        response = _client.messages.create(
            model=model,
            max_tokens=1500,
            # See generators.py's ClaudeGenerator.answer() — some
            # claude-sonnet-5 calls spend the whole max_tokens budget on
            # invisible extended thinking before any text. Not needed for
            # this structured extraction task.
            thinking={"type": "disabled"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:  # noqa: BLE001 — a verifier outage shouldn't crash the chat turn
        return _empty_result(f"verifier API call failed: {e}")

    text = next((b.text for b in response.content if b.type == "text"), "")
    parsed = _parse_response(text)
    if parsed is None:
        return _empty_result("verifier response was not parseable JSON")

    if parsed.get("is_abstention"):
        return {"is_abstention": True, "claims": [], "n_claims": 0, "n_grounded": 0, "groundedness_score": None}

    claims = parsed.get("claims", [])
    # Defensive: seen for real (Phase 5's Colab run, arm B) — the verifier
    # model can put a stray malformed entry (e.g. a bare string) in an
    # otherwise well-formed claims list. _parse_response already rejects a
    # non-list "claims" outright, but a single bad *element* used to crash
    # every downstream c.get(...) call and take down the entire eval run
    # over one bad claim. Count it as ungrounded, not a fatal error.
    valid_claims = [c for c in claims if isinstance(c, dict)]
    n_claims = len(claims)
    n_grounded = sum(1 for c in valid_claims if c.get("citation_valid") and c.get("supported"))
    result = {
        "is_abstention": False,
        "claims": claims,
        "n_claims": n_claims,
        "n_grounded": n_grounded,
        "groundedness_score": (n_grounded / n_claims) if n_claims else None,
    }
    if len(valid_claims) < len(claims):
        result["n_malformed_claims"] = len(claims) - len(valid_claims)
    return result
