"""
Phase 5: MCQ accuracy on the held-out 2025+2026 board exams, arms A-D
(derma_guide_plan.md):

    A. Qwen2.5-3B base, no RAG   - closed-book floor (parametric knowledge only)
    B. Qwen2.5-3B base + RAG     - value of retrieval alone
    C. Qwen2.5-3B LoRA + RAG     - value of fine-tuning (the deliverable)
    D. Claude + RAG              - upper bound / the teacher

Retrieval for B/C/D is FIXED to one config — 400_80/bge, the exact index
training/prepare_dataset.py retrieved against to build the LoRA fine-tune
set — so accuracy differences between B/C/D reflect the GENERATOR, not
different context. Only A is closed-book by design (arm A isolates
parametric knowledge with no context at all). RETRIEVE_K and the greedy
token-budget packing below mirror prepare_dataset.py exactly for the same
reason: arm C should see eval-time input from the same distribution it was
fine-tuned on.

Query text for retrieval is the English translation (eval_mcq needs a
NEW, structured, per-option translation of the held-out set —
translate_queries.py's Phase-3 translations only cover the 2021-2024 develop
years, as one merged string; see translate_heldout_mcq.py).

Scoring method is genuinely different between A/B/C and D, and that's a
documented limitation, not an oversight:
  - A/B/C (Qwen2.5-3B, base or LoRA) are scored by LENGTH-NORMALISED
    LOG-LIKELIHOOD of each option's English text as a continuation of the
    prompt (decision #3 in derma_guide_plan.md's System design section) —
    the standard way to MCQ-evaluate a model with no MCQ-answering
    instruction tuning; it removes output-format failures as a confound.
  - D (Claude) exposes no logprobs via the public API, so it is instead
    asked to generate the correct letter directly. Claude also reads the
    ORIGINAL HEBREW stem+options for this (it handles Hebrew natively, and
    generation-format failures are not a real risk for an instruction-tuned
    frontier model the way they are for a 3B base model) — only the
    RETRIEVAL query is translated, to keep B/C/D's retrieved context
    identical.

Usage (arms A/B/C need a GPU — run via notebooks/04_generation_eval.ipynb;
arm D runs anywhere, no GPU, ~$ a few cents in Claude API calls):
    /home/guynitz/venvs/derma_eval/bin/python3 eval_mcq.py --arms D
    python3 eval_mcq.py --arms A,B,C --adapter /path/to/r16/adapter --device cuda

Output: eval/data/mcq_results.json, merged by arm (repeat runs for
different arms compose — same pattern as eval_retrieval.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np

EVAL_DIR = Path(__file__).parent
BACKEND_DIR = EVAL_DIR.parent / "backend"
PIPELINE_DIR = EVAL_DIR.parent / "pipeline"
CHUNKS_PATH = PIPELINE_DIR / "data" / "chunks" / "chunks.jsonl"   # the 400_80 config
INDEX_PATH = EVAL_DIR / "data" / "indices" / "400_80__bge.faiss"
META_PATH = EVAL_DIR / "data" / "indices" / "400_80__bge.meta.jsonl"
ALL_PATH = EVAL_DIR / "data" / "exam_parsed" / "all.jsonl"
TRANSLATIONS_PATH = EVAL_DIR / "data" / "exam_heldout_translations.json"
RESULTS_PATH = EVAL_DIR / "data" / "mcq_results.json"

sys.path.insert(0, str(EVAL_DIR))
from eval_retrieval import embed_queries, load_chunks  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
load_dotenv(BACKEND_DIR / ".env")  # config.py (imported by generators.py) needs these at import time

sys.path.insert(0, str(BACKEND_DIR))
import generators  # noqa: E402 — SYSTEM_PROMPT, build_user_message

HELD_OUT_YEARS = {"2025", "2026"}
SCORED_BUCKETS = ["text_answerable", "image_dependent"]
LETTERS = ["א", "ב", "ג", "ד"]
RETRIEVE_K = 5           # matches training/prepare_dataset.py
MAX_SEQ_LEN = 2048       # matches training/prepare_dataset.py / Phase 4
ANSWER_TOKEN_BUFFER = 64  # options are short; nowhere near the 200 reserved for full answers in training
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CLAUDE_MODEL = "claude-sonnet-5"


# --------------------------------------------------------------- loading ---

def load_heldout(limit: int | None = None) -> list[dict]:
    recs = []
    with ALL_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["year"] in HELD_OUT_YEARS and r["bucket"] in SCORED_BUCKETS:
                recs.append(r)
    if limit:
        recs = recs[:limit]
    return recs


def load_translations() -> dict:
    if not TRANSLATIONS_PATH.exists():
        sys.exit(f"No translations at {TRANSLATIONS_PATH} — run translate_heldout_mcq.py first.")
    return json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))


def usable_records(records: list[dict], translations: dict) -> tuple[list[dict], int, int]:
    """Filters to records that are actually scoreable: all 4 options present
    (options_complete) and a translation exists. Returns (usable, n_no_translation,
    n_incomplete_options) so callers can report honest exclusion counts."""
    usable, n_no_tr, n_incomplete = [], 0, 0
    for r in records:
        key = f"{r['year']}::{r['q_num']}"
        if key not in translations:
            n_no_tr += 1
            continue
        if not r.get("options_complete", True) or any(not r["options"][l].strip() for l in LETTERS):
            n_incomplete += 1
            continue
        usable.append(r)
    return usable, n_no_tr, n_incomplete


# ------------------------------------------------------------- retrieval ---

_chunks_by_id = None
_index = None
_meta = None


def _load_retrieval_backend():
    global _chunks_by_id, _index, _meta
    if _index is not None:
        return
    if not INDEX_PATH.exists():
        sys.exit(f"No index at {INDEX_PATH} — run build_index.py --config 400_80 --embedder bge first.")
    _index = faiss.read_index(str(INDEX_PATH))
    _meta = load_chunks(META_PATH)
    _chunks_by_id = {c["chunk_id"]: c for c in load_chunks(CHUNKS_PATH)}


def retrieval_query_text(record: dict, translations: dict) -> str:
    tr = translations[f"{record['year']}::{record['q_num']}"]
    parts = [tr["stem_en"]]
    for letter in LETTERS:
        text = tr["options_en"].get(letter, "").strip()
        if text:
            parts.append(f"{letter}: {text}")
    return "\n".join(parts)


def retrieve_batch(records: list[dict], translations: dict, k: int = RETRIEVE_K) -> dict[str, list[dict]]:
    """Returns {f"{year}::{q_num}": [chunk_dict, ...]} in rank order."""
    _load_retrieval_backend()
    queries = [retrieval_query_text(r, translations) for r in records]
    qvecs = embed_queries(queries, "bge")
    _, idxs = _index.search(qvecs, k)
    out = {}
    for r, row in zip(records, idxs):
        chunk_ids = [_meta[i]["chunk_id"] for i in row if i != -1]
        out[f"{r['year']}::{r['q_num']}"] = [_chunks_by_id[cid] for cid in chunk_ids if cid in _chunks_by_id]
    return out


def pack_chunks_to_budget(tokenizer, candidate_chunks: list[dict], stem_en: str, reserve: int = ANSWER_TOKEN_BUFFER) -> list[dict]:
    """Greedily pack retrieved chunks (already rank-ordered) up to
    MAX_SEQ_LEN - reserve prompt tokens, same algorithm as
    training/prepare_dataset.py so arm C sees the same input distribution
    it was trained on."""
    budget = MAX_SEQ_LEN - reserve
    packed = []
    for c in candidate_chunks:
        trial = packed + [c]
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": generators.SYSTEM_PROMPT},
             {"role": "user", "content": generators.build_user_message(stem_en, trial)}],
            tokenize=False, add_generation_prompt=True,
        )
        n_tokens = len(tokenizer.encode(prompt_text))
        if n_tokens > budget and packed:
            break
        packed = trial
        if n_tokens > budget:
            break
    return packed


# -------------------------------------------------------- A/B/C scoring ---

def load_qwen(adapter_path: str | None, device: str, load_in_4bit: bool):
    """Loads the shared Qwen2.5-3B base model once; adapter_path=None gives
    arms A/B, a path gives arm C. Kept separate from generators.LoRAGenerator
    because MCQ scoring needs raw logits, not a .generate() call."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    kwargs = {}
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def score_options_likelihood(model, tokenizer, stem_en: str, options_en: dict, chunks: list[dict]) -> dict[str, float]:
    """Length-normalised average log-likelihood of each option's text as a
    continuation of (system + context + stem). chunks=[] scores arm A
    (closed-book)."""
    import torch

    prefix_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": generators.SYSTEM_PROMPT},
         {"role": "user", "content": generators.build_user_message(stem_en, chunks)}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids

    scores = {}
    for letter in LETTERS:
        option_text = options_en.get(letter, "").strip()
        cont_ids = tokenizer(" " + option_text, add_special_tokens=False).input_ids
        full_ids = prefix_ids + cont_ids
        input_tensor = torch.tensor([full_ids], device=model.device)
        with torch.no_grad():
            logits = model(input_tensor).logits[0]  # [seq, vocab]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        # position i's logits predict token i+1; continuation starts at index len(prefix_ids)
        start = len(prefix_ids)
        total = 0.0
        for i, tok in enumerate(cont_ids):
            total += log_probs[start + i - 1, tok].item()
        scores[letter] = total / max(len(cont_ids), 1)
    return scores


def run_local_arm(arm: str, records: list[dict], translations: dict, retrieved: dict,
                   adapter_path: str | None, device: str, load_in_4bit: bool) -> list[dict]:
    model, tokenizer = load_qwen(adapter_path, device, load_in_4bit)
    out = []
    for i, r in enumerate(records, 1):
        key = f"{r['year']}::{r['q_num']}"
        tr = translations[key]
        chunks = [] if arm == "A" else pack_chunks_to_budget(tokenizer, retrieved[key], tr["stem_en"])
        scores = score_options_likelihood(model, tokenizer, tr["stem_en"], tr["options_en"], chunks)
        predicted = max(scores, key=scores.get)
        out.append({**_base_record(r), "predicted": predicted, "option_scores": scores})
        if i % 25 == 0 or i == len(records):
            print(f"  [{arm}] {i}/{len(records)}")
    return out


# ------------------------------------------------------------ D scoring ---

CLAUDE_SYSTEM_PROMPT = """You are answering a multiple-choice question from an Israeli dermatology \
board exam. You will be given book excerpts (may be empty or insufficient) and the question with \
four Hebrew-lettered options (א/ב/ג/ד). Use ONLY the excerpts plus standard dermatology knowledge \
to pick the single best answer.

Respond with ONLY the one Hebrew letter of the correct option (א, ב, ג, or ד) — no explanation, \
no punctuation, nothing else."""


def claude_choose_letter(client, record: dict, chunks: list[dict]) -> str | None:
    excerpts_text = generators.format_context(chunks) if chunks else "(no excerpts retrieved)"
    options_text = "\n".join(f"{l}: {record['options'][l]}" for l in LETTERS)
    user_message = f"Excerpts:\n\n{excerpts_text}\n\n---\n\nQuestion: {record['stem']}\n\n{options_text}"
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=20,
            # Without this, claude-sonnet-5 sometimes spends its whole
            # max_tokens budget on invisible extended thinking before any
            # text — at max_tokens=10 that's a GUARANTEED empty response
            # whenever it triggers. Measured impact before this fix: 93/258
            # (36%) of this exact call came back unparseable — badly
            # deflating arm D's accuracy, not a rare edge case. Same root
            # cause as translate_heldout_mcq.py / generators.py.
            thinking={"type": "disabled"},
            system=CLAUDE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: Claude call failed for {record['year']} Q{record['q_num']}: {e}")
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    m = re.search(r"[אבגד]", text)
    return m.group(0) if m else None


CLAUDE_WORKERS = 8  # arm D is pure API I/O — sequential calls took >20 min for 258 items on a first run


def run_claude_arm(records: list[dict], translations: dict, retrieved: dict) -> list[dict]:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from dotenv import load_dotenv
    import anthropic
    load_dotenv(BACKEND_DIR / ".env")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _one(r):
        key = f"{r['year']}::{r['q_num']}"
        chunks = pack_chunks_to_budget_no_tokenizer(retrieved[key])
        predicted = claude_choose_letter(client, r, chunks)
        return {**_base_record(r), "predicted": predicted, "option_scores": None}

    out = [None] * len(records)
    done = 0
    with ThreadPoolExecutor(max_workers=CLAUDE_WORKERS) as pool:
        futures = {pool.submit(_one, r): i for i, r in enumerate(records)}
        for fut, i in futures.items():
            out[i] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(records):
                print(f"  [D] {done}/{len(records)}")
    return out


def pack_chunks_to_budget_no_tokenizer(chunks: list[dict], max_chunks: int = RETRIEVE_K) -> list[dict]:
    """Claude's context window makes token budgeting unnecessary — cap at
    the same RETRIEVE_K chunks as B/C so all three RAG arms see the same
    NUMBER of excerpts, even though C's budget can trim it further."""
    return chunks[:max_chunks]


def _base_record(r: dict) -> dict:
    return {
        "year": r["year"], "q_num": r["q_num"], "bucket": r["bucket"],
        "correct": r["correct"], "has_multi_answer": r["has_multi_answer"],
    }


# ------------------------------------------------------------- metrics ---

def compute_metrics(scored: list[dict]) -> dict:
    n = len(scored)
    n_correct = sum(1 for s in scored if s["predicted"] in s["correct"])
    by_bucket = {}
    for bucket in SCORED_BUCKETS:
        subset = [s for s in scored if s["bucket"] == bucket]
        if subset:
            by_bucket[bucket] = {
                "n": len(subset),
                "accuracy": sum(1 for s in subset if s["predicted"] in s["correct"]) / len(subset),
            }

    # Macro-F1 / position bias: single-answer questions only, and only
    # where the model actually produced a letter (Claude parse failures
    # excluded and reported, not silently counted wrong).
    single = [s for s in scored if not s["has_multi_answer"] and s["predicted"] is not None]
    n_unparseable = sum(1 for s in scored if s["predicted"] is None)
    f1_per_letter = {}
    for letter in LETTERS:
        tp = sum(1 for s in single if s["predicted"] == letter and letter in s["correct"])
        fp = sum(1 for s in single if s["predicted"] == letter and letter not in s["correct"])
        fn = sum(1 for s in single if s["predicted"] != letter and letter in s["correct"])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_per_letter[letter] = {"precision": precision, "recall": recall, "f1": f1,
                                  "n_predicted": sum(1 for s in single if s["predicted"] == letter),
                                  "n_correct": sum(1 for s in single if letter in s["correct"])}
    macro_f1 = sum(v["f1"] for v in f1_per_letter.values()) / len(LETTERS)

    return {
        "n": n,
        "accuracy": n_correct / n if n else None,
        "n_unparseable": n_unparseable,
        "by_bucket": by_bucket,
        "macro_f1_single_answer": macro_f1,
        "position_bias": f1_per_letter,
    }


# ----------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="D", help="comma-separated subset of A,B,C,D")
    ap.add_argument("--adapter", default=None, help="path to a PEFT adapter dir (arm C only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit quantization (CPU / debugging)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions (smoke test)")
    args = ap.parse_args()
    arms = [a.strip().upper() for a in args.arms.split(",")]

    records = load_heldout(args.limit)
    translations = load_translations()
    usable, n_no_tr, n_incomplete = usable_records(records, translations)
    print(f"{len(records)} held-out questions (2025+2026, text_answerable+image_dependent)")
    print(f"{len(usable)} usable for MCQ scoring ({n_no_tr} no translation yet, {n_incomplete} incomplete options)")
    if not usable:
        sys.exit("Nothing to score.")

    needs_retrieval = any(a in arms for a in "BCD")
    retrieved = retrieve_batch(usable, translations) if needs_retrieval else {}

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}

    for arm in arms:
        if arm == "A":
            scored = run_local_arm("A", usable, translations, retrieved, None, args.device, not args.no_4bit)
        elif arm == "B":
            scored = run_local_arm("B", usable, translations, retrieved, None, args.device, not args.no_4bit)
        elif arm == "C":
            if not args.adapter:
                sys.exit("--adapter is required for arm C")
            scored = run_local_arm("C", usable, translations, retrieved, args.adapter, args.device, not args.no_4bit)
        elif arm == "D":
            scored = run_claude_arm(usable, translations, retrieved)
        else:
            sys.exit(f"unknown arm {arm!r}")

        metrics = compute_metrics(scored)
        results[arm] = {"metrics": metrics, "records": scored, "adapter": args.adapter if arm == "C" else None}
        print(f"\n=== Arm {arm} ===")
        print(f"  overall accuracy: {metrics['accuracy']:.3f} (n={metrics['n']}, random baseline = 0.25)")
        for bucket, m in metrics["by_bucket"].items():
            print(f"  {bucket}: {m['accuracy']:.3f} (n={m['n']})")
        print(f"  macro-F1 (position bias, single-answer only): {metrics['macro_f1_single_answer']:.3f}")

        # Written after EVERY arm, not once at the end — seen for real on
        # Colab (eval_generation.py's twin of this loop): a crash on arm 2
        # of a 2-arm run used to lose arm 1's already-computed results too,
        # since nothing was persisted until the whole loop finished.
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Written -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
