"""
Phase 5: open-ended generation quality, arms A-D, on two sets:

  synthetic test set (eval/data/test.jsonl, 99 items, English, has a
      reference answer)          -> ROUGE-L, BLEU, citation accuracy
                                     (backend/verifier.py), LLM-judge
                                     correctness/groundedness (judge.py)
  out-of-corpus set (eval/data/ood_questions.json, 24 items, no reference,
      no correct answer exists)  -> abstention rate (the honest metric:
                                     correct behaviour here is refusal)

Retrieval is fixed the same way eval_mcq.py fixes it: one config
(400_80/bge) shared by arms B/C/D so the comparison isolates the
generator, not different context. The synthetic set's questions are
already English (Phase 2 generated them that way), so — unlike
eval_mcq.py — no translation step is needed here.

Usage (arms A/B/C need a GPU; see notebooks/04_generation_eval.ipynb):
    python3 eval_generation.py --arms D --set test
    python3 eval_generation.py --arms D --set ood
    python3 eval_generation.py --arms A,B,C --adapter <path> --set test --device cuda

Output: eval/data/generation_results.json, merged by (arm, set).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss

EVAL_DIR = Path(__file__).parent
BACKEND_DIR = EVAL_DIR.parent / "backend"
PIPELINE_DIR = EVAL_DIR.parent / "pipeline"
CHUNKS_PATH = PIPELINE_DIR / "data" / "chunks" / "chunks.jsonl"
INDEX_PATH = EVAL_DIR / "data" / "indices" / "400_80__bge.faiss"
META_PATH = EVAL_DIR / "data" / "indices" / "400_80__bge.meta.jsonl"
TEST_PATH = EVAL_DIR / "data" / "test.jsonl"
OOD_PATH = EVAL_DIR / "data" / "ood_questions.json"
RESULTS_PATH = EVAL_DIR / "data" / "generation_results.json"

sys.path.insert(0, str(EVAL_DIR))
from eval_retrieval import embed_queries, load_chunks  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
load_dotenv(BACKEND_DIR / ".env")

sys.path.insert(0, str(BACKEND_DIR))
import generators  # noqa: E402
import verifier    # noqa: E402

import judge  # noqa: E402

RETRIEVE_K = 5
MAX_SEQ_LEN = 2048
ANSWER_TOKEN_BUFFER = 300  # generous — unlike MCQ options, full answers need room
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CLAUDE_MODEL = "claude-sonnet-5"

# The literal string training/prepare_dataset.py trains the LoRA arm to
# produce when it can't answer — matched here (and via generators.ABSTENTION_ANSWER)
# to detect abstention in A/B/C's raw generations without another API call.
ABSTENTION_MARKERS = ["do not contain enough information", "does not contain enough information",
                      "insufficient information", "cannot answer", "excerpts do not"]


def is_abstention_text(answer: str) -> bool:
    lower = answer.lower()
    return any(marker in lower for marker in ABSTENTION_MARKERS)


# --------------------------------------------------------------- loading ---

def load_test_set(limit: int | None = None) -> list[dict]:
    with TEST_PATH.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return recs[:limit] if limit else recs


def load_ood_set(limit: int | None = None) -> list[dict]:
    recs = json.loads(OOD_PATH.read_text(encoding="utf-8"))
    return recs[:limit] if limit else recs


# ------------------------------------------------------------- retrieval ---

_chunks_by_id = _index = _meta = None


def _load_retrieval_backend():
    global _chunks_by_id, _index, _meta
    if _index is not None:
        return
    if not INDEX_PATH.exists():
        sys.exit(f"No index at {INDEX_PATH} — run build_index.py --config 400_80 --embedder bge first.")
    _index = faiss.read_index(str(INDEX_PATH))
    _meta = load_chunks(META_PATH)
    _chunks_by_id = {c["chunk_id"]: c for c in load_chunks(CHUNKS_PATH)}


def retrieve_batch(queries: list[str], k: int = RETRIEVE_K) -> list[list[dict]]:
    _load_retrieval_backend()
    qvecs = embed_queries(queries, "bge")
    _, idxs = _index.search(qvecs, k)
    out = []
    for row in idxs:
        chunk_ids = [_meta[i]["chunk_id"] for i in row if i != -1]
        out.append([_chunks_by_id[cid] for cid in chunk_ids if cid in _chunks_by_id])
    return out


def pack_chunks_to_budget(tokenizer, candidate_chunks: list[dict], question: str,
                           reserve: int = ANSWER_TOKEN_BUFFER) -> list[dict]:
    """Same greedy packer as eval_mcq.py / training/prepare_dataset.py."""
    budget = MAX_SEQ_LEN - reserve
    packed = []
    for c in candidate_chunks:
        trial = packed + [c]
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": generators.SYSTEM_PROMPT},
             {"role": "user", "content": generators.build_user_message(question, trial)}],
            tokenize=False, add_generation_prompt=True,
        )
        if len(tokenizer.encode(prompt_text)) > budget and packed:
            break
        packed = trial
        if len(tokenizer.encode(prompt_text)) > budget:
            break
    return packed


# ------------------------------------------------------------- text metrics ---

def rouge_l(reference: str, candidate: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)["rougeL"].fmeasure


def bleu(reference: str, candidate: str) -> float:
    import sacrebleu
    return sacrebleu.sentence_bleu(candidate, [reference]).score / 100.0


# ------------------------------------------------------------------ arms ---

def get_generator_for_arm(arm: str, adapter_path: str | None, device: str, load_in_4bit: bool):
    if arm == "D":
        return generators.ClaudeGenerator(model=CLAUDE_MODEL)
    return generators.LoRAGenerator(
        adapter_path=adapter_path if arm == "C" else None,
        device=device, load_in_4bit=load_in_4bit,
    )


def run_test_set(arm: str, gen, records: list[dict]) -> list[dict]:
    context_by_q = {}
    if arm != "A":
        retrieved = retrieve_batch([r["question"] for r in records])
        for r, chunks in zip(records, retrieved):
            if arm in ("B", "C"):  # Qwen's fixed input budget — see pack_chunks_to_budget's docstring
                chunks = pack_chunks_to_budget(gen.tokenizer, chunks, r["question"])
            else:  # D (Claude): full top-k, uncapped — matches the product's real retrieval depth
                chunks = chunks[:RETRIEVE_K]
            context_by_q[r["question"]] = chunks
    else:
        context_by_q = {r["question"]: [] for r in records}

    def _score_one(r):
        chunks = context_by_q[r["question"]]
        answer = gen.answer(r["question"], chunks, [])
        v = verifier.verify(answer, chunks)
        j = judge.judge(r["question"], r["answer"], answer, chunks)
        return {
            "question": r["question"], "reference": r["answer"], "candidate": answer,
            "n_chunks": len(chunks),
            "rouge_l": rouge_l(r["answer"], answer),
            "bleu": bleu(r["answer"], answer),
            "is_abstention": is_abstention_text(answer),
            "verification": v,
            "judge": j,
        }

    return _run_scored(arm, records, _score_one, "test")


def run_ood_set(arm: str, gen, records: list[dict]) -> list[dict]:
    questions = [r["question"] for r in records]
    context_by_q = {}
    if arm != "A":
        retrieved = retrieve_batch(questions)
        for r, chunks in zip(records, retrieved):
            if arm in ("B", "C"):
                chunks = pack_chunks_to_budget(gen.tokenizer, chunks, r["question"])
            else:
                chunks = chunks[:RETRIEVE_K]
            context_by_q[r["question"]] = chunks
    else:
        context_by_q = {q: [] for q in questions}

    def _score_one(r):
        chunks = context_by_q[r["question"]]
        answer = gen.answer(r["question"], chunks, [])
        v = verifier.verify(answer, chunks)
        return {
            "question": r["question"], "domain": r.get("domain"),
            "candidate": answer, "n_chunks": len(chunks),
            "is_abstention": is_abstention_text(answer) or v.get("is_abstention", False),
            "verification": v,
        }

    return _run_scored(arm, records, _score_one, "ood")


GENERATOR_WORKERS = 8  # only used for arm D — pure API I/O, same rationale as eval_mcq.py's CLAUDE_WORKERS


def _safe_score(score_one, r: dict) -> dict:
    """One bad record must not cost the whole run's already-scored records —
    seen for real on Colab: a malformed verifier response for a single
    synthetic-test item raised out of score_one() and killed the entire
    arm/set (and, before results were persisted per-arm-and-set, took
    already-completed arms down with it too). Record the failure inline
    instead of propagating it."""
    try:
        return score_one(r)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: scoring failed for {r.get('question', r)!r}: {e}")
        return {"question": r.get("question"), "error": str(e)}


def _run_scored(arm: str, records: list[dict], score_one, label: str) -> list[dict]:
    """arm D is pure Claude-API I/O, safe (and much faster) to fan out with
    threads; A/B/C hold the single local GPU model, so those stay
    sequential — concurrent .generate() calls on one CUDA device would just
    serialize anyway, with none of the benefit and more risk."""
    if arm != "D":
        out = []
        for i, r in enumerate(records, 1):
            out.append(_safe_score(score_one, r))
            if i % 10 == 0 or i == len(records):
                print(f"  [{arm}/{label}] {i}/{len(records)}")
        return out

    from concurrent.futures import ThreadPoolExecutor
    out = [None] * len(records)
    done = 0
    with ThreadPoolExecutor(max_workers=GENERATOR_WORKERS) as pool:
        futures = {pool.submit(_safe_score, score_one, r): i for i, r in enumerate(records)}
        for fut, i in futures.items():
            out[i] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(records):
                print(f"  [{arm}/{label}] {done}/{len(records)}")
    return out


# ------------------------------------------------------------- summaries ---

def _split_failed(scored: list[dict]) -> tuple[list[dict], int]:
    """_safe_score() replaces a record that raised with a bare
    {"question", "error"} stub — exclude those from metric denominators
    (reported honestly via n_failed) rather than letting a missing key
    (KeyError) or a silently-wrong 0 skew the average."""
    ok = [s for s in scored if "error" not in s]
    return ok, len(scored) - len(ok)


def summarize_test(scored: list[dict]) -> dict:
    ok, n_failed = _split_failed(scored)
    n = len(ok)
    judged = [s for s in ok if "error" not in s["judge"]]
    return {
        "n": n,
        "n_failed": n_failed,
        "mean_rouge_l": sum(s["rouge_l"] for s in ok) / n if n else None,
        "mean_bleu": sum(s["bleu"] for s in ok) / n if n else None,
        "abstention_rate": sum(s["is_abstention"] for s in ok) / n if n else None,
        "mean_citation_accuracy": _mean_groundedness(ok),
        "n_judged": len(judged),
        "mean_judge_correctness": sum(j["judge"]["correctness"] for j in judged) / len(judged) if judged else None,
        "mean_judge_groundedness": sum(j["judge"]["groundedness"] for j in judged) / len(judged) if judged else None,
    }


def _mean_groundedness(scored: list[dict]) -> float | None:
    scores = [s["verification"]["groundedness_score"] for s in scored
              if s["verification"].get("groundedness_score") is not None]
    return sum(scores) / len(scores) if scores else None


def summarize_ood(scored: list[dict]) -> dict:
    ok, n_failed = _split_failed(scored)
    n = len(ok)
    return {
        "n": n,
        "n_failed": n_failed,
        "abstention_rate": sum(s["is_abstention"] for s in ok) / n if n else None,
        # A confident (non-abstaining), well-cited answer to an out-of-corpus
        # question would still show high "groundedness" if the model cites a
        # plausible-but-irrelevant excerpt — this is the confident-guess
        # failure mode the plan's Analysis section wants surfaced separately.
        "mean_groundedness_when_not_abstaining": _mean_groundedness(
            [s for s in ok if not s["is_abstention"]]
        ),
    }


# ----------------------------------------------------------------- main ---

def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a (0 scored)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="D")
    ap.add_argument("--set", choices=["test", "ood", "both"], default="both")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    arms = [a.strip().upper() for a in args.arms.split(",")]
    sets = ["test", "ood"] if args.set == "both" else [args.set]

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}

    for arm in arms:
        if arm == "C" and not args.adapter:
            sys.exit("--adapter is required for arm C")
        gen = get_generator_for_arm(arm, args.adapter, args.device, not args.no_4bit)
        results.setdefault(arm, {})

        # Written after every (arm, set) — NOT once at the end. Seen for
        # real on Colab: arm B crashed partway through the "test" set (a
        # malformed verifier response — see backend/verifier.py), which
        # took down the whole `--arms A,B` invocation and silently discarded
        # arm A's already-computed, already-printed results along with it,
        # since nothing had been persisted yet.
        if "test" in sets:
            records = load_test_set(args.limit)
            scored = run_test_set(arm, gen, records)
            summary = summarize_test(scored)
            results[arm]["test"] = {"summary": summary, "records": scored}
            print(f"\n=== Arm {arm} / synthetic test (n={summary['n']}, failed={summary['n_failed']}) ===")
            print(f"  ROUGE-L: {_fmt(summary['mean_rouge_l'])}  BLEU: {_fmt(summary['mean_bleu'])}")
            print(f"  citation accuracy (verifier): {summary['mean_citation_accuracy']}")
            print(f"  judge correctness/groundedness: {summary['mean_judge_correctness']} / {summary['mean_judge_groundedness']}")
            print(f"  abstention rate: {_fmt(summary['abstention_rate'])}")
            RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Written -> {RESULTS_PATH}")

        if "ood" in sets:
            records = load_ood_set(args.limit)
            scored = run_ood_set(arm, gen, records)
            summary = summarize_ood(scored)
            results[arm]["ood"] = {"summary": summary, "records": scored}
            print(f"\n=== Arm {arm} / out-of-corpus (n={summary['n']}, failed={summary['n_failed']}) ===")
            print(f"  abstention rate (correct behaviour = high): {_fmt(summary['abstention_rate'])}")
            print(f"  mean groundedness when NOT abstaining (confident-guess check): "
                  f"{summary['mean_groundedness_when_not_abstaining']}")
            RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Written -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
