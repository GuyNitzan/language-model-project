"""
Phase 4: turn the synthetic QA set (eval/data/{train,val,test}.jsonl) into
chat-format instruction-tuning data for the QLoRA fine-tune.

For each question, retrieves top-k context chunks from the Phase 3 winning
config (400_80, bge-small index — free/local/reproducible, no Colab API keys
needed to regenerate) using the *actual* retriever, not the oracle gold
chunk. This matters: Phase 3 measured 400_80/bge's real Hit@5 at ~0.76-0.82,
so a meaningful fraction of examples will have imperfect or noisy retrieved
context, exactly like production. Training the model only on oracle-perfect
context would teach it a task it will never actually face.

Prompt format matches backend/llm.py's `_format_context`/`SYSTEM_PROMPT`
exactly, so the fine-tuned model sees the same input distribution the
product's ClaudeGenerator sees today (the two are meant to be interchangeable
behind the pluggable generator interface). The target completion is the
synthetic set's own `answer` field, which already carries a
"[Chapter: <name>, p.<page>]" citation in the product's own format.

Context chunks are greedily added (by retrieval rank) up to a token budget
computed with the *actual* Qwen2.5-3B-Instruct tokenizer and chat template —
not a word-count heuristic — so every example provably fits `max_seq_len`
with room left for the full answer, which is never truncated (truncating
the answer would silently delete the training signal).

Usage:
    /home/guynitz/venvs/derma_eval/bin/python3 training/prepare_dataset.py

Output: training/data/{train,val,test}.jsonl, each line:
    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", "content": <answer>}],
     "gold_chunk_id": ..., "retrieved_chunk_ids": [...], "gold_in_context": bool,
     "prompt_tokens": int, "total_tokens": int}
"""

import json
import sys
from pathlib import Path

import faiss

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
OUT_DIR = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(EVAL_DIR))
from eval_retrieval import embed_queries, load_chunks, CHUNK_CONFIGS  # noqa: E402

INDEX_PATH = EVAL_DIR / "data" / "indices" / "400_80__bge.faiss"
META_PATH  = EVAL_DIR / "data" / "indices" / "400_80__bge.meta.jsonl"
CHUNKS_PATH = CHUNK_CONFIGS["400_80"]

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# The plan originally set max_seq_len=1024 for T4 memory headroom, but real
# content doesn't fit it: dermatology chunk text tokenizes to a median ~600
# tokens (heavy Latin/technical vocabulary inflates the token count well
# past a word-count estimate), plus per-excerpt location labels ("[Excerpt N
# | <chapter> › <section> | p.N]") and chat-template overhead. At 1024,
# gold-in-context landed at ~51% — matching this config's measured Hit@1 of
# 51.5%, not its Hit@5 of ~82% — because the budget only ever fit ONE
# context chunk, silently collapsing "top-5 retrieval" to "top-1". At 1536
# it barely moved (still ~1 chunk typically: 1 chunk + overhead already
# ~850-950 tokens, and a second ~600-token chunk overshoots a ~1300-token
# budget). Raised to 2048, which reliably fits 2 and often 3 chunks; a T4
# running 4-bit QLoRA on a 3B model at batch=1 + gradient accumulation +
# gradient checkpointing has comfortable headroom at this length.
MAX_SEQ_LEN = 2048
ANSWER_TOKEN_BUFFER = 200   # reserve at least this much for the target answer + special tokens
RETRIEVE_K = 5              # candidate pool to greedily pack from

# When retrieval still doesn't surface the gold chunk within budget, do NOT
# train the model to produce the original citation-grounded answer anyway —
# that would teach it to state facts not actually present in its shown
# context, i.e. exactly the unsupported-citation behavior this project's
# verifier agent (Phase 5) exists to catch. Swap the target to an explicit
# abstention, matching system-prompt rule #2 ("if the excerpts do not
# contain enough information, say so explicitly"). Documented limitation:
# this training-time retrieval depth (2-3 chunks, budget-limited for a 3B
# model) has a lower hit-rate than the deployed product's real top-8
# retrieval, so this training set contains a higher share of abstention
# examples than the model will actually need at production retrieval depth.
ABSTENTION_ANSWER = "The provided excerpts do not contain enough information to answer this question."

# Mirrors backend/llm.py exactly — the fine-tuned model should see the same
# input format the product's ClaudeGenerator does.
SYSTEM_PROMPT = """\
You are a dermatology assistant trained on Bolognia's Dermatology (5th edition, 2024).

Rules:
1. Answer ONLY from the provided book excerpts. Never use outside knowledge.
2. If the excerpts do not contain enough information to answer, say so explicitly.
3. Always cite your sources inline using [Chapter: <name>, p.<page>] notation.
4. Be precise and clinically accurate. Use proper dermatology terminology.
5. Structure longer answers with clear headings when appropriate.
"""


def format_excerpt(i: int, chunk: dict) -> str:
    location = chunk["chapter"]
    if chunk.get("section"):
        location += f" › {chunk['section']}"
    if chunk.get("subsection"):
        location += f" › {chunk['subsection']}"
    page_ref = (f"p.{chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
                else f"pp.{chunk['page_start']}–{chunk['page_end']}")
    return f"[Excerpt {i} | {location} | {page_ref}]\n{chunk['text']}"


def build_user_message(question: str, excerpts: list[str]) -> str:
    context = "\n\n---\n\n".join(excerpts)
    return f"Book excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"


def load_synthetic(split: str) -> list[dict]:
    path = EVAL_DIR / "data" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {BASE_MODEL} (small download, no GPU needed)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading retrieval index ({INDEX_PATH.name})...")
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_chunks(META_PATH)
    chunks_by_id = {c["chunk_id"]: c for c in load_chunks(CHUNKS_PATH)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ["train", "val", "test"]:
        records = load_synthetic(split)
        print(f"\n{split}: {len(records)} questions")

        qvecs = embed_queries([r["question"] for r in records], "bge")
        _, idxs = index.search(qvecs, RETRIEVE_K)

        out_records = []
        n_gold_in_context = 0
        n_truncated_to_fewer = 0
        for r, row in zip(records, idxs):
            candidate_ids = [meta[i]["chunk_id"] for i in row if i != -1]

            # Greedily pack chunks by retrieval rank until the token budget
            # (max_seq_len minus a reserved buffer for the answer) is hit.
            # The full answer is never truncated — only context is capped.
            answer = r["answer"]
            answer_tokens = len(tokenizer.encode(answer))
            budget = MAX_SEQ_LEN - max(answer_tokens, ANSWER_TOKEN_BUFFER)

            packed_ids, excerpts = [], []
            for cid in candidate_ids:
                trial_excerpts = excerpts + [format_excerpt(len(excerpts) + 1, chunks_by_id[cid])]
                user_msg = build_user_message(r["question"], trial_excerpts)
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                trial_tokens = len(tokenizer.encode(prompt_text))
                if trial_tokens > budget and excerpts:
                    break  # this chunk doesn't fit and we already have >=1 — stop here
                excerpts, packed_ids = trial_excerpts, packed_ids + [cid]
                if trial_tokens > budget:
                    break  # took the first chunk even though it's tight; don't add more

            if len(packed_ids) < RETRIEVE_K:
                n_truncated_to_fewer += 1
            gold_in_context = r["gold_chunk_id"] in packed_ids
            if gold_in_context:
                n_gold_in_context += 1

            # See ABSTENTION_ANSWER's docstring above: don't train on a
            # citation-grounded answer the shown context doesn't support.
            target_answer = answer if gold_in_context else ABSTENTION_ANSWER

            user_message = build_user_message(r["question"], excerpts)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": target_answer},
            ]
            prompt_only = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_tokens = len(tokenizer.encode(prompt_only))
            total_tokens = len(tokenizer.encode(full_text))

            out_records.append({
                "messages": messages,
                "gold_chunk_id": r["gold_chunk_id"],
                "retrieved_chunk_ids": packed_ids,
                "gold_in_context": gold_in_context,
                "target_type": "grounded" if gold_in_context else "abstention",
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            })

        out_path = OUT_DIR / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        over_budget = sum(1 for rec in out_records if rec["total_tokens"] > MAX_SEQ_LEN)
        print(f"  gold chunk retrieved in context: {n_gold_in_context}/{len(records)} "
              f"({n_gold_in_context/len(records):.1%})")
        print(f"  examples packed with <{RETRIEVE_K} chunks (budget-limited): {n_truncated_to_fewer}")
        print(f"  examples still over {MAX_SEQ_LEN} tokens total (answer alone exceeded budget): {over_budget}")
        print(f"  written -> {out_path}")
        splits[split] = out_records

    # Leakage check — must never trip; a failure here means Phase 2's split
    # logic itself is broken, so fail loudly rather than hand QLoRA a leaky
    # training set.
    gold_ids = {s: {r["gold_chunk_id"] for r in recs} for s, recs in splits.items()}
    for a in gold_ids:
        for b in gold_ids:
            if a < b:
                overlap = gold_ids[a] & gold_ids[b]
                assert not overlap, f"LEAKAGE: {len(overlap)} gold_chunk_ids in both {a} and {b}"
    print("\nLeakage check passed: train/val/test gold_chunk_ids are disjoint.")


if __name__ == "__main__":
    main()
