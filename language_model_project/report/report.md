# DermaForce: RAG + QLoRA for Hebrew Dermatology Board-Exam QA

*Applied Language Models — Group Project B (solo submission)*

## 1. Introduction & motivation

We extend a working Hebrew-to-English RAG chatbot over *Bolognia's Dermatology*
(5th ed.) into a measured research system, and evaluate it against a source of
ground truth most course RAG projects don't have: **949 real questions from
seven sittings (2021–2026) of the Israeli Medical Association's Stage-A
dermatology board exam**, each with a human expert's page/chapter citation
into the same textbook already indexed. This closes the main methodological
gap of a synthetic-only evaluation (a model grading its own generated
questions) and adds the assignment's two hard requirements a pure product
build doesn't need: a measured retrieval ablation and a QLoRA fine-tune.

The exam data also exposes two real, quantifiable ceilings for a text-only
RAG system: **image-dependent questions** (17% of the real corpus — a
generator-side ceiling) and **table/figure-pointer references** (a
retriever-side ceiling, since captions are dropped at extraction time). Both
are measured below rather than treated as excuses.

## 2. Datasets & preprocessing

| | Bolognia corpus + synthetic QA | Israeli board-exam corpus |
|---|---|---|
| Source | *Bolognia Dermatology* 5th ed., 3,204 pages | 7 sittings, 2021–2026, IMA Stage-A |
| Size | 6,115 chunks (400w/80-overlap) → 1,219 synthetic QA items | 949/950 MCQs parsed |
| Ground truth | Claude-generated, gold = exact source chunk | **Human expert page/chapter citation** — independent |
| Split | 1,000 train / 120 val / 99 test, by chunk (zero leakage) | Temporal: develop 2021–2024 (n=517 text-answerable) / **hold out 2025+2026** (n=258 usable) |
| Licensing | Elsevier — gitignored, small sample committed | IMA — gitignored, small sample committed |

Parsing the exam corpus required per-file format detection across the seven
years: two numbering conventions (`N.` vs. a reversed `.N`), two distinct
broken-glyph bugs, three reference-file structures (tabular page-only / prose
book+page / prose book+chapter+page), one cross-version alignment table
(2023), and crop-box intersection (not `get_images()`) for image-page
association, since one year's PDF shares its content stream across all 44
pages. `pipeline/02_chunk.py` was also fixed: the original chunker
accumulated whole pages until crossing a word target, but a Bolognia page is
~700 words, so the window always overshot on page 1 and closed immediately —
median 766 words/chunk with **zero effective overlap**. The rewritten
word-stream chunker produces an exact 400-word median with exact 80-word
overlap on every pair.

## 3. Model design

```
  Hebrew MCQ ──▶ glyph fix ──▶ translate to EN ──▶ retrieve (EN corpus) ──▶ score 4 options
                    PRODUCT: Qdrant + OpenAI embeds → Claude → answer → verifier badge
                    MEASUREMENT: FAISS index variants → retrieval metrics
                                  Claude-distilled QA → QLoRA Qwen2.5-3B-Instruct
```

Four design decisions: (1) FAISS for measurement (no infra to reproduce),
Qdrant stays for the product; (2) translate the *query*, not the 2,521-chunk
corpus — three orders of magnitude cheaper; (3) score MCQ options by
length-normalised log-likelihood, not by generating a letter — the standard
way to make a no-RAG base-model arm comparable; (4) `backend/llm.py` behind a
`Generator` protocol (`ClaudeGenerator` / `LoRAGenerator`), so the
baseline-vs-fine-tuned comparison is a live, demoable flip in the product UI.

Four evaluation arms, on the 258 held-out 2025+2026 questions: **A.**
Qwen2.5-3B base, no RAG (closed-book floor) · **B.** base + RAG (retrieval
alone) · **C.** LoRA + RAG (the fine-tuning deliverable) · **D.** Claude + RAG
(upper bound / teacher — cannot use log-likelihood scoring since Anthropic
exposes no logprobs, so it generates the Hebrew letter directly reading the
un-translated stem: a documented, deliberate scoring asymmetry).

## 4. Training & metrics

**Retrieval** (`eval/eval_retrieval.py`): 3 chunk configs × 2 embedders
(`bge-small-en-v1.5` free, `text-embedding-3-large` paid) × 3 query-language
variants × reranker on/off, scored as Hit@{1,3,5,10} and MRR@10 against two
independent gold sources (synthetic exact-chunk, expert page/chapter
citation).

**QLoRA fine-tune** (`training/`): `Qwen2.5-3B-Instruct`, 4-bit NF4, LoRA rank
∈ {8, 16, 32} (α=2r, dropout 0.05, lr 2e-4, 2 epochs), trained on Colab.
Target: citation-grounded answers in the product's own `[Chapter, p.N]`
format, distilled from Claude, with training examples where the actual
retriever misses the gold chunk relabeled to an explicit abstention — so the
model isn't taught to state facts absent from its own shown context.

**Generation eval**: MCQ accuracy (likelihood-scored), macro-F1 (position
bias), Hit@k/MRR against expert references, abstention rate on 24
genuinely-out-of-corpus questions, ROUGE-L/BLEU/citation-accuracy/LLM-judge
groundedness on the synthetic test slice.

## 5. Results

**4-arm MCQ accuracy** (n=258; random floor 0.25; ceiling = 86.3% given 47/258
questions are image-dependent and unanswerable from text):

| Arm | Accuracy | Text-answerable | Image-dependent | % of ceiling |
|---|---|---|---|---|
| A. base, no RAG | 0.225 | 0.218 | 0.255 | 26.0% |
| B. base + RAG | 0.302 | 0.308 | 0.277 | 35.0% |
| C. LoRA r16 + RAG | 0.318 | 0.313 | 0.340 | 36.8% |
| D. Claude + RAG | **0.721** | 0.754 | 0.574 | **83.5%** |

Retrieval alone buys arm B +7.7pp over the closed-book floor; the LoRA
fine-tune buys a further +1.6pp — real but modest, consistent with r=32's
measured overfitting (see below) and with a shallower training-time
retrieval depth (2–3 chunks at a 2048-token budget) than the deployed
product's top-8. Claude (D) sits 40 points above C, and framed against the
ceiling — 83.5% of everything a text-only system could get right, not "72%
accurate" — the gap is concentrated in translation/retrieval noise and
reasoning depth, not architecture.

**Cross-lingual retrieval** — the headline experiment: translating the
Hebrew query to English before retrieval buys `bge-small` (English-only)
**+19% relative Hit@10** (0.106→0.124→0.128 across raw/translated/+glossary);
`text-embedding-3-large` (real cross-lingual capability) stays flat
(0.114→0.112→0.116). Two hand-read cases confirm the mechanism: a raw-Hebrew
query for an atopic-dermatitis question retrieves top-1 from *Fungal
Diseases* (unrelated); translated, it retrieves top-1 from *Atopic
Dermatitis* — the English-only embedder has no lexical grip on untranslated
Hebrew, and translation restores literal term overlap it can use directly.

**Reranking** (`ms-marco-MiniLM-L-6-v2`) improved MRR@10 in **every one of 6
query-set × embedder combinations tested** — a clean, low-risk addition.
**Table/figure-pointer references** (37/517 dev questions) retrieve
**3.7–4.6× worse** than ordinary page references (Hit@1 = 0% in both query
languages) — the direct cost of dropping figure captions at extraction time,
a distinct, retriever-side failure mode from the generator-side
image-dependent-question ceiling.

**Position bias**: the exam's own answer distribution is skewed (option א
correct 75/258 times, ד only 42 — a real exam-writing convention). Arm A
shows genuine *model* bias on top of that, over-predicting the last option
2× its true rate; arm D tracks the true distribution almost exactly
(macro-F1 ≈ accuracy), consistent with larger models showing less MCQ
selection bias.

**QLoRA loss curves**: val loss plateaus flat (~0.29–0.31) for all three
ranks from step ~50; train loss keeps dropping fastest for r=32 — the
train/val gap widening monotonically with rank is a measured overfitting
signal, not a prediction. r=16 was carried into the 4-arm comparison as the
rank least likely to be overfit while retaining real capacity.

**Two measurement artifacts caught before being reported as findings**:
multi-hop references score *better* on Hit@1/MRR (0.087 vs. 0.028) only
because they resolve to a 1.7× larger gold-chunk set on average — not because
retrieval reasons across locations better; and 97.3% of exam-dev questions
overlap the synthetic set's sampled content (stratified sampling already
touches all 160 chapters), which rules out "coverage gap" as the explanation
for the synthetic-vs-exam accuracy gap and leaves the already-documented one
standing: exam citations point at *where a topic is discussed*, not an exact
paragraph.

## 6. Limitations

Translation is a real, now-demonstrated (not just aggregate) error source,
though translation calls themselves rarely fail outright (≤1/500–1200 across
every batch run). The printed-to-PDF page map is an interpolation (122/160
chapters anchored, rest interpolated; validated to 1–2 pages), so every
non-chapter-tagged reference inherits residual uncertainty. Arm D's
letter-generation scoring is not directly comparable to A/B/C's
log-likelihood scoring. Both corpora are copyrighted and not redistributed.
Image-dependent questions are a hard architectural ceiling for any text-only
system, quantified rather than assumed.

## 7. Conclusion & future work

A real RAG product, measured honestly against independent human ground
truth rather than its own generated questions, clears the assignment's
retrieval-ablation and fine-tuning requirements with a full 4-arm comparison
and a verifier agent that demonstrably distinguishes grounded claims from
fabricated citations. The largest open levers, in order of expected payoff:
re-including figure captions as retrievable text (raises the measured
ceiling rather than just describing it), a deeper training-time retrieval
budget for the LoRA arm (closer to the product's real top-8), and a
larger/better-curated multi-hop and table/figure evaluation slice now that
both are cleanly isolated as distinct failure classes.
