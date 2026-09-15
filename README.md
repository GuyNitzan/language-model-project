# DermaForce — RAG + QLoRA for Dermatology Board-Exam Question Answering

> **Educational project, not a clinical tool.** This system is a course project on
> retrieval-augmented generation and parameter-efficient fine-tuning. It is not
> validated for clinical use, must not be used for real diagnostic or treatment
> decisions, and its outputs should not be trusted over a qualified physician.

A Hebrew-language dermatology board-exam question-answering system built on a
retrieval-augmented generation (RAG) pipeline over *Bolognia's Dermatology* (5th
ed.), extended with a measurement path (retrieval ablations, QLoRA fine-tuning,
a 4-arm generation comparison) and evaluated against 949 real questions from
seven sittings (2021–2026) of the Israeli Medical Association's Stage-A
dermatology board exam.

The full working design log — every decision, bug, and measurement, in the
order it happened — is in **[`derma_guide_plan.md`](derma_guide_plan.md)**.
This README is the submission-facing summary of it.

---

## 1. System design

```
  Hebrew MCQ ──▶ glyph fix ──▶ translate to EN ──▶ retrieve (EN corpus) ──▶ score 4 options ──▶ א/ב/ג/ד
                                     │                                              │
                                     └─ compare: raw-Hebrew query vs translated ────┘   (cross-lingual experiment)

                    ┌─ PRODUCT (exists) ────────────────────────┐
  question ────────▶│ Qdrant + OpenAI embeds → Claude → answer  │──▶ Next.js UI
                    │                          ↳ Verifier agent │
                    └───────────────────────────────────────────┘

                    ┌─ MEASUREMENT (built for this course project) ─┐
  chunks.jsonl ────▶│ FAISS index variants → retrieval metrics      │──▶ plots
       └───────────▶│ Claude distills QA set → QLoRA Qwen2.5-3B     │──▶ curves + tables
                    └────────────────────────────────────────────┘
```

- **Product path** (`backend/`, `frontend/`, `pipeline/`) is a real, working
  RAG chatbot: Qdrant + OpenAI embeddings + Claude, with citations and a
  citation-groundedness **verifier agent** surfaced live in the UI.
- **Measurement path** (`eval/`, `training/`) is FAISS-based (no infra needed
  to reproduce), and adds the fine-tuning stage the product doesn't need:
  QLoRA-adapted `Qwen2.5-3B-Instruct`, compared against the Claude-backed
  product across four arms (below).

---

## 2. Datasets

### 2a. Bolognia corpus + synthetic training QA

| | |
|---|---|
| Source | *Bolognia Dermatology*, 5th ed. (Elsevier), 3,204 extracted pages |
| Chunking | 400-word window, 80-word overlap (fixed from an accidental ~770w/0-overlap chunker — see `derma_guide_plan.md` Phase 2) → **6,115 chunks** |
| Synthetic QA | Claude-generated, one question per sampled chunk, citation-grounded answer in the product's own `[Chapter, p.N]` format |
| Split | **1,000 train / 120 val / 99 test**, split *by chunk* before generation (zero cross-split chunk overlap, asserted) |
| Role | Training data for the QLoRA fine-tune and the retrieval eval's "easy" (gold-chunk) arm |
| Licensing | Elsevier-copyrighted — **not redistributed**; `data/sample_chunks.jsonl` (~20 chunks) ships as a demonstrable sample |

### 2b. Israeli Medical Association board-exam questions (the real test set)

| | |
|---|---|
| Source | Stage-A dermatology board exam (מחלות עור ומין, שלב א'), 7 sittings, 2021–2026 |
| Volume | **949 of 950 downloaded MCQs parsed** (150×5 + 100×2; 2020 dropped, 2022/2023 kept version 1 only) |
| Buckets | 748 text-answerable · 165 image-dependent · 25 excluded (Lever-referenced, per instruction) · 11 unresolvable |
| Ground truth | **Human expert page/chapter references**, independent of any model — not circular, unlike the synthetic set |
| Split | Temporal: **develop on 2021–2024 (n=517 text-answerable), hold out 2025+2026 completely (n=258 usable)** |
| Role | The honest evaluation set — accuracy, Hit@k/MRR against real citations |
| Licensing | Israeli Medical Association material — **not redistributed**; a small redacted sample can be added at `data/exams/2026/parsed_sample.json` |

Parsing this corpus required per-file format detection across the seven years
(two numbering conventions, two different broken-glyph bugs, three reference-file
structures, one cross-version alignment table for 2023, and a crop-box-intersection
fix for image-page association) — see `derma_guide_plan.md` Phase 1 for the full
audit. Both datasets combined clear the assignment's "own dataset ≥1,000 examples"
bar, and unlike the synthetic set, every exam question is real.

---

## 3. Methods

1. **Chunking fix** (`pipeline/02_chunk.py`) — word-stream sliding window with
   exact size/overlap, replacing an accidental page-boundary chunker.
2. **Retrieval ablation** (`eval/build_index.py`, `eval/eval_retrieval.py`) —
   3 chunk configs × 2 embedders (`bge-small-en-v1.5` free, OpenAI
   `text-embedding-3-large` paid) × 3 query-language variants (raw Hebrew,
   Claude-translated English, translated + glossary) × reranker
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on/off, evaluated against **two
   independent gold-label sources** (synthetic gold-chunk vs. expert exam
   references).
3. **QLoRA fine-tune** (`training/`, `notebooks/03_finetune.ipynb`) —
   `Qwen2.5-3B-Instruct`, 4-bit NF4, LoRA rank ∈ {8, 16, 32}, trained on Colab
   (this dev sandbox is CPU-only). Target: citation-grounded answers,
   including explicit abstention when the retrieved context doesn't cover
   the question.
4. **4-arm generation/MCQ evaluation** (`eval/eval_mcq.py`,
   `eval/eval_generation.py`, `notebooks/04_generation_eval.ipynb`) on the
   258 held-out 2025+2026 questions:

   | Arm | Isolates |
   |---|---|
   | A. Qwen2.5-3B base, no RAG | Closed-book floor |
   | B. Qwen2.5-3B base + RAG | Value of retrieval alone |
   | C. Qwen2.5-3B LoRA + RAG | Value of fine-tuning |
   | D. Claude + RAG | Upper bound / the teacher |

   A/B/C are scored by length-normalised log-likelihood of each MCQ option
   (no logprobs API for D — it generates the Hebrew letter directly reading
   the original stem, a documented methodological asymmetry).
5. **Verifier agent** (`backend/verifier.py`) — after generation, extracts
   each claim + citation and checks it against the retrieved chunks; returns
   a groundedness score, shown as a live UI badge (`frontend/`).

---

## 4. Results

### 4-arm MCQ accuracy, held-out 2025+2026 exams (n=258)

![4-arm MCQ accuracy](outputs/plots/mcq_accuracy_by_arm_bucket.png)

| Arm | Accuracy | Text-answerable | Image-dependent | % of ceiling captured |
|---|---|---|---|---|
| A. base, no RAG | 0.225 | 0.218 | 0.255 | 26.0% |
| B. base + RAG | 0.302 | 0.308 | 0.277 | 35.0% |
| C. LoRA r16 + RAG | 0.318 | 0.313 | 0.340 | 36.8% |
| D. Claude + RAG | **0.721** | 0.754 | 0.574 | **83.5%** |

Ceiling = `(211/258)×100% + (47/258)×25%` = **0.863** — the maximum any
text-only system can score once 47/258 questions require an image it can't
see. Framed against that ceiling, arm D isn't "72% accurate," it's capturing
**84% of everything a text-only system could possibly get right** on this exam.

### Cross-lingual retrieval — translate before you embed

![Cross-lingual Hit@10](outputs/plots/cross_lingual_hit10.png)

`bge-small-en-v1.5` (English-only) gains **+19% relative Hit@10** from
translating the Hebrew query to English first; `text-embedding-3-large`
(real cross-lingual capability) stays essentially flat. Two worked examples
of *why* (`eval/data/translation_case_studies.json`): a raw-Hebrew query for
an atopic-dermatitis question retrieves top-1 from *Fungal Diseases*
(unrelated); the translated query retrieves top-1 from *Atopic Dermatitis*
(exactly right) — the English-only embedder simply has no lexical grip on
untranslated Hebrew.

### Chunking config and the reranker

![Hit@k by chunk config](outputs/plots/hit_at_k_by_chunk_config.png)
![Reranker before/after](outputs/plots/reranker_before_after.png)

`400w/80-overlap` (the Phase 2 chunking fix's direct output) is carried
forward — not the highest-scoring config, deliberately: the synthetic and
exam ground truths bias the size metric in *opposite* directions (denser
chunking mechanically inflates synthetic Hit@k; wider chunks mechanically
inflate exam Hit@k via the page-tolerance match), so `400_80` sitting between
both artifacts and never being the worst on either is the honest choice, not
a cherry-picked one. The reranker (`ms-marco-MiniLM-L-6-v2`) improves MRR@10
in **every single query set tested**, for both embedders — a clean, low-risk
addition.

### Position bias in MCQ scoring

![Position bias](outputs/plots/position_bias.png)

The exam's own gold-answer distribution is skewed (option א correct 75/258
times, ד only 42) — real exam-writing convention, not a model artifact.
Against that backdrop, the 3B base model (arm A) shows genuine *model*
position bias, over-predicting the last option (ד) roughly 2× its true rate;
Claude (arm D) tracks the true distribution almost exactly (macro-F1 ≈
accuracy), consistent with larger models showing far less MCQ selection bias.

### A distinct, worse retrieval-failure class

![Table/figure vs normal page pointer](outputs/plots/table_figure_vs_normal.png)

References that point at a specific table or figure (37/517 dev questions)
retrieve **3.7–4.6× worse** than ordinary page-range references, with
**Hit@1 = 0%** in both query languages — the direct, measured consequence of
the extraction pipeline dropping figure captions and dense table text at
chunking time. This is a *retriever* ceiling, distinct from the
*generator*-side image-dependent-question ceiling above.

### QLoRA fine-tuning

Three LoRA ranks (8, 16, 32) trained end-to-end on Colab
(`training/`, `notebooks/03_finetune.ipynb`; loss curves at
`notebooks/03_finetune_output.ipynb`). Val loss plateaus flat (~0.29–0.31)
across all three ranks from step ~50 on, while train loss keeps dropping
fastest for r=32 — the widening train/val gap scaling monotonically with rank
is a clean, measured overfitting signal (not just a prediction). Rank 16 was
carried into the 4-arm comparison as the rank least likely to be overfit
while still showing real capacity.

*(The rendered `loss_curves.png` lives only on the Google Drive this
training ran against — see Phase 4 in `derma_guide_plan.md` for the full
numeric write-up; it isn't checked into this repo copy.)*

### Two measurement artifacts, caught rather than reported at face value

- **Multi-hop references "score better"** (Hit@1 0.087 vs 0.028, hebrew_raw)
  — but only because they resolve to a **1.7× larger gold-chunk set on
  average** (21.5 vs 12.4 chunks), not because retrieval handles multi-hop
  reasoning better. Reported as an artifact, not a finding.
- **97.3% of exam-dev questions overlap the synthetic set's sampled content**
  (stratified sampling across all 160 chapters already gets there) — so the
  large synthetic-vs-exam accuracy gap is **not a coverage gap**; it's the
  same measurement (approximate exam citations vs. exact synthetic gold),
  seen twice.

Full numeric detail for every result above — including the two metric
artifacts and the four-way retrieval grid — is in `derma_guide_plan.md`,
Phases 3 and 5. Raw data: `eval/data/*.json`. Plots: `outputs/plots/`.

---

## 5. Qualitative examples

**Grounded, correctly-scoped answer** (arm D, synthetic test set) — the model
cites its source and, when the source only partially answers the question,
says so instead of filling the gap from parametric knowledge:

> *Q: What is the most common valvular abnormality associated with rheumatic
> heart disease?*
> *A: ...in rheumatic fever, the late manifestations include mitral and/or
> aortic valve dysfunction [Chapter 53: Dermatologic Manifestations in
> Patients with Systemic Disease, p.1051]. The excerpt does not further
> specify which of the two... is more common, so I cannot state definitively
> from this text...*

**Correct abstention on a genuinely out-of-corpus question** (arm D,
out-of-corpus set, 21/24 correctly abstained):

> *Q: Which coronary artery is most frequently occluded in an anterior wall
> myocardial infarction?*
> *A: The provided excerpts do not contain information about coronary artery
> anatomy or myocardial infarction. They focus on cutaneous microvascular
> occlusion syndromes... I cannot answer this question from the provided
> material.*

**Verifier catching a real hallucination** (`backend/verifier.py`, spot-check
in Phase 5): a fabricated citation attached to an unrelated claim
(`[Chapter: 12 Psoriasis, p.999]`) is correctly flagged
`citation_valid=false, supported=false`, while a real, correctly-cited claim
scores `groundedness_score=1.0` — the verifier distinguishes real grounding
from real hallucination, not just JSON shape.

---

## 6. Limitations

- **Translation is a real, demonstrated error source** — concrete cases
  where raw-Hebrew retrieval misses entirely and translated retrieval hits
  cleanly (§4) — though the translation step itself rarely fails outright
  (≤1/500-1200 calls across every batch run).
- **The page map is an interpolation, not ground truth** — printed→PDF page
  mapping anchors 122/160 chapters from the table of contents and
  interpolates the rest; validated to within 1–2 pages on hand-checked
  points, but every reference for the 5/7 exam years with no chapter field
  inherits this residual uncertainty.
- **Arm D is not scored identically to A/B/C** — D generates the answer
  letter directly (Anthropic's API exposes no logprobs); A/B/C are scored by
  length-normalised log-likelihood. A genuine asymmetry, not an oversight.
- **Two Hit@k/MRR metric artifacts** (multi-hop gold-set size, near-total
  synthetic/exam content overlap) mean every retrieval number in this report
  should be read as "retrieval quality *given this gold-set geometry*," not
  an absolute — each one here was cross-checked against a second explanation
  before being written up.
- **Both corpora are copyrighted and not redistributable** — Bolognia
  (Elsevier) and the IMA board exams. Private repo, gitignored, small
  redacted samples only.
- **Image-dependent questions (17% of the real exam corpus) are a hard
  architectural ceiling** for any text-only system — measured, not guessed
  (§4's ceiling calculation).

---

## 7. Setup & running it

Three independent Python environments (see `derma_guide_plan.md` Phase 0) —
`backend/` and `pipeline/` each manage their own `uv`-based venv; `eval/` +
`training/` run from a fourth ad-hoc venv (`requirements.txt` at the repo
root is the consolidated reference for it). GPU steps (QLoRA fine-tune, arms
A/B/C generation eval) run on Colab — see `notebooks/`.

### Product (backend + frontend)

```bash
# backend
cd backend
cp .env.example .env   # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, QDRANT_*
uv sync
uv run uvicorn main:app --reload   # http://localhost:8000

# frontend (separate shell)
cd frontend
cp .env.local.example .env.local
npm install
npm run dev   # http://localhost:3000
```

To demo the fine-tuned arm live instead of Claude, swap `backend/llm.py`'s
`_default = ClaudeGenerator()` for
`_default = get_generator("lora", adapter_path="<path to a saved r16 adapter>")`
(see `backend/generators.py`) and restart the backend.

### Data pipeline (rebuilding the corpus — requires the Bolognia PDF, not shipped)

```bash
cd pipeline
cp .env.example .env   # OPENAI_API_KEY (embeddings), QDRANT_* (ingestion)
uv sync
./run_pipeline.sh   # extract -> chunk -> embed -> ingest
```

### Evaluation / research path (requires the exam PDFs, not shipped)

```bash
# from repo root, using requirements.txt's eval/training section
uv run --with-requirements requirements.txt python3 eval/parse_exam.py          # Phase 1
uv run --with-requirements requirements.txt python3 eval/build_index.py         # Phase 3
uv run --with-requirements requirements.txt python3 eval/eval_retrieval.py --configs 400_80 --embedder bge
```

Notebooks (`notebooks/01_parse_exams.ipynb` .. `04_generation_eval.ipynb`)
wrap the same scripts for a fresh-Colab run — see each notebook's own first
cell for the exact file-upload list. These are written to be reproducible by
someone with their own copy of the (non-redistributable) source PDFs on
their own Drive, not by an anonymous third party with no access to either
corpus — that's what `outputs/` and this README's results section are for.

---

## 8. Repo layout

```
language_model_project/
├── derma_guide_plan.md     ← the full working design log (start here for detail)
├── README.md  requirements.txt
├── data/                   ← exams/ and the Bolognia PDF are gitignored (copyrighted)
├── pipeline/                ← extract → chunk → embed → ingest
├── backend/                 ← FastAPI RAG API + verifier agent
├── frontend/                ← Next.js chat UI
├── eval/                    ← exam parsing, retrieval eval, MCQ/generation eval
├── training/                ← QLoRA dataset prep + configs
├── notebooks/                ← Colab notebooks (01–04)
└── outputs/plots/           ← rendered result plots (this README's images)
```

---

## Disclaimer

This project is for educational purposes as part of an Applied Language
Models course assignment. It is **not a medical device, not clinically
validated, and not intended to inform real diagnostic or treatment
decisions.** Both underlying corpora (Bolognia's Dermatology, the Israeli
Medical Association board exams) are copyrighted and used here under
fair-use/educational terms for a private, non-redistributed course
submission.
