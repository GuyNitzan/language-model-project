"""
Phase 6 deliverable: render the report/slide plots from data already computed
in eval/data/*.json (Phase 3's retrieval_results.json, Phase 5's
mcq_results.json, and the Phase-5-closeout multihop/table-figure numbers
documented in derma_guide_plan.md). No new experiments run here -- this is
purely the rendering step for numbers already verified elsewhere.

Palette: the dataviz skill's validated categorical order (blue, orange, aqua,
yellow, ...) applied in fixed order per series identity, never re-cycled
across charts. Direct value labels on every bar satisfy the skill's "relief
rule" for the lower-contrast slots (aqua/yellow) on the light print surface.

Usage:
    uv run --with matplotlib python3 outputs/make_plots.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA = ROOT / "eval" / "data"
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- dataviz skill palette (light/print surface) ---------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11, "figure.dpi": 150,
})


def load(name):
    return json.loads((EVAL_DATA / name).read_text(encoding="utf-8"))


def bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", va="bottom", fontsize=9, color=INK)


# ---------------------------------------------------------------------------
# 1. Hit@k vs k, by chunk config (synthetic_test, bge) — with the artifact
#    caveat Phase 3 documents (chunk-density inflates this metric).
# ---------------------------------------------------------------------------
def plot_hit_at_k():
    rr = load("retrieval_results.json")
    configs = [("baseline_buggy", "current (~770w/0ov)", BLUE),
               ("400_80", "400w/80ov — carried forward", ORANGE),
               ("250_50", "250w/50ov", AQUA)]
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for cfg, label, color in configs:
        rec = rr[f"{cfg}__bge__synthetic_test"]
        ys = [rec["hit_at_k"][str(k)] for k in ks]
        ax.plot(ks, ys, marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("k")
    ax.set_ylabel("Hit@k")
    ax.set_title("Hit@k by chunk config (synthetic_test, bge-small)\n"
                  "Smaller chunks score higher here mechanically (more, denser candidates\n"
                  "per fixed target window) — not evidence they retrieve better.",
                  color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hit_at_k_by_chunk_config.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Cross-lingual retrieval — the headline query-language experiment.
# ---------------------------------------------------------------------------
def plot_cross_lingual():
    # Hit@10, not MRR@10 -- this is the metric the plan's own headline
    # cross-lingual finding is built on (derma_guide_plan.md Phase 3
    # Results). Checked both: on MRR@10 the openai series is NOT flat
    # (0.057->0.063->0.066), only Hit@10 shows the "bge gains, openai stays
    # flat" story cleanly, so Hit@10 is the one that matches the reported
    # +19% bge / ~flat openai conclusion and is used here to stay consistent
    # with the rest of the report rather than quietly picking a different
    # metric that tells a different story.
    rr = load("retrieval_results.json")
    variants = [("hebrew_raw", "raw Hebrew"), ("english_claude", "Claude-translated"),
                ("english_glossary", "+ glossary")]
    embedders = [("bge", "bge-small (English-only)", BLUE), ("openai", "text-embedding-3-large", ORANGE)]
    x = range(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for i, (emb, label, color) in enumerate(embedders):
        ys = [rr[f"400_80__{emb}__exam_dev__{v}"]["hit_at_k"]["10"] for v, _ in variants]
        offs = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offs, ys, width=width, color=color, label=label)
        bar_labels(ax, bars)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in variants])
    ax.set_ylabel("Hit@10 (exam_dev, 400_80)")
    ax.set_ylim(0, max(rr[f"400_80__{e}__exam_dev__{v}"]["hit_at_k"]["10"]
                        for e, _, _ in embedders for v, _ in variants) * 1.35)
    ax.set_title("Translating the query before retrieval — cross-lingual gap", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_lingual_hit10.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Reranker before/after.
# ---------------------------------------------------------------------------
def plot_reranker():
    rr = load("retrieval_results.json")
    rows = [("bge", "synthetic_val"), ("bge", "synthetic_test"), ("bge", "exam_dev__english_claude"),
            ("openai", "synthetic_val"), ("openai", "synthetic_test"), ("openai", "exam_dev__english_claude")]
    labels = ["bge\nsyn_val", "bge\nsyn_test", "bge\nexam_dev", "openai\nsyn_val", "openai\nsyn_test", "openai\nexam_dev"]
    before = [rr[f"400_80__{emb}__{s}"]["mrr_at_10"] for emb, s in rows]
    after = [rr[f"400_80__{emb}__rerank__{s}"]["mrr_at_10"] for emb, s in rows]
    x = range(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    b1 = ax.bar([xi - width / 2 for xi in x], before, width=width, color=BLUE, label="before rerank")
    b2 = ax.bar([xi + width / 2 for xi in x], after, width=width, color=ORANGE, label="after rerank")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MRR@10")
    ax.set_ylim(0, max(after + before) * 1.22)
    ax.set_title("Reranker (ms-marco-MiniLM-L-6-v2): consistent gain everywhere tried", color=INK, fontsize=12, pad=32)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reranker_before_after.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 4-arm MCQ accuracy by bucket, with ceiling + random-baseline reference lines.
# ---------------------------------------------------------------------------
def plot_mcq_accuracy():
    mcq = load("mcq_results.json")
    arms = ["A", "B", "C", "D"]
    arm_labels = {"A": "A. base, no RAG", "B": "B. base + RAG", "C": "C. LoRA + RAG", "D": "D. Claude + RAG"}
    buckets = [("overall", None), ("text_answerable", "text_answerable"), ("image_dependent", "image_dependent")]
    colors = {"overall": BLUE, "text_answerable": ORANGE, "image_dependent": AQUA}

    n_text = mcq["D"]["metrics"]["by_bucket"]["text_answerable"]["n"]
    n_img = mcq["D"]["metrics"]["by_bucket"]["image_dependent"]["n"]
    n_total = n_text + n_img
    ceiling = (n_text / n_total) * 1.0 + (n_img / n_total) * 0.25

    x = range(len(arms))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (name, key) in enumerate(buckets):
        ys = []
        for arm in arms:
            m = mcq[arm]["metrics"]
            ys.append(m["accuracy"] if key is None else m["by_bucket"][key]["accuracy"])
        offs = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offs, ys, width=width, color=colors[name], label=name.replace("_", " "))
        bar_labels(ax, bars, fmt="{:.2f}")
    ax.axhline(ceiling, color=INK, linewidth=1.2, linestyle="--")
    ax.text(len(arms) - 1 + 0.55, ceiling, f"ceiling {ceiling:.3f}", va="center", fontsize=9, color=INK)
    ax.axhline(0.25, color=INK_SECONDARY, linewidth=1, linestyle=":")
    ax.text(len(arms) - 1 + 0.55, 0.25, "random 0.25", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([arm_labels[a] for a in arms], fontsize=9)
    ax.set_ylabel("MCQ accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("4-arm MCQ accuracy, held-out 2025+2026 exams (n=258)", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mcq_accuracy_by_arm_bucket.png", bbox_inches="tight")
    plt.close(fig)
    return ceiling


# ---------------------------------------------------------------------------
# 5. Position bias — predicted vs. true letter distribution, arms A and D.
# ---------------------------------------------------------------------------
def plot_position_bias():
    mcq = load("mcq_results.json")
    letters = ["א", "ב", "ג", "ד"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True)
    for ax, arm, title in zip(axes, ["A", "D"], ["Arm A (base, no RAG)", "Arm D (Claude + RAG)"]):
        pb = mcq[arm]["metrics"]["position_bias"]
        predicted = [pb[l]["n_predicted"] for l in letters]
        true = [pb[l]["n_correct"] for l in letters]
        x = range(len(letters))
        width = 0.35
        b1 = ax.bar([xi - width / 2 for xi in x], predicted, width=width, color=BLUE, label="predicted")
        b2 = ax.bar([xi + width / 2 for xi in x], true, width=width, color=ORANGE, label="true")
        bar_labels(ax, b1, fmt="{:.0f}")
        bar_labels(ax, b2, fmt="{:.0f}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(letters, fontsize=12)
        ax.set_ylim(0, max(predicted + true) * 1.18)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylabel("count of 258 questions")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.suptitle("Position bias: a 3B base model over-predicts the last option; Claude tracks the true distribution",
                 color=INK, fontsize=11, y=1.1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "position_bias.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Table/figure vs. normal page-pointer references — a distinct, worse
#    retrieval failure class (Phase 5 Analysis finding #6). These exact
#    numbers are documented directly in derma_guide_plan.md; no separate
#    JSON was cached for this ad-hoc split, so they're inlined here rather
#    than fabricating a source file.
# ---------------------------------------------------------------------------
def plot_table_figure():
    rows = [("table/figure pointer\n(n=37)", 0.014, 0.014), ("normal page pointer\n(n=480)", 0.052, 0.064)]
    labels = [r[0] for r in rows]
    heb = [r[1] for r in rows]
    eng = [r[2] for r in rows]
    x = range(len(rows))
    width = 0.3
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    b1 = ax.bar([xi - width / 2 for xi in x], heb, width=width, color=BLUE, label="raw Hebrew")
    b2 = ax.bar([xi + width / 2 for xi in x], eng, width=width, color=ORANGE, label="Claude-translated")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("MRR@10 (exam_dev, 400_80/bge)")
    ax.set_title("Table/figure-pointer references retrieve 3.7-4.6x worse\n(Hit@1 = 0% for both, either language)",
                  color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table_figure_vs_normal.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_hit_at_k()
    plot_cross_lingual()
    plot_reranker()
    ceiling = plot_mcq_accuracy()
    plot_position_bias()
    plot_table_figure()
    print(f"ceiling = {ceiling:.4f}")
    print(f"6 plots written -> {OUT_DIR}")
