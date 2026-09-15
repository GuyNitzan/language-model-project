"""
Phase 5, decision #4 (derma_guide_plan.md): pluggable answer-generation
backends behind one shared interface, so the 4-arm comparison (A/B/C/D) and
the live app can swap generators with no other code change.

ClaudeGenerator reproduces backend/llm.py's pre-Phase-5 behaviour exactly —
this module is the new canonical home for the prompt format (SYSTEM_PROMPT,
format_context) that llm.py used to own; llm.py is now a thin wrapper
around ClaudeGenerator for backward compatibility with main.py.

LoRAGenerator wraps the local Qwen2.5-3B-Instruct base model, optionally
with one of the Phase 4 QLoRA adapters (r8/r16/r32, trained in
notebooks/03_finetune.ipynb) merged in via PEFT. torch/transformers/peft
are imported lazily inside __init__ so importing this module — or using
only ClaudeGenerator — never requires them. In practice LoRAGenerator needs
a GPU: this dev sandbox is CPU-only with 3.7GB RAM (see derma_guide_plan.md
Phase 4), which can't even hold Qwen2.5-3B in fp32. It runs on the same
Colab environment Phase 4 trained on (notebooks/04_generation_eval.ipynb).

The prompt format here (SYSTEM_PROMPT, the "[Excerpt N | chapter | p.N]"
excerpt layout, the "Book excerpts:\\n\\n...\\n\\n---\\n\\nQuestion: ..."
user-message shape) is deliberately identical to
training/prepare_dataset.py's copy — that module predates this refactor and
was already commented as "mirrors backend/llm.py exactly" for the same
reason: the fine-tuned model must see the same input distribution at eval
time that it was trained on.
"""

from __future__ import annotations

from typing import Protocol

import config

SYSTEM_PROMPT = """\
You are a dermatology assistant trained on Bolognia's Dermatology (5th edition, 2024).

Rules:
1. Answer ONLY from the provided book excerpts. Never use outside knowledge.
2. If the excerpts do not contain enough information to answer, say so explicitly.
3. Always cite your sources inline using [Chapter: <name>, p.<page>] notation.
4. Be precise and clinically accurate. Use proper dermatology terminology.
5. Structure longer answers with clear headings when appropriate.
"""

# Matches training/prepare_dataset.py's ABSTENTION_ANSWER exactly — this is
# the literal string the LoRA arm was trained to produce when its shown
# context doesn't support a grounded answer. Exposed here so eval scripts
# (e.g. eval_generation.py's abstention-rate metric) can match against it
# without duplicating the literal.
ABSTENTION_ANSWER = "The provided excerpts do not contain enough information to answer this question."


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        location = f"{c['chapter']}"
        if c.get("section"):
            location += f" › {c['section']}"
        if c.get("subsection"):
            location += f" › {c['subsection']}"
        page_ref = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] \
                   else f"pp.{c['page_start']}–{c['page_end']}"
        parts.append(
            f"[Excerpt {i} | {location} | {page_ref}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_user_message(question: str, chunks: list[dict]) -> str:
    if not chunks:
        # Arm A (closed-book) shape — no excerpts block at all, rather than
        # an empty one, so the no-RAG arm isn't handed a formatting tell.
        return f"Question: {question}"
    context = format_context(chunks)
    return f"Book excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"


def _format_history(history: list[dict]) -> list[dict]:
    """Convert stored history to {role, content} messages, most-recent-first
    truncated to config.MAX_HISTORY pairs — identical to the old llm.py."""
    messages = []
    for turn in history[-(config.MAX_HISTORY * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


class Generator(Protocol):
    """Every arm implements this. `chunks` may be empty (closed-book)."""

    def answer(self, question: str, chunks: list[dict], history: list[dict]) -> str: ...


class ClaudeGenerator:
    """Default generator — unchanged behaviour from the pre-Phase-5 llm.py."""

    def __init__(self, model: str | None = None):
        import anthropic
        self.model = model or config.CLAUDE_MODEL
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def answer(self, question: str, chunks: list[dict], history: list[dict]) -> str:
        user_message = build_user_message(question, chunks)
        messages = _format_history(history) + [{"role": "user", "content": user_message}]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            # Without this, some claude-sonnet-5 calls spend their entire
            # max_tokens budget on an invisible extended-thinking block
            # before emitting any text — confirmed directly during Phase 5's
            # eval_generation.py run: stop_reason "max_tokens", content[0] a
            # ThinkingBlock, zero text blocks, `content[0].text` crashing
            # with AttributeError. Not needed for citation-grounded QA.
            thinking={"type": "disabled"},
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        # Defense in depth even with thinking disabled: never assume
        # content[0] is the text block.
        return next((b.text for b in response.content if b.type == "text"), "")


class LoRAGenerator:
    """Local Qwen2.5-3B-Instruct, optionally with a QLoRA adapter (arm C).
    With adapter_path=None this IS arm B (base model + RAG); pass chunks=[]
    at call time to get arm A (base model, closed-book) from the same
    instance — no need for a fourth class.
    """

    BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(
        self,
        adapter_path: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
        load_in_4bit: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)

        kwargs = {}
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # Turing T4 has no bf16 — see Phase 4 notes
            )
        else:
            kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.BASE_MODEL, **kwargs)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        if not load_in_4bit:
            model = model.to(device)
        self.model = model.eval()
        self._torch = torch

    def answer(self, question: str, chunks: list[dict], history: list[dict]) -> str:
        user_message = build_user_message(question, chunks)
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + _format_history(history)
            + [{"role": "user", "content": user_message}]
        )
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def get_generator(name: str = "claude", **kwargs) -> Generator:
    """name: 'claude' | 'lora'. kwargs forwarded to the chosen class
    (e.g. adapter_path=... for the fine-tuned arm)."""
    if name == "claude":
        return ClaudeGenerator(**kwargs)
    if name == "lora":
        return LoRAGenerator(**kwargs)
    raise ValueError(f"unknown generator {name!r}")
