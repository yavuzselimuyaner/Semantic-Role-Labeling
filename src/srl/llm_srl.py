"""LLM-based Semantic Role Labeling: Zero-Shot, Few-Shot (Pydantic JSON), and QLoRA Fine-Tuning.

Benchmarks open-source instruction LLMs (e.g., Qwen2.5-3B-Instruct in 4-bit NF4 on Colab T4)
against specialized Span-Biaffine Encoders on identical CoNLL-2005 span-level metrics,
also tracking JSON schema validity rate and inference latency (ms/instance).
"""

import json
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError

from .metrics import bio_to_spans, compute_metrics
from .schema import spans_to_bio

SYSTEM_PROMPT = (
    "You are an expert PropBank Semantic Role Labeling (SRL) annotator. "
    "Given a tokenized sentence (with 0-based token indices) and a target predicate index, "
    "identify all semantic arguments of that predicate and return ONLY valid JSON matching:\n"
    '{"arguments": [{"role": "ARG0", "start": 0, "end": 2, "text": "..."}]}\n'
    "Core roles: ARG0 (agent/experiencer), ARG1 (patient/theme), ARG2 (recipient/instrument/attribute), "
    "ARG3, ARG4. Adjunct roles: ARGM-TMP (time), ARGM-LOC (location), ARGM-MNR (manner), "
    "ARGM-ADV (general adverbial), ARGM-MOD (modal), ARGM-NEG (negation), ARGM-DIS (discourse), "
    "ARGM-CAU (cause), ARGM-PRP (purpose), ARGM-DIR (direction), ARGM-EXT (extent), ARGM-PRD (secondary predication).\n"
    "Rules: `start` is inclusive, `end` is exclusive (Python slice). Do NOT include the predicate itself (V). "
    "Output ONLY raw JSON with no markdown or explanation."
)


class ArgumentSpan(BaseModel):
    role: str = Field(..., description="PropBank role tag such as ARG0, ARG1, ARGM-LOC")
    start: int = Field(..., ge=0, description="Inclusive 0-based start word index")
    end: int = Field(..., gt=0, description="Exclusive 0-based end word index")
    text: Optional[str] = Field(default=None, description="Surface text of the argument span")


class SRLPrediction(BaseModel):
    arguments: List[ArgumentSpan] = Field(default_factory=list)


def format_indexed_sentence(words: Sequence[str], predicate_idx: int) -> str:
    indexed = " ".join(f"{i}:{w}" for i, w in enumerate(words))
    pred_w = words[predicate_idx] if 0 <= predicate_idx < len(words) else ""
    return (
        f"Tokens: {indexed}\n"
        f"Target predicate: index={predicate_idx}, word=\"{pred_w}\"\n"
        f"JSON:"
    )


def instance_to_target_json(inst: dict) -> str:
    """Convert a unified PropBank instance into canonical target JSON for few-shot / QLoRA SFT."""
    words = inst["words"]
    spans = [
        (role, s, e)
        for (role, s, e) in bio_to_spans(inst["labels"])
        if role not in ("O", "V")
    ]
    payload = {
        "arguments": [
            {
                "role": role,
                "start": s,
                "end": e,
                "text": " ".join(words[s:e]),
            }
            for role, s, e in spans
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def build_chat_messages(
    inst: dict,
    few_shot_examples: Optional[Sequence[dict]] = None,
) -> List[Dict[str, str]]:
    """Build ChatML messages list for zero-shot, few-shot, or QLoRA inference."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append(
                {"role": "user", "content": format_indexed_sentence(ex["words"], ex["predicate_idx"])}
            )
            messages.append({"role": "assistant", "content": instance_to_target_json(ex)})
    messages.append(
        {"role": "user", "content": format_indexed_sentence(inst["words"], inst["predicate_idx"])}
    )
    return messages


def _extract_json_block(raw_text: str) -> Optional[str]:
    """Extract the first JSON object from model output (tolerates stray fences or chatter)."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _align_text_span(words: Sequence[str], text: str, hint_start: int) -> Optional[Tuple[int, int]]:
    """If LLM start/end indices drifted by 1-2 tokens, locate exact contiguous token slice for `text`."""
    if not text:
        return None
    target_toks = [t.lower() for t in text.strip().split() if t.strip()]
    if not target_toks:
        return None
    low_words = [w.lower() for w in words]
    k = len(target_toks)
    matches = []
    for s in range(len(low_words) - k + 1):
        if low_words[s : s + k] == target_toks:
            matches.append((abs(s - hint_start), s, s + k))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[0][1], matches[0][2]


def parse_llm_response(
    raw_text: str,
    words: Sequence[str],
    predicate_idx: int,
    align_text_fallback: bool = True,
) -> Tuple[List[str], bool]:
    """Parse & Pydantic-validate LLM JSON output into a BIO sequence.

    Returns (bio_labels, is_schema_valid).
    """
    n = len(words)
    default_bio = spans_to_bio(n, predicate_idx, [])
    block = _extract_json_block(raw_text)
    if block is None:
        return default_bio, False

    try:
        parsed = SRLPrediction.model_validate_json(block)
    except (ValidationError, ValueError):
        return default_bio, False

    spans: List[Tuple[str, int, int]] = []
    for arg in parsed.arguments:
        role = arg.role.strip().upper()
        if role in ("V", "O", "NULL", ""):
            continue
        s, e = arg.start, arg.end
        valid_idx = 0 <= s < e <= n
        text_matches = (
            valid_idx
            and arg.text is not None
            and " ".join(words[s:e]).lower() == arg.text.strip().lower()
        )
        if align_text_fallback and arg.text and not text_matches:
            aligned = _align_text_span(words, arg.text, hint_start=s)
            if aligned is not None:
                s, e = aligned
                valid_idx = True
        if valid_idx and not (s == predicate_idx and e == predicate_idx + 1):
            spans.append((role, s, e))

    return spans_to_bio(n, predicate_idx, spans), True


class LocalHFLlmSRL:
    """4-bit NF4 HuggingFace LLM wrapper (e.g., Qwen/Qwen2.5-3B-Instruct) for Zero/Few-Shot & QLoRA."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if load_in_4bit and torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

    def evaluate(
        self,
        instances: Sequence[dict],
        few_shot_examples: Optional[Sequence[dict]] = None,
        batch_size: int = 8,
    ) -> Dict[str, object]:
        import torch
        from tqdm.auto import tqdm

        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        self.tokenizer.padding_side = "left"
        gold = [inst["labels"] for inst in instances]
        raw_preds_bio: List[List[str]] = []
        aligned_preds_bio: List[List[str]] = []
        valid_count = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        for st in tqdm(range(0, len(instances), batch_size), desc=f"LLM eval ({self.model_name.split('/')[-1]})"):
            chunk = instances[st : st + batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    build_chat_messages(inst, few_shot_examples=few_shot_examples),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for inst in chunk
            ]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
                self.model.device
            )
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for i, inst in enumerate(chunk):
                out_text = self.tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=True)
                bio_raw, ok = parse_llm_response(out_text, inst["words"], inst["predicate_idx"], align_text_fallback=False)
                bio_aln, _ = parse_llm_response(out_text, inst["words"], inst["predicate_idx"], align_text_fallback=True)
                raw_preds_bio.append(bio_raw)
                aligned_preds_bio.append(bio_aln)
                if ok:
                    valid_count += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000.0

        m_raw = compute_metrics(gold, raw_preds_bio)
        m_aln = compute_metrics(gold, aligned_preds_bio)
        return {
            "model_name": self.model_name,
            "num_eval": len(instances),
            "few_shot_k": len(few_shot_examples) if few_shot_examples else 0,
            "json_valid_rate": round(valid_count / max(1, len(instances)), 4),
            "avg_latency_ms": round(elapsed_ms / max(1, len(instances)), 2),
            "strict_index": {"precision": m_raw["precision"], "recall": m_raw["recall"], "f1": m_raw["f1"]},
            "text_aligned": {"precision": m_aln["precision"], "recall": m_aln["recall"], "f1": m_aln["f1"]},
            "precision": m_aln["precision"],
            "recall": m_aln["recall"],
            "f1": m_aln["f1"],
            "per_role": m_aln["per_role"],
        }

    def finetune_qlora(
        self,
        train_instances: Sequence[dict],
        epochs: int = 2,
        batch_size: int = 4,
        grad_accum: int = 4,
        lr: float = 2e-4,
        lora_r: int = 16,
        max_length: int = 512,
    ) -> float:
        """QLoRA supervised fine-tuning on prompt->JSON pairs (assistant-token loss only)."""
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        self.model = prepare_model_for_kbit_training(self.model)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

        # Pre-tokenize assistant-only loss examples
        self.tokenizer.padding_side = "right"
        encoded_items = []
        for inst in train_instances:
            prompt_msgs = build_chat_messages(inst)
            full_msgs = prompt_msgs + [{"role": "assistant", "content": instance_to_target_json(inst)}]
            prompt_txt = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_txt = self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
            p_ids = self.tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
            f_ids = self.tokenizer(full_txt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[min(len(p_ids), len(f_ids)) :]
            if all(l == -100 for l in labels):
                continue
            encoded_items.append((f_ids, labels))

        pad_id = self.tokenizer.pad_token_id or 0

        def collate(batch):
            ml = max(len(x[0]) for x in batch)
            inp = torch.full((len(batch), ml), pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), ml), dtype=torch.long)
            lbl = torch.full((len(batch), ml), -100, dtype=torch.long)
            for i, (ids, ls) in enumerate(batch):
                inp[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                att[i, : len(ids)] = 1
                lbl[i, : len(ls)] = torch.tensor(ls, dtype=torch.long)
            return {"input_ids": inp, "attention_mask": att, "labels": lbl}

        dl = DataLoader(encoded_items, batch_size=batch_size, shuffle=True, collate_fn=collate)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        t0 = time.time()
        for ep in range(1, epochs + 1):
            tot_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            bar = tqdm(dl, desc=f"QLoRA Epoch {ep}/{epochs}")
            for step, batch in enumerate(bar, start=1):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                loss = self.model(**batch).loss / grad_accum
                loss.backward()
                tot_loss += loss.item() * grad_accum
                if step % grad_accum == 0 or step == len(dl):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                bar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")
            print(f"  QLoRA Epoch {ep} avg loss: {tot_loss / max(1, len(dl)):.4f}")

        return round(time.time() - t0, 1)
