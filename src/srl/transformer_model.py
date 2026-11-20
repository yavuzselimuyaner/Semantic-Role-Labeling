"""Predicate-Conditioned Transformer + Constrained Linear-Chain CRF for SRL (v2)."""

from copy import deepcopy
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .decoding import ConstrainedLinearCRF, viterbi_decode_bio
from .metrics import compute_metrics

PRED_START_TOKEN = "[PRED]"
PRED_END_TOKEN = "[/PRED]"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(model_name: str):
    """Load fast tokenizer with add_prefix_space=True when needed (RoBERTa / BPE models)."""
    from transformers import AutoTokenizer

    lower_name = model_name.lower()
    needs_prefix_space = any(k in lower_name for k in ("roberta", "gpt", "bart", "bne"))
    kwargs = {"use_fast": True}
    if needs_prefix_space:
        kwargs["add_prefix_space"] = True
    try:
        tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": [PRED_START_TOKEN, PRED_END_TOKEN]})
    return tok


class SRLWordDataset(Dataset):
    """Tokenizes instances with predicate boundary markers and records word->subword alignment."""

    def __init__(
        self,
        instances: Sequence[dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.instances = list(instances)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        inst = self.instances[idx]
        words = inst["words"]
        pred_idx = inst["predicate_idx"]
        gold_labels = inst.get("labels", ["O"] * len(words))

        marked_words: List[str] = []
        marked_to_orig: List[Optional[int]] = []
        for i, w in enumerate(words):
            if i == pred_idx:
                marked_words.extend([PRED_START_TOKEN, w, PRED_END_TOKEN])
                marked_to_orig.extend([None, i, None])
            else:
                marked_words.append(w)
                marked_to_orig.append(i)

        enc = self.tokenizer(
            marked_words,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        word_ids = enc.word_ids()

        # Map each original word index -> first subword token index
        word_subword_idx: List[int] = []
        word_labels: List[int] = []
        kept_orig_indices: List[int] = []
        seen_orig = set()
        kept_pred_pos = 0

        for sub_i, wid in enumerate(word_ids):
            if wid is None:
                continue
            orig_i = marked_to_orig[wid]
            if orig_i is None or orig_i in seen_orig:
                continue
            seen_orig.add(orig_i)
            if orig_i == pred_idx:
                kept_pred_pos = len(word_subword_idx)
            word_subword_idx.append(sub_i)
            word_labels.append(self.label2id.get(gold_labels[orig_i], 0))
            kept_orig_indices.append(orig_i)

        # Safety fallback if sentence was completely empty
        if not word_subword_idx:
            word_subword_idx = [0]
            word_labels = [0]
            kept_orig_indices = [0] if words else []
            kept_pred_pos = 0

        rel_pos = [
            0 if i < kept_pred_pos else (1 if i == kept_pred_pos else 2)
            for i in range(len(word_subword_idx))
        ]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "word_subword_idx": word_subword_idx,
            "word_labels": word_labels,
            "kept_orig_indices": kept_orig_indices,
            "rel_pos": rel_pos,
            "pred_word_pos": kept_pred_pos,
            "orig_len": len(words),
        }


def make_collate_fn(pad_token_id: int):
    """Dynamic-padding collator for both subword sequences and word-level tensors."""

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        max_sub = max(len(x["input_ids"]) for x in batch)
        max_words = max(len(x["word_subword_idx"]) for x in batch)

        input_ids = torch.full((bsz, max_sub), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_sub), dtype=torch.long)
        word_subword_idx = torch.zeros((bsz, max_words), dtype=torch.long)
        word_labels = torch.zeros((bsz, max_words), dtype=torch.long)
        word_mask = torch.zeros((bsz, max_words), dtype=torch.bool)
        rel_pos = torch.zeros((bsz, max_words), dtype=torch.long)
        pred_word_pos = torch.zeros(bsz, dtype=torch.long)
        orig_lens = torch.zeros(bsz, dtype=torch.long)

        for b, item in enumerate(batch):
            s_len = len(item["input_ids"])
            w_len = len(item["word_subword_idx"])
            input_ids[b, :s_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[b, :s_len] = torch.tensor(item["attention_mask"], dtype=torch.long)
            word_subword_idx[b, :w_len] = torch.tensor(item["word_subword_idx"], dtype=torch.long)
            word_labels[b, :w_len] = torch.tensor(item["word_labels"], dtype=torch.long)
            word_mask[b, :w_len] = True
            rel_pos[b, :w_len] = torch.tensor(item["rel_pos"], dtype=torch.long)
            pred_word_pos[b] = item["pred_word_pos"]
            orig_lens[b] = item["orig_len"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_subword_idx": word_subword_idx,
            "word_labels": word_labels,
            "word_mask": word_mask,
            "rel_pos": rel_pos,
            "pred_word_pos": pred_word_pos,
            "orig_lens": orig_lens,
            "kept_orig_indices": [item.get("kept_orig_indices", list(range(len(item["word_subword_idx"])))) for item in batch],
        }

    return collate


class PredicateConditionedSRLNet(nn.Module):
    """Transformer encoder + predicate-interaction MLP + optional BIO-constrained Linear-Chain CRF."""

    def __init__(
        self,
        model_name: str,
        id2label: Dict[int, str],
        vocab_size: int,
        dropout: float = 0.2,
        use_crf: bool = True,
        enforce_bio: bool = True,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.id2label = id2label
        self.num_labels = len(id2label)
        self.use_crf = use_crf
        self.enforce_bio = enforce_bio

        if encoder is not None:
            self.encoder = encoder
            hidden_size = encoder.config.hidden_size
        else:
            from transformers import AutoConfig, AutoModel

            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=config)
            hidden_size = config.hidden_size
            self.encoder.resize_token_embeddings(vocab_size)

        rel_dim = 32
        self.rel_emb = nn.Embedding(3, rel_dim)  # 0=before, 1=predicate, 2=after
        in_dim = hidden_size * 3 + rel_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_labels),
        )

        if self.use_crf:
            self.crf = ConstrainedLinearCRF(id2label, enforce_bio=enforce_bio)
        else:
            self.crf = None

    def compute_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_subword_idx: torch.Tensor,
        rel_pos: torch.Tensor,
        pred_word_pos: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sub_h = out.last_hidden_state  # (B, L_sub, D)
        B, L_w = word_subword_idx.shape
        D = sub_h.size(-1)

        # Gather first-subword hidden states for each word
        gather_idx = word_subword_idx.unsqueeze(-1).expand(B, L_w, D)
        word_h = torch.gather(sub_h, dim=1, index=gather_idx)  # (B, L_w, D)

        # Gather predicate word hidden vector and broadcast across sentence
        pred_gather = pred_word_pos.view(B, 1, 1).expand(B, 1, D)
        pred_h = torch.gather(word_h, dim=1, index=pred_gather).expand(B, L_w, D)

        # Predicate-conditioned features: [h_i ; h_p ; h_i * h_p ; e_rel]
        rel_h = self.rel_emb(rel_pos)
        fused = torch.cat([word_h, pred_h, word_h * pred_h, rel_h], dim=-1)
        return self.head(fused)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_subword_idx: torch.Tensor,
        rel_pos: torch.Tensor,
        pred_word_pos: torch.Tensor,
        word_mask: torch.Tensor,
        word_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emissions = self.compute_emissions(
            input_ids, attention_mask, word_subword_idx, rel_pos, pred_word_pos
        )
        result = {"emissions": emissions}
        if word_labels is not None:
            if self.use_crf:
                loss = self.crf(emissions, word_labels, word_mask)
            else:
                flat_logits = emissions.view(-1, self.num_labels)
                flat_targets = word_labels.masked_fill(~word_mask, -100).view(-1)
                loss = nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
            result["loss"] = loss
        return result

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_subword_idx: torch.Tensor,
        rel_pos: torch.Tensor,
        pred_word_pos: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> List[List[int]]:
        emissions = self.compute_emissions(
            input_ids, attention_mask, word_subword_idx, rel_pos, pred_word_pos
        )
        if self.use_crf:
            return self.crf.decode(emissions, word_mask)
        if self.enforce_bio:
            return viterbi_decode_bio(emissions, word_mask, self.id2label)
        # Unconstrained argmax fallback (for ablation comparisons)
        pred_ids = emissions.argmax(dim=-1).cpu().tolist()
        lengths = word_mask.long().sum(dim=1).tolist()
        return [row[:l] for row, l in zip(pred_ids, lengths)]


# Recommended language-specific backbones
RECOMMENDED_MODELS = {
    "propbank": "microsoft/deberta-v3-base",
    "propbank_bert": "bert-base-cased",
    "propbank_es": "PlanTL-GOB-ES/roberta-base-bne",
    "propbank_es_beto": "dccuchile/bert-base-spanish-wwm-cased",
    "propbank_es_mdeberta": "microsoft/mdeberta-v3-base",
}


class TransformerSRLModel:
    """High-level trainer & predictor with AMP, differential LR, dev F1 checkpointing, and early stopping."""

    def __init__(
        self,
        label_set: Sequence[str],
        model_name: Optional[str] = None,
        language: str = "propbank",
        max_length: int = 256,
        batch_size: int = 32,
        epochs: int = 8,
        backbone_lr: float = 2e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 0.01,
        dropout: float = 0.2,
        use_crf: bool = True,
        enforce_bio: bool = True,
        patience: int = 3,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        set_seed(seed)
        self.label_set = list(label_set)
        self.label2id = {l: i for i, l in enumerate(self.label_set)}
        self.id2label = {i: l for i, l in enumerate(self.label_set)}
        self.model_name = model_name or RECOMMENDED_MODELS.get(language, "bert-base-cased")
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = _load_tokenizer(self.model_name)
        self.net = PredicateConditionedSRLNet(
            model_name=self.model_name,
            id2label=self.id2label,
            vocab_size=len(self.tokenizer),
            dropout=dropout,
            use_crf=use_crf,
            enforce_bio=enforce_bio,
        ).to(self.device)
        self.collate_fn = make_collate_fn(self.tokenizer.pad_token_id or 0)
        self.history: List[Dict[str, float]] = []
        self._fitted = False

    def _build_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        enc_decay, enc_no_decay, head_decay, head_no_decay = [], [], [], []
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            is_enc = name.startswith("encoder.")
            is_nd = any(nd in name for nd in no_decay)
            if is_enc and not is_nd:
                enc_decay.append(param)
            elif is_enc and is_nd:
                enc_no_decay.append(param)
            elif not is_enc and not is_nd:
                head_decay.append(param)
            else:
                head_no_decay.append(param)

        return torch.optim.AdamW(
            [
                {"params": enc_decay, "lr": self.backbone_lr, "weight_decay": self.weight_decay},
                {"params": enc_no_decay, "lr": self.backbone_lr, "weight_decay": 0.0},
                {"params": head_decay, "lr": self.head_lr, "weight_decay": self.weight_decay},
                {"params": head_no_decay, "lr": self.head_lr, "weight_decay": 0.0},
            ]
        )

    def fit(
        self,
        train_instances: Sequence[dict],
        dev_instances: Optional[Sequence[dict]] = None,
        save_path: Optional[str] = None,
        num_workers: int = 0,
    ) -> List[Dict[str, float]]:
        train_ds = SRLWordDataset(train_instances, self.tokenizer, self.label2id, self.max_length)
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        from transformers import get_cosine_schedule_with_warmup

        optimizer = self._build_optimizer()
        total_steps = len(train_dl) * self.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps,
        )

        is_deberta = "deberta" in self.model_name.lower()
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                use_amp = True
                use_scaler = False
            elif is_deberta:
                # DeBERTa-v3 disentangled attention can overflow float16 on T4 GPUs
                amp_dtype = torch.float32
                use_amp = False
                use_scaler = False
            else:
                amp_dtype = torch.float16
                use_amp = True
                use_scaler = True
        else:
            amp_dtype = torch.float32
            use_amp = False
            use_scaler = False

        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        best_dev_f1 = -1.0
        best_state = None
        no_improve = 0
        self.history = []
        dev_gold = [inst["labels"] for inst in dev_instances] if dev_instances else None

        print(
            f"Training {self.model_name} (CRF={self.net.use_crf}, BIO={self.net.enforce_bio}) "
            f"on {len(train_instances):,} instances | device={self.device} | AMP={use_amp} ({amp_dtype})"
        )

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total_loss = 0.0
            bar = tqdm(train_dl, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in bar:
                optimizer.zero_grad(set_to_none=True)
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k not in ("orig_lens", "kept_orig_indices")
                }
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    loss = self.net(**inputs)["loss"]

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(1, len(train_dl))
            record: Dict[str, float] = {"epoch": epoch, "train_loss": round(avg_loss, 4)}

            if dev_instances:
                self._fitted = True
                dev_preds = self.predict(dev_instances, num_workers=num_workers, show_progress=False)
                dev_res = compute_metrics(dev_gold, dev_preds)
                record.update(
                    {
                        "dev_p": dev_res["precision"],
                        "dev_r": dev_res["recall"],
                        "dev_f1": dev_res["f1"],
                        "dev_illegal_bio": dev_res["illegal_bio"],
                    }
                )
                print(
                    f"  Epoch {epoch}: loss={avg_loss:.4f} | "
                    f"Dev P={dev_res['precision']:.4f} R={dev_res['recall']:.4f} "
                    f"F1={dev_res['f1']:.4f} (illegal_bio={dev_res['illegal_bio']})"
                )
                if dev_res["f1"] > best_dev_f1:
                    best_dev_f1 = dev_res["f1"]
                    best_state = deepcopy(self.net.state_dict())
                    no_improve = 0
                    if save_path:
                        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                        torch.save(
                            {
                                "state_dict": best_state,
                                "model_name": self.model_name,
                                "label_set": self.label_set,
                                "use_crf": self.net.use_crf,
                                "enforce_bio": self.net.enforce_bio,
                                "best_dev_f1": best_dev_f1,
                            },
                            save_path,
                        )
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        print(f"  Early stopping triggered at epoch {epoch} (best Dev F1={best_dev_f1:.4f}).")
                        break
            else:
                print(f"  Epoch {epoch}: avg loss={avg_loss:.4f}")

            self.history.append(record)

        if best_state is not None:
            self.net.load_state_dict(best_state)
            print(f"  Restored best checkpoint (Dev F1={best_dev_f1:.4f}).")

        self._fitted = True
        return self.history

    @torch.no_grad()
    def predict(
        self,
        instances: Sequence[dict],
        num_workers: int = 2,
        show_progress: bool = True,
    ) -> List[List[str]]:
        assert self._fitted, "Call fit() before predict()."
        ds = SRLWordDataset(instances, self.tokenizer, self.label2id, self.max_length)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers if self.device.type == "cuda" else 0,
            pin_memory=(self.device.type == "cuda"),
        )
        self.net.eval()
        is_deberta = "deberta" in self.model_name.lower()
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype, use_amp = torch.bfloat16, True
        elif self.device.type == "cuda" and not is_deberta:
            amp_dtype, use_amp = torch.float16, True
        else:
            amp_dtype, use_amp = torch.float32, False
        all_preds: List[List[str]] = []

        iterator = tqdm(dl, desc="Predicting") if show_progress else dl
        for batch in iterator:
            orig_lens = batch["orig_lens"].tolist()
            kept_indices_batch = batch["kept_orig_indices"]
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k not in ("orig_lens", "word_labels", "kept_orig_indices")
            }
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                tag_id_seqs = self.net.decode(**inputs)
            for tag_ids, orig_n, kept_idxs in zip(tag_id_seqs, orig_lens, kept_indices_batch):
                labels = ["O"] * orig_n
                for t_id, orig_i in zip(tag_ids, kept_idxs):
                    if 0 <= orig_i < orig_n:
                        labels[orig_i] = self.id2label[t_id]
                all_preds.append(labels)
        return all_preds
