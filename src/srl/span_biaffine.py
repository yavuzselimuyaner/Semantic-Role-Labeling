"""Multi-Predicate Span-Biaffine SRL with Global Role-Uniqueness Decoding.

Encodes each unique sentence ONCE (3x faster than predicate-duplicated BIO models),
enumerates & prunes candidate spans, scores (predicate, span, role) triples via a
Biaffine scorer, and decodes globally valid non-overlapping spans with unique core roles.
"""

from copy import deepcopy
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .metrics import compute_metrics
from .schema import CORE_ROLES, group_by_sentence, spans_to_bio


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    lower_name = model_name.lower()
    needs_prefix_space = any(k in lower_name for k in ("roberta", "gpt", "bart", "bne")) and "xlm" not in lower_name
    kwargs = {"use_fast": True}
    if needs_prefix_space:
        kwargs["add_prefix_space"] = True
    try:
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)


class GroupedSentenceDataset(Dataset):
    """Sentence-level multi-predicate dataset (encodes each unique sentence once)."""

    def __init__(
        self,
        grouped_sentences: Sequence[dict],
        tokenizer,
        role2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.records = list(grouped_sentences)
        self.tokenizer = tokenizer
        self.role2id = role2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        words = rec["words"]
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        word_ids = enc.word_ids()

        # Map original word index -> first subword index
        orig_to_kept: Dict[int, int] = {}
        word_subword_idx: List[int] = []
        kept_orig_indices: List[int] = []

        for sub_i, wid in enumerate(word_ids):
            if wid is None or wid in orig_to_kept:
                continue
            orig_to_kept[wid] = len(word_subword_idx)
            word_subword_idx.append(sub_i)
            kept_orig_indices.append(wid)

        if not word_subword_idx:
            word_subword_idx = [0]
            kept_orig_indices = [0]
            orig_to_kept[0] = 0

        # Remap predicates & gold spans into kept word coordinates
        pred_positions: List[int] = []
        gold_maps: List[Dict[Tuple[int, int], int]] = []
        for p_orig, spans in zip(rec["predicates"], rec["gold_spans"]):
            p_kept = orig_to_kept.get(p_orig, 0)
            pred_positions.append(p_kept)
            smap: Dict[Tuple[int, int], int] = {}
            for role, s_orig, e_orig in spans:
                r_id = self.role2id.get(role)
                if r_id is None:
                    continue
                s_k = orig_to_kept.get(s_orig)
                e_k = orig_to_kept.get(e_orig - 1)
                if s_k is not None and e_k is not None and s_k <= e_k:
                    smap[(s_k, e_k)] = r_id
            gold_maps.append(smap)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "word_subword_idx": word_subword_idx,
            "kept_orig_indices": kept_orig_indices,
            "pred_positions": pred_positions,
            "gold_maps": gold_maps,
            "orig_indices": rec["orig_indices"],
            "orig_len": len(words),
        }


def make_grouped_collate_fn(pad_token_id: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        bsz = len(batch)
        max_sub = max(len(x["input_ids"]) for x in batch)
        max_words = max(len(x["word_subword_idx"]) for x in batch)

        input_ids = torch.full((bsz, max_sub), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_sub), dtype=torch.long)
        word_subword_idx = torch.zeros((bsz, max_words), dtype=torch.long)
        word_lengths = torch.zeros(bsz, dtype=torch.long)

        for b, item in enumerate(batch):
            s_len = len(item["input_ids"])
            w_len = len(item["word_subword_idx"])
            input_ids[b, :s_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[b, :s_len] = torch.tensor(item["attention_mask"], dtype=torch.long)
            word_subword_idx[b, :w_len] = torch.tensor(item["word_subword_idx"], dtype=torch.long)
            word_lengths[b] = w_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_subword_idx": word_subword_idx,
            "word_lengths": word_lengths,
            "pred_positions": [x["pred_positions"] for x in batch],
            "gold_maps": [x["gold_maps"] for x in batch],
            "kept_orig_indices": [x["kept_orig_indices"] for x in batch],
            "orig_indices": [x["orig_indices"] for x in batch],
            "orig_lens": [x["orig_len"] for x in batch],
        }

    return collate


def enumerate_spans(seq_len: int, max_width: int) -> Tuple[List[int], List[int]]:
    """Enumerate all (start, end_inclusive) pairs with 0 <= start <= end < seq_len and width <= max_width."""
    starts: List[int] = []
    ends: List[int] = []
    for s in range(seq_len):
        max_e = min(seq_len, s + max_width)
        for e in range(s, max_e):
            starts.append(s)
            ends.append(e)
    return starts, ends


def decode_predicate_spans(
    logits_kr: torch.Tensor,
    cand_starts: Sequence[int],
    cand_ends: Sequence[int],
    pred_pos: int,
    id2role: Dict[int, str],
    enforce_unique_core: bool = True,
) -> List[Tuple[str, int, int]]:
    """Globally constrained span decoding for one predicate:
    1. Only keeps spans whose best non-NULL role beats NULL (margin > 0).
    2. Prevents overlapping spans.
    3. Enforces unique core roles (ARG0..ARG5) per predicate.
    Returns half-open spans (role, start, end_exclusive) in kept-word indices.
    """
    if logits_kr.numel() == 0:
        return []

    null_scores = logits_kr[:, 0]  # (K,)
    role_scores, role_ids = logits_kr[:, 1:].max(dim=-1)  # (K,)
    role_ids = role_ids + 1  # shift back to 1..R-1
    margins = (role_scores - null_scores).cpu().tolist()
    r_ids = role_ids.cpu().tolist()

    candidates = []
    for k, (margin, r_id) in enumerate(zip(margins, r_ids)):
        if margin <= 0.0:
            continue
        s = cand_starts[k]
        e = cand_ends[k]
        # Skip single-token span on the predicate itself
        if s == e == pred_pos:
            continue
        candidates.append((margin, id2role[r_id], s, e + 1))

    # Highest-confidence spans first
    candidates.sort(key=lambda x: -x[0])

    accepted: List[Tuple[str, int, int]] = []
    used_core = set()
    for _, role, s, e_excl in candidates:
        if enforce_unique_core and role in CORE_ROLES and role in used_core:
            continue
        # Non-overlapping check
        if any(max(s, as_) < min(e_excl, ae_) for _, as_, ae_ in accepted):
            continue
        accepted.append((role, s, e_excl))
        if role in CORE_ROLES:
            used_core.add(role)

    accepted.sort(key=lambda x: x[1])
    return accepted


class SpanBiaffineSRLNet(nn.Module):
    """Multi-predicate Span-Biaffine SRL network with unary span pruner & Biaffine role scorer."""

    def __init__(
        self,
        model_name: str,
        id2role: Dict[int, str],
        max_width: int = 30,
        biaffine_dim: int = 192,
        width_emb_dim: int = 32,
        dropout: float = 0.2,
        prune_ratio: float = 2.0,
        min_keep_spans: int = 32,
        span_bce_weight: float = 0.2,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.id2role = id2role
        self.num_roles = len(id2role)  # includes NULL at index 0
        self.max_width = max_width
        self.prune_ratio = prune_ratio
        self.min_keep_spans = min_keep_spans
        self.span_bce_weight = span_bce_weight

        if encoder is not None:
            self.encoder = encoder
            hidden = encoder.config.hidden_size
        else:
            from transformers import AutoConfig, AutoModel

            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=config)
            hidden = config.hidden_size

        self.width_emb = nn.Embedding(max_width, width_emb_dim)
        span_in_dim = hidden * 3 + width_emb_dim  # [h_s ; h_e ; h_e - h_s ; width_emb]

        # Fast unary span scorer for pruning
        self.span_scorer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(span_in_dim, biaffine_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(biaffine_dim, 1),
        )

        # Predicate & span projections into Biaffine space
        self.pred_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, biaffine_dim),
            nn.LayerNorm(biaffine_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.span_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(span_in_dim, biaffine_dim),
            nn.LayerNorm(biaffine_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Biaffine tensor U: (d, R, d) + linear terms + bias
        self.biaffine_U = nn.Parameter(torch.empty(biaffine_dim, self.num_roles, biaffine_dim))
        nn.init.xavier_uniform_(self.biaffine_U)
        self.pred_linear = nn.Linear(biaffine_dim, self.num_roles, bias=False)
        self.span_linear = nn.Linear(biaffine_dim, self.num_roles, bias=False)
        self.role_bias = nn.Parameter(torch.zeros(self.num_roles))

    def _encode_words(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_subword_idx: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sub_h = out.last_hidden_state  # (B, L_sub, D)
        B, L_w = word_subword_idx.shape
        D = sub_h.size(-1)
        gather_idx = word_subword_idx.unsqueeze(-1).expand(B, L_w, D)
        return torch.gather(sub_h, dim=1, index=gather_idx)

    def _score_sentence(
        self,
        word_h: torch.Tensor,
        pred_positions: Sequence[int],
        gold_maps: Optional[Sequence[Dict[Tuple[int, int], int]]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[torch.Tensor, List[int], List[int]]]]:
        """Score all predicates and pruned spans for one sentence.

        Returns:
            (loss or None, list of (logits_KR, cand_starts, cand_ends) per predicate)
        """
        device = word_h.device
        L = word_h.size(0)
        starts, ends = enumerate_spans(L, self.max_width)
        S = len(starts)
        if S == 0 or not pred_positions:
            zero = word_h.sum() * 0.0
            return (zero if gold_maps is not None else None), []

        s_idx = torch.tensor(starts, dtype=torch.long, device=device)
        e_idx = torch.tensor(ends, dtype=torch.long, device=device)
        widths = e_idx - s_idx

        h_s = word_h[s_idx]
        h_e = word_h[e_idx]
        w_e = self.width_emb(widths)
        span_repr = torch.cat([h_s, h_e, h_e - h_s, w_e], dim=-1)  # (S, span_in_dim)
        span_scores = self.span_scorer(span_repr).squeeze(-1)  # (S,)

        # Auxiliary BCE loss on unary span scores during training
        bce_loss = None
        gold_span_indices = set()
        if gold_maps is not None:
            span_to_idx = {(s, e): idx for idx, (s, e) in enumerate(zip(starts, ends))}
            span_targets = torch.zeros(S, dtype=torch.float32, device=device)
            for smap in gold_maps:
                for pair in smap:
                    idx = span_to_idx.get(pair)
                    if idx is not None:
                        span_targets[idx] = 1.0
                        gold_span_indices.add(idx)
            # Upweight positive spans since most candidate spans are negative
            pos_cnt = span_targets.sum().clamp_min(1.0)
            neg_cnt = (S - pos_cnt).clamp_min(1.0)
            pos_weight = (neg_cnt / pos_cnt).clamp(max=15.0)
            bce_loss = F.binary_cross_entropy_with_logits(
                span_scores.float(), span_targets, pos_weight=pos_weight
            )

        # Prune to top-K candidate spans (during training, always include gold spans)
        k_keep = min(S, max(self.min_keep_spans, int(math.ceil(self.prune_ratio * L))))
        _, top_k_idx = torch.topk(span_scores, k=k_keep, sorted=False)

        if self.training and gold_span_indices:
            keep_set = set(top_k_idx.tolist()) | gold_span_indices
            keep_idx = torch.tensor(sorted(keep_set), dtype=torch.long, device=device)
        else:
            keep_idx = top_k_idx

        kept_span_repr = span_repr[keep_idx]  # (K, span_in_dim)
        kept_span_scores = span_scores[keep_idx]  # (K,)
        kept_starts = [starts[i] for i in keep_idx.tolist()]
        kept_ends = [ends[i] for i in keep_idx.tolist()]

        # Project predicates & kept spans -> Biaffine role logits (P, K, R)
        p_idx = torch.tensor(pred_positions, dtype=torch.long, device=device)
        u_p = self.pred_mlp(word_h[p_idx])  # (P, d)
        v_k = self.span_mlp(kept_span_repr)  # (K, d)

        # Vectorized Biaffine: (P, d) x (d, R, d) x (K, d) -> (P, K, R)
        biaff = torch.einsum("pd,drq,kq->pkr", u_p, self.biaffine_U, v_k)
        logits = (
            biaff
            + self.pred_linear(u_p).unsqueeze(1)
            + self.span_linear(v_k).unsqueeze(0)
            + self.role_bias.view(1, 1, -1)
        )
        # Add unary span score prior to non-NULL roles
        logits[:, :, 1:] = logits[:, :, 1:] + kept_span_scores.view(1, -1, 1)

        loss = None
        if gold_maps is not None:
            P, K, R = logits.shape
            kept_pair_to_k = {(s, e): k for k, (s, e) in enumerate(zip(kept_starts, kept_ends))}
            targets = torch.zeros((P, K), dtype=torch.long, device=device)
            for p_i, smap in enumerate(gold_maps):
                for pair, r_id in smap.items():
                    k_pos = kept_pair_to_k.get(pair)
                    if k_pos is not None:
                        targets[p_i, k_pos] = r_id
            role_loss = F.cross_entropy(logits.float().reshape(-1, R), targets.reshape(-1))
            loss = role_loss + self.span_bce_weight * bce_loss

        per_pred_out = [(logits[p_i], kept_starts, kept_ends) for p_i in range(len(pred_positions))]
        return loss, per_pred_out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_subword_idx: torch.Tensor,
        word_lengths: torch.Tensor,
        pred_positions: Sequence[Sequence[int]],
        gold_maps: Optional[Sequence[Sequence[Dict[Tuple[int, int], int]]]] = None,
    ) -> Dict[str, object]:
        word_h_batch = self._encode_words(input_ids, attention_mask, word_subword_idx)
        losses = []
        batch_outputs = []
        for b in range(word_h_batch.size(0)):
            L = int(word_lengths[b].item())
            gm = gold_maps[b] if gold_maps is not None else None
            loss_b, pred_outs = self._score_sentence(word_h_batch[b, :L], pred_positions[b], gm)
            if loss_b is not None:
                losses.append(loss_b)
            batch_outputs.append(pred_outs)
        out: Dict[str, object] = {"outputs": batch_outputs}
        if losses:
            out["loss"] = torch.stack(losses).mean()
        return out


class SpanBiaffineTrainer:
    """High-speed trainer for Multi-Predicate Span-Biaffine SRL (Monolingual or Polyglot EN+ES)."""

    def __init__(
        self,
        role_inventory: Sequence[str],
        model_name: str = "xlm-roberta-base",
        max_length: int = 256,
        max_width: int = 30,
        batch_size: int = 16,
        epochs: int = 8,
        backbone_lr: float = 2e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 0.01,
        dropout: float = 0.2,
        patience: int = 3,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        _set_seed(seed)
        self.role_inventory = list(role_inventory)
        self.role2id = {r: i for i, r in enumerate(self.role_inventory)}
        self.id2role = {i: r for i, r in enumerate(self.role_inventory)}
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = _load_tokenizer(model_name)
        self.net = SpanBiaffineSRLNet(
            model_name=model_name,
            id2role=self.id2role,
            max_width=max_width,
            dropout=dropout,
        ).to(self.device)
        self.collate_fn = make_grouped_collate_fn(self.tokenizer.pad_token_id or 0)
        self.history: List[Dict[str, float]] = []
        self._fitted = False

    def _amp_config(self) -> Tuple[torch.dtype, bool, bool]:
        is_deberta = "deberta" in self.model_name.lower()
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, True, False
            if is_deberta:
                return torch.float32, False, False
            return torch.float16, True, True
        return torch.float32, False, False

    def _build_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        enc_d, enc_nd, head_d, head_nd = [], [], [], []
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            is_enc = name.startswith("encoder.")
            is_nd = any(nd in name for nd in no_decay)
            if is_enc and not is_nd:
                enc_d.append(param)
            elif is_enc and is_nd:
                enc_nd.append(param)
            elif not is_enc and not is_nd:
                head_d.append(param)
            else:
                head_nd.append(param)
        return torch.optim.AdamW(
            [
                {"params": enc_d, "lr": self.backbone_lr, "weight_decay": self.weight_decay},
                {"params": enc_nd, "lr": self.backbone_lr, "weight_decay": 0.0},
                {"params": head_d, "lr": self.head_lr, "weight_decay": self.weight_decay},
                {"params": head_nd, "lr": self.head_lr, "weight_decay": 0.0},
            ]
        )

    def fit(
        self,
        train_instances: Sequence[dict],
        dev_instances: Optional[Sequence[dict]] = None,
        save_path: Optional[str] = None,
        num_workers: int = 2,
    ) -> List[Dict[str, float]]:
        from transformers import get_cosine_schedule_with_warmup

        grouped_train = group_by_sentence(train_instances)
        train_ds = GroupedSentenceDataset(grouped_train, self.tokenizer, self.role2id, self.max_length)
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers if self.device.type == "cuda" else 0,
            pin_memory=(self.device.type == "cuda"),
        )

        optimizer = self._build_optimizer()
        total_steps = len(train_dl) * self.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps,
        )
        amp_dtype, use_amp, use_scaler = self._amp_config()
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        best_dev_f1 = -1.0
        best_state = None
        no_improve = 0
        self.history = []
        dev_gold = [inst["labels"] for inst in dev_instances] if dev_instances else None

        print(
            f"Training Span-Biaffine ({self.model_name}) on {len(train_instances):,} pred instances "
            f"grouped into {len(grouped_train):,} unique sentences ({len(train_instances)/max(1,len(grouped_train)):.2f}x speedup) "
            f"| device={self.device} | AMP={use_amp} ({amp_dtype})"
        )

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total_loss = 0.0
            t_ep = time.time()
            bar = tqdm(train_dl, desc=f"Epoch {epoch}/{self.epochs}")
            for batch in bar:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = self.net(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        word_subword_idx=batch["word_subword_idx"].to(self.device),
                        word_lengths=batch["word_lengths"],
                        pred_positions=batch["pred_positions"],
                        gold_maps=batch["gold_maps"],
                    )
                    loss = out["loss"]

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
            ep_sec = round(time.time() - t_ep, 1)
            rec: Dict[str, float] = {"epoch": epoch, "train_loss": round(avg_loss, 4), "time_s": ep_sec}

            if dev_instances:
                self._fitted = True
                dev_preds, _ = self.predict_with_latency(dev_instances, num_workers=num_workers, show_progress=False)
                dev_m = compute_metrics(dev_gold, dev_preds)
                rec.update({"dev_p": dev_m["precision"], "dev_r": dev_m["recall"], "dev_f1": dev_m["f1"]})
                print(
                    f"  Epoch {epoch} ({ep_sec}s): loss={avg_loss:.4f} | "
                    f"Dev P={dev_m['precision']:.4f} R={dev_m['recall']:.4f} F1={dev_m['f1']:.4f}"
                )
                if dev_m["f1"] > best_dev_f1:
                    best_dev_f1 = dev_m["f1"]
                    best_state = deepcopy(self.net.state_dict())
                    no_improve = 0
                    if save_path:
                        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                        torch.save(
                            {
                                "state_dict": best_state,
                                "model_name": self.model_name,
                                "role_inventory": self.role_inventory,
                                "best_dev_f1": best_dev_f1,
                            },
                            save_path,
                        )
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        print(f"  Early stopping at epoch {epoch} (best Dev F1={best_dev_f1:.4f}).")
                        break
            else:
                print(f"  Epoch {epoch} ({ep_sec}s): avg loss={avg_loss:.4f}")
            self.history.append(rec)

        if best_state is not None:
            self.net.load_state_dict(best_state)
            print(f"  Restored best checkpoint (Dev F1={best_dev_f1:.4f}).")
        self._fitted = True
        return self.history

    @torch.no_grad()
    def predict_with_latency(
        self,
        instances: Sequence[dict],
        enforce_unique_core: bool = True,
        num_workers: int = 2,
        show_progress: bool = True,
    ) -> Tuple[List[List[str]], float]:
        """Predict BIO sequences aligned 1-to-1 with `instances` and return avg latency (ms/instance)."""
        assert self._fitted, "Call fit() before predict()."
        grouped = group_by_sentence(instances)
        ds = GroupedSentenceDataset(grouped, self.tokenizer, self.role2id, self.max_length)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers if self.device.type == "cuda" else 0,
            pin_memory=(self.device.type == "cuda"),
        )
        self.net.eval()
        amp_dtype, use_amp, _ = self._amp_config()

        preds_by_orig_idx: List[Optional[List[str]]] = [None] * len(instances)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        iterator = tqdm(dl, desc="Predicting") if show_progress else dl
        for batch in iterator:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = self.net(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    word_subword_idx=batch["word_subword_idx"].to(self.device),
                    word_lengths=batch["word_lengths"],
                    pred_positions=batch["pred_positions"],
                    gold_maps=None,
                )

            for b, pred_outs in enumerate(out["outputs"]):
                kept_orig = batch["kept_orig_indices"][b]
                orig_n = batch["orig_lens"][b]
                orig_indices = batch["orig_indices"][b]
                pred_positions = batch["pred_positions"][b]

                for p_local, (logits_kr, c_starts, c_ends) in enumerate(pred_outs):
                    p_kept = pred_positions[p_local]
                    decoded_kept = decode_predicate_spans(
                        logits_kr,
                        c_starts,
                        c_ends,
                        pred_pos=p_kept,
                        id2role=self.id2role,
                        enforce_unique_core=enforce_unique_core,
                    )
                    # Map kept-word spans back to original word indices
                    orig_spans = []
                    for role, s_k, e_k_excl in decoded_kept:
                        s_orig = kept_orig[s_k]
                        e_orig = kept_orig[e_k_excl - 1] + 1
                        orig_spans.append((role, s_orig, e_orig))
                    orig_i = orig_indices[p_local]
                    pred_orig_idx = instances[orig_i]["predicate_idx"]
                    preds_by_orig_idx[orig_i] = spans_to_bio(orig_n, pred_orig_idx, orig_spans)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000.0
        ms_per_inst = round(elapsed_ms / max(1, len(instances)), 3)

        final_preds = [
            p if p is not None else ["O"] * len(instances[i]["words"])
            for i, p in enumerate(preds_by_orig_idx)
        ]
        return final_preds, ms_per_inst

    def predict(self, instances: Sequence[dict], **kwargs) -> List[List[str]]:
        preds, _ = self.predict_with_latency(instances, **kwargs)
        return preds
