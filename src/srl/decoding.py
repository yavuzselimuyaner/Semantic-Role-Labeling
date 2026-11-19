"""BIO-constrained Viterbi decoding and pure-PyTorch Linear-Chain CRF layer."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


def build_bio_constraint_masks(
    id2label: Dict[int, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build boolean validity masks for BIO sequence tagging.

    Returns:
        start_allowed: (C,) bool tensor — True if label can start a sequence (O or B-*).
        trans_allowed: (C, C) bool tensor — trans_allowed[i, j] is True if i -> j is valid BIO.
    """
    num_labels = len(id2label)
    start_allowed = torch.ones(num_labels, dtype=torch.bool)
    trans_allowed = torch.ones((num_labels, num_labels), dtype=torch.bool)

    parsed = []
    for i in range(num_labels):
        lbl = id2label[i]
        if lbl.startswith("B-"):
            parsed.append(("B", lbl[2:]))
        elif lbl.startswith("I-"):
            parsed.append(("I", lbl[2:]))
            start_allowed[i] = False  # Sequence cannot start with I-*
        else:
            parsed.append(("O", None))

    for prev_i in range(num_labels):
        prev_prefix, prev_role = parsed[prev_i]
        for next_j in range(num_labels):
            next_prefix, next_role = parsed[next_j]
            if next_prefix == "I":
                # I-X can only follow B-X or I-X with matching role X
                if prev_prefix == "O" or prev_role != next_role:
                    trans_allowed[prev_i, next_j] = False

    return start_allowed, trans_allowed


class ConstrainedLinearCRF(nn.Module):
    """Exact word-level Linear-Chain CRF with hard BIO grammar constraints.

    Computes exact log-partition function via forward algorithm during training
    and exact constrained Viterbi decoding during inference.
    """

    def __init__(self, id2label: Dict[int, str], enforce_bio: bool = True):
        super().__init__()
        self.num_tags = len(id2label)
        self.enforce_bio = enforce_bio

        self.start_transitions = nn.Parameter(torch.zeros(self.num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(self.num_tags))
        self.transitions = nn.Parameter(torch.zeros(self.num_tags, self.num_tags))

        start_mask, trans_mask = build_bio_constraint_masks(id2label)
        self.register_buffer("start_allowed", start_mask)
        self.register_buffer("trans_allowed", trans_mask)

    def _masked_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enforce_bio:
            return self.start_transitions, self.transitions, self.end_transitions
        neg_inf = -1e4
        start_t = self.start_transitions.masked_fill(~self.start_allowed, neg_inf)
        trans_t = self.transitions.masked_fill(~self.trans_allowed, neg_inf)
        return start_t, trans_t, self.end_transitions

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean negative log-likelihood over batch.

        Args:
            emissions: (B, L, C) float tensor of word-level tag logits.
            tags:      (B, L) long tensor of gold tag ids (valid everywhere mask==True).
            mask:      (B, L) bool tensor (True for real words, False for padding).
        """
        # Cast to float32 for numerical stability under AMP fp16/bf16.
        # Keep transitions unconstrained during NLL training so noisy gold BIO
        # annotations never cause -1e4 loss spikes; hard BIO constraints apply in decode().
        emissions = emissions.float()
        mask = mask.bool()
        start_t, trans_t, end_t = self.start_transitions, self.transitions, self.end_transitions

        gold_score = self._score_sentence(emissions, tags, mask, start_t, trans_t, end_t)
        log_z = self._log_partition(emissions, mask, start_t, trans_t, end_t)
        nll = log_z - gold_score
        # Normalize by total valid tokens for stable gradient scale across batch lengths
        return nll.sum() / mask.float().sum().clamp_min(1.0)

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        start_t: torch.Tensor,
        trans_t: torch.Tensor,
        end_t: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = emissions.shape
        safe_tags = tags.clamp_min(0)
        score = start_t[safe_tags[:, 0]] + emissions[torch.arange(B, device=emissions.device), 0, safe_tags[:, 0]]

        for t in range(1, L):
            m = mask[:, t].float()
            emit = emissions[torch.arange(B, device=emissions.device), t, safe_tags[:, t]]
            trans = trans_t[safe_tags[:, t - 1], safe_tags[:, t]]
            score = score + (emit + trans) * m

        last_idx = mask.long().sum(dim=1) - 1
        last_tags = safe_tags[torch.arange(B, device=emissions.device), last_idx]
        score = score + end_t[last_tags]
        return score

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        start_t: torch.Tensor,
        trans_t: torch.Tensor,
        end_t: torch.Tensor,
    ) -> torch.Tensor:
        _, L, _ = emissions.shape
        # alpha: (B, C)
        alpha = start_t.unsqueeze(0) + emissions[:, 0, :]

        for t in range(1, L):
            # scores: (B, C_prev, C_next)
            scores = alpha.unsqueeze(2) + trans_t.unsqueeze(0) + emissions[:, t, :].unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m, next_alpha, alpha)

        alpha = alpha + end_t.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Exact Viterbi decoding returning list of tag-id sequences."""
        emissions = emissions.float()
        mask = mask.bool()
        B, L, _ = emissions.shape
        start_t, trans_t, end_t = self._masked_params()

        score = start_t.unsqueeze(0) + emissions[:, 0, :]  # (B, C)
        backpointers: List[torch.Tensor] = []

        for t in range(1, L):
            # (B, C_prev, C_next)
            next_scores = score.unsqueeze(2) + trans_t.unsqueeze(0)
            best_prev_score, best_prev_tag = next_scores.max(dim=1)
            next_score = best_prev_score + emissions[:, t, :]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, next_score, score)
            backpointers.append(best_prev_tag)

        score = score + end_t.unsqueeze(0)
        lengths = mask.long().sum(dim=1).tolist()

        best_paths: List[List[int]] = []
        for b in range(B):
            seq_len = lengths[b]
            best_last = int(score[b].argmax().item())
            path = [best_last]
            for t in range(seq_len - 2, -1, -1):
                best_last = int(backpointers[t][b, best_last].item())
                path.append(best_last)
            path.reverse()
            best_paths.append(path)
        return best_paths


@torch.no_grad()
def viterbi_decode_bio(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    id2label: Dict[int, str],
) -> List[List[int]]:
    """Zero-parameter BIO-constrained Viterbi decoding for plain softmax/linear heads."""
    crf = ConstrainedLinearCRF(id2label, enforce_bio=True).to(emissions.device)
    return crf.decode(emissions, mask)
