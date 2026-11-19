"""CoNLL-2005 span-level SRL evaluation and fine-grained error analysis."""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Span = Tuple[str, int, int]


def bio_to_spans(labels: Sequence[str]) -> List[Span]:
    """Decode a BIO label sequence into (role, start, end) half-open spans."""
    spans: List[Span] = []
    cur_role = None
    cur_start = None
    for i, lbl in enumerate(labels):
        if lbl.startswith("B-"):
            if cur_role is not None:
                spans.append((cur_role, cur_start, i))
            cur_role, cur_start = lbl[2:], i
        elif lbl.startswith("I-"):
            role = lbl[2:]
            if cur_role != role:
                if cur_role is not None:
                    spans.append((cur_role, cur_start, i))
                cur_role, cur_start = role, i
        else:
            if cur_role is not None:
                spans.append((cur_role, cur_start, i))
            cur_role = cur_start = None
    if cur_role is not None:
        spans.append((cur_role, cur_start, len(labels)))
    return spans


def count_illegal_bio(pred_list: Sequence[Sequence[str]]) -> int:
    """Count illegal BIO transitions (O -> I-*, B-X -> I-Y, I-X -> I-Y with X != Y)."""
    illegal = 0
    for seq in pred_list:
        prev_tag = "O"
        prev_role = None
        for lbl in seq:
            if lbl.startswith("I-"):
                role = lbl[2:]
                if prev_tag == "O" or prev_role != role:
                    illegal += 1
                prev_tag, prev_role = "I", role
            elif lbl.startswith("B-"):
                prev_tag, prev_role = "B", lbl[2:]
            else:
                prev_tag, prev_role = "O", None
    return illegal


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def compute_metrics(
    gold_list: Sequence[Sequence[str]],
    pred_list: Sequence[Sequence[str]],
    skip_labels: Iterable[str] = ("O", "V"),
) -> Dict[str, object]:
    """Compute exact span-level Precision, Recall, and F1 following CoNLL-2005."""
    skip = set(skip_labels)
    per_role = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp = total_fp = total_fn = 0

    for gold_seq, pred_seq in zip(gold_list, pred_list):
        n = max(len(gold_seq), len(pred_seq))
        g = list(gold_seq) + ["O"] * (n - len(gold_seq))
        p = list(pred_seq) + ["O"] * (n - len(pred_seq))
        gold_spans = {s for s in bio_to_spans(g) if s[0] not in skip}
        pred_spans = {s for s in bio_to_spans(p) if s[0] not in skip}

        for sp in pred_spans:
            if sp in gold_spans:
                per_role[sp[0]]["tp"] += 1
                total_tp += 1
            else:
                per_role[sp[0]]["fp"] += 1
                total_fp += 1
        for sp in gold_spans:
            if sp not in pred_spans:
                per_role[sp[0]]["fn"] += 1
                total_fn += 1

    prec, rec, f1 = _prf(total_tp, total_fp, total_fn)
    per_role_out = {}
    for role, c in sorted(per_role.items()):
        rp, rr, rf = _prf(c["tp"], c["fp"], c["fn"])
        per_role_out[role] = {
            "precision": rp,
            "recall": rr,
            "f1": rf,
            "support": c["tp"] + c["fn"],
        }

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "illegal_bio": count_illegal_bio(pred_list),
        "per_role": per_role_out,
    }


def error_analysis(
    gold_list: Sequence[Sequence[str]],
    pred_list: Sequence[Sequence[str]],
    skip_labels: Iterable[str] = ("O", "V"),
) -> Dict[str, object]:
    """Decompose span-level errors into role confusion, boundary mismatch, spurious, and missed."""
    skip = set(skip_labels)
    exact_match = 0
    role_confusion = 0
    boundary_only = 0
    role_and_boundary = 0
    missed = 0
    spurious = 0
    confusions = Counter()

    for gold_seq, pred_seq in zip(gold_list, pred_list):
        n = max(len(gold_seq), len(pred_seq))
        g = list(gold_seq) + ["O"] * (n - len(gold_seq))
        p = list(pred_seq) + ["O"] * (n - len(pred_seq))

        gold_spans = {s for s in bio_to_spans(g) if s[0] not in skip}
        pred_spans = {s for s in bio_to_spans(p) if s[0] not in skip}

        matched = gold_spans & pred_spans
        exact_match += len(matched)

        rem_gold = list(gold_spans - matched)
        rem_pred = list(pred_spans - matched)

        # Match exact boundaries first (pure role confusion)
        gold_by_bounds = {(s[1], s[2]): s for s in rem_gold}
        used_gold = set()
        used_pred = set()
        for pi, ps in enumerate(rem_pred):
            b = (ps[1], ps[2])
            if b in gold_by_bounds:
                gs = gold_by_bounds[b]
                role_confusion += 1
                confusions[(gs[0], ps[0])] += 1
                used_gold.add(gs)
                used_pred.add(pi)

        # Match overlapping spans (boundary error vs role+boundary error)
        for pi, ps in enumerate(rem_pred):
            if pi in used_pred:
                continue
            overlaps = [
                gs
                for gs in rem_gold
                if gs not in used_gold and max(gs[1], ps[1]) < min(gs[2], ps[2])
            ]
            if not overlaps:
                spurious += 1
                continue
            # Prefer same-role overlap if available
            same_role = [gs for gs in overlaps if gs[0] == ps[0]]
            chosen = same_role[0] if same_role else overlaps[0]
            used_gold.add(chosen)
            if chosen[0] == ps[0]:
                boundary_only += 1
            else:
                role_and_boundary += 1
                confusions[(chosen[0], ps[0])] += 1

        missed += sum(1 for gs in rem_gold if gs not in used_gold)

    return {
        "exact_match": exact_match,
        "role_confusion": role_confusion,
        "boundary_only": boundary_only,
        "role_and_boundary": role_and_boundary,
        "missed_fn": missed,
        "spurious_fp": spurious,
        "illegal_bio": count_illegal_bio(pred_list),
        "top_confusions": confusions.most_common(12),
    }


def print_results(name: str, results: Dict[str, object], top_k: int = 15) -> None:
    sep = "-" * 66
    print(f"\n{sep}\n  {name}\n{sep}")
    print(
        f"  Precision={results['precision']:.4f}  Recall={results['recall']:.4f}  "
        f"F1={results['f1']:.4f}  IllegalBIO={results.get('illegal_bio', 0)}"
    )
    print(f"  TP={results['tp']}  FP={results['fp']}  FN={results['fn']}")
    per_role = results.get("per_role") or {}
    if per_role:
        print(f"\n  {'Role':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Sup':>6}")
        for role, m in sorted(per_role.items(), key=lambda x: -x[1]["support"])[:top_k]:
            print(
                f"  {role:<20} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                f"{m['f1']:>8.4f} {m['support']:>6}"
            )
    print(sep)


def print_error_analysis(name: str, analysis: Dict[str, object]) -> None:
    print(f"\n  [Error Analysis: {name}]")
    print(
        f"    Exact={analysis['exact_match']}  "
        f"BoundaryOnly={analysis['boundary_only']}  "
        f"RoleConfusion={analysis['role_confusion']}  "
        f"Role+Boundary={analysis['role_and_boundary']}  "
        f"Missed={analysis['missed_fn']}  "
        f"Spurious={analysis['spurious_fp']}  "
        f"IllegalBIO={analysis['illegal_bio']}"
    )
    if analysis["top_confusions"]:
        print("    Top role confusions (Gold -> Pred):")
        for (g, p), cnt in analysis["top_confusions"][:8]:
            print(f"      {g:<14} -> {p:<14} ({cnt})")
