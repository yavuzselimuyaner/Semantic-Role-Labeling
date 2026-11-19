"""Dataset loading, label inventory extraction, and diagnostics for PropBank BIO datasets."""

from collections import Counter
import json
import random
from typing import Dict, List, Sequence, Tuple


def load_dataset(json_path: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """Load preprocessed PropBank-EWT or PropBank-ES JSON into (train, dev, test) splits."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    train, dev, test = [], [], []
    for inst in data["instances"]:
        sp = inst.get("split", "train")
        if sp == "train":
            train.append(inst)
        elif sp == "dev":
            dev.append(inst)
        else:
            test.append(inst)
    return train, dev, test


def truncate(items: Sequence[dict], max_n: int, seed: int = 42) -> List[dict]:
    """Deterministically subsample a list of instances."""
    items = list(items)
    if max_n is None or len(items) <= max_n:
        return items
    return random.Random(seed).sample(items, max_n)


def get_label_set(*splits: Sequence[dict]) -> List[str]:
    """Return sorted BIO label vocabulary with 'O' fixed at index 0."""
    labels = set()
    for split in splits:
        for inst in split:
            labels.update(inst["labels"])
    labels.discard("O")
    return ["O"] + sorted(labels)


def dataset_stats(instances: Sequence[dict], name: str = "dataset") -> Dict[str, object]:
    """Compute span length distribution, sentence length stats, and role counts."""
    from .metrics import bio_to_spans

    sent_lens = [len(inst["words"]) for inst in instances]
    span_lens = []
    role_counts = Counter()
    for inst in instances:
        for role, start, end in bio_to_spans(inst["labels"]):
            if role in ("O", "V"):
                continue
            span_lens.append(end - start)
            role_counts[role] += 1

    n_inst = len(instances)
    n_spans = len(span_lens)
    single_token_ratio = (
        sum(1 for l in span_lens if l == 1) / n_spans if n_spans else 0.0
    )
    return {
        "name": name,
        "num_instances": n_inst,
        "num_spans": n_spans,
        "avg_sent_len": round(sum(sent_lens) / n_inst, 2) if n_inst else 0.0,
        "max_sent_len": max(sent_lens) if sent_lens else 0,
        "pct_sent_gt_100": round(
            100.0 * sum(1 for l in sent_lens if l > 100) / n_inst, 2
        )
        if n_inst
        else 0.0,
        "avg_span_len": round(sum(span_lens) / n_spans, 2) if n_spans else 0.0,
        "single_token_span_ratio": round(single_token_ratio, 4),
        "top_roles": role_counts.most_common(12),
    }
