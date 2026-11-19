"""Unified cross-lingual PropBank schema (EN + ES) and multi-predicate sentence grouping."""

from collections import OrderedDict
import random
from typing import Dict, List, Sequence, Tuple

from .metrics import bio_to_spans

CORE_ROLES = {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}

# Normalize Spanish AnCora / UP role names into canonical PropBank-EWT role names
_ES_TO_UNIFIED_MOD = {
    "AM-LOC": "ARGM-LOC",
    "AM-TMP": "ARGM-TMP",
    "AM-MNR": "ARGM-MNR",
    "AM-ADV": "ARGM-ADV",
    "AM-NEG": "ARGM-NEG",
    "AM-MOD": "ARGM-MOD",
    "AM-CAU": "ARGM-CAU",
    "AM-DIS": "ARGM-DIS",
    "AM-EXT": "ARGM-EXT",
    "AM-DIR": "ARGM-DIR",
    "AM-PRD": "ARGM-PRD",
    "AM-PRP": "ARGM-PRP",
    "AM-PNC": "ARGM-PRP",
    "AM-ADJ": "ARGM-ADJ",
    "AM-GOL": "ARGM-GOL",
    "AM-COM": "ARGM-COM",
    "AM-REC": "ARGM-REC",
    "AM-LVB": "ARGM-LVB",
    "AM-CXN": "ARGM-CXN",
}


def normalize_role(role: str, language: str = "propbank") -> str:
    """Map Spanish AnCora role notation (A0, AM-LOC, R-A0, C-A1) to unified PropBank notation."""
    if language != "propbank_es":
        # Harmonize PNC -> PRP if present
        if role == "ARGM-PNC":
            return "ARGM-PRP"
        return role

    # Handle continuation/reference prefixes (C-, R-)
    prefix = ""
    base = role
    for p in ("C-", "R-"):
        if base.startswith(p):
            prefix = p
            base = base[len(p) :]
            break

    if base in ("O", "V"):
        return role
    if len(base) == 2 and base[0] == "A" and base[1].isdigit():
        return f"{prefix}ARG{base[1]}"
    if base.startswith("AM-"):
        mapped = _ES_TO_UNIFIED_MOD.get(base, "ARGM-" + base[3:])
        return f"{prefix}{mapped}"
    return role


def normalize_bio_label(label: str, language: str = "propbank") -> str:
    """Normalize a BIO tag string (e.g. B-A0 -> B-ARG0, I-AM-TMP -> I-ARGM-TMP)."""
    if label == "O" or "-" not in label:
        return label
    bio, role = label.split("-", 1)
    return f"{bio}-{normalize_role(role, language=language)}"


def unify_instances(instances: Sequence[dict], language: str) -> List[dict]:
    """Return copies of instances with unified cross-lingual PropBank labels and language tag."""
    out = []
    for idx, inst in enumerate(instances):
        out.append(
            {
                "id": inst.get("id", f"{language}_{idx}"),
                "language": language,
                "split": inst.get("split", "train"),
                "words": list(inst["words"]),
                "predicate_idx": int(inst["predicate_idx"]),
                "labels": [normalize_bio_label(l, language=language) for l in inst["labels"]],
            }
        )
    return out


def group_by_sentence(instances: Sequence[dict]) -> List[dict]:
    """Group predicate-level instances sharing identical token sequences into multi-predicate sentences.

    Each grouped record has:
        words: List[str]
        language: str
        predicates: List[int]                    # predicate word indices
        orig_indices: List[int]                  # index back into original `instances`
        gold_spans: List[List[Tuple[str,int,int]]]  # per-predicate (role, start, end) excluding V
    """
    groups: "OrderedDict[Tuple[str, Tuple[str, ...]], dict]" = OrderedDict()
    for orig_idx, inst in enumerate(instances):
        lang = inst.get("language", "propbank")
        key = (lang, tuple(inst["words"]))
        if key not in groups:
            groups[key] = {
                "words": list(inst["words"]),
                "language": lang,
                "predicates": [],
                "orig_indices": [],
                "gold_spans": [],
            }
        g = groups[key]
        g["predicates"].append(int(inst["predicate_idx"]))
        g["orig_indices"].append(orig_idx)
        spans = [
            (role, s, e)
            for (role, s, e) in bio_to_spans(inst["labels"])
            if role not in ("O", "V")
        ]
        g["gold_spans"].append(spans)
    return list(groups.values())


def get_role_inventory(*instance_lists: Sequence[dict], min_count: int = 1) -> List[str]:
    """Extract sorted span-role inventory with 'NULL' fixed at index 0 (excludes 'O' and 'V')."""
    from collections import Counter

    counts = Counter()
    for lst in instance_lists:
        for inst in lst:
            for role, _, _ in bio_to_spans(inst["labels"]):
                if role not in ("O", "V"):
                    counts[role] += 1
    roles = sorted(r for r, c in counts.items() if c >= min_count)
    return ["NULL"] + roles


def spans_to_bio(n_words: int, predicate_idx: int, spans: Sequence[Tuple[str, int, int]]) -> List[str]:
    """Convert non-overlapping (role, start, end) spans back into a BIO sequence for evaluation."""
    labels = ["O"] * n_words
    if 0 <= predicate_idx < n_words:
        labels[predicate_idx] = "B-V"
    for role, start, end in spans:
        if start < 0 or end > n_words or start >= end:
            continue
        if any(labels[i] != "O" for i in range(start, end)):
            continue
        labels[start] = f"B-{role}"
        for i in range(start + 1, end):
            labels[i] = f"I-{role}"
    return labels


def make_stratified_subset(instances: Sequence[dict], n: int = 100, seed: int = 42) -> List[dict]:
    """Deterministically sample instances covering short, medium, and long sentences."""
    items = list(instances)
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    short = [x for x in items if len(x["words"]) <= 12]
    med = [x for x in items if 12 < len(x["words"]) <= 25]
    long_ = [x for x in items if len(x["words"]) > 25]
    buckets = [short, med, long_]
    chosen = []
    quota = n // 3
    for b in buckets:
        k = min(len(b), quota)
        chosen.extend(rng.sample(b, k))
    remaining = [x for x in items if x not in chosen]
    if len(chosen) < n and remaining:
        chosen.extend(rng.sample(remaining, min(n - len(chosen), len(remaining))))
    rng.shuffle(chosen)
    return chosen
