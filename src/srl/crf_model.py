"""Linear-Chain CRF SRL (v2) with 1-to-1 spaCy alignment, LCA dependency paths, and subtree cues."""

from typing import Dict, List, Optional, Sequence, Tuple
import sklearn_crfsuite
from spacy.tokens import Doc


def _word_shape(word: str) -> str:
    out = []
    prev = None
    for ch in word:
        c = "X" if ch.isupper() else "x" if ch.islower() else "D" if ch.isdigit() else ch
        if c != prev:
            out.append(c)
            prev = c
    return "".join(out)[:8]


def _safe(seq: Sequence[str], i: int, default: str = "") -> str:
    return seq[i] if 0 <= i < len(seq) else default


def _ancestors_chain(tok) -> List:
    """Return [tok, tok.head, ..., ROOT] with cycle guard."""
    chain = [tok]
    seen = {tok.i}
    cur = tok
    while cur.head != cur and cur.head.i not in seen:
        cur = cur.head
        chain.append(cur)
        seen.add(cur.i)
    return chain


def get_lca_path(token_idx: int, pred_idx: int, doc: Doc, max_len: int = 8) -> Tuple[str, int]:
    """Compute directional dependency path via Lowest Common Ancestor (token -> LCA -> pred).

    Returns (path_string, path_length).
    Examples:
        - token == pred -> ("SELF", 0)
        - token.head == pred (direct child) -> ("^nsubj", 1)
        - token under prep under pred -> ("^pobj^prep", 2)
        - shared subject via matrix verb -> ("^nsubj_v_xcomp", 2)
    """
    if token_idx == pred_idx:
        return "SELF", 0
    if token_idx >= len(doc) or pred_idx >= len(doc) or token_idx < 0 or pred_idx < 0:
        return "OOB", 99

    t_chain = _ancestors_chain(doc[token_idx])
    p_chain = _ancestors_chain(doc[pred_idx])
    p_pos = {tok.i: idx for idx, tok in enumerate(p_chain)}

    lca_t_idx = None
    lca_p_idx = None
    for idx, tok in enumerate(t_chain):
        if tok.i in p_pos:
            lca_t_idx = idx
            lca_p_idx = p_pos[tok.i]
            break

    if lca_t_idx is None:
        return "DISCONNECTED", 99

    up_steps = [f"^{t_chain[i].dep_}" for i in range(lca_t_idx)]
    down_steps = [f"v{p_chain[i].dep_}" for i in range(lca_p_idx - 1, -1, -1)]
    steps = up_steps + down_steps
    path_len = len(steps)
    if path_len > max_len:
        return f"LONG_{len(up_steps)}u_{len(down_steps)}d", path_len
    return "".join(steps), path_len


def _compute_pred_subtree_info(doc: Doc, pred_idx: int) -> Dict[int, Tuple[str, str, bool, bool]]:
    """For each token dominated by a direct child of pred_tok, map:
        tok.i -> (top_child_dep, governing_prep, is_leftmost_in_subtree, is_rightmost_in_subtree)
    This is one of the strongest span-consistency features in statistical SRL.
    """
    info: Dict[int, Tuple[str, str, bool, bool]] = {}
    if pred_idx < 0 or pred_idx >= len(doc):
        return info

    pred_tok = doc[pred_idx]
    for child in pred_tok.children:
        if child.i == pred_idx:
            continue
        # Governing preposition: either child itself (English ClearNLP 'prep') or a 'case' child (UD 'obl')
        prep = ""
        if child.dep_ in ("prep", "case", "mark"):
            prep = child.lower_
        else:
            for gc in child.children:
                if gc.dep_ in ("case", "mark"):
                    prep = gc.lower_
                    break

        sub_idxs = [s.i for s in child.subtree if s.i != pred_idx]
        if not sub_idxs:
            continue
        left_i = min(sub_idxs)
        right_i = max(sub_idxs)
        for idx in sub_idxs:
            info[idx] = (child.dep_, prep, idx == left_i, idx == right_i)
    return info


def sent2features(words: Sequence[str], pred_idx: int, doc: Doc) -> List[Dict[str, object]]:
    """Extract rich lexical, positional, and aligned syntactic features for one sentence."""
    n = len(words)
    pos_tags = [tok.pos_ for tok in doc]
    tag_tags = [tok.tag_ for tok in doc]
    lemmas = [tok.lemma_.lower() if tok.lemma_ else tok.lower_ for tok in doc]
    deps = [tok.dep_ for tok in doc]

    valid_pred = 0 <= pred_idx < len(doc)
    pred_word = _safe(words, pred_idx).lower()
    pred_lemma = _safe(lemmas, pred_idx)
    pred_pos = _safe(pos_tags, pred_idx)
    is_passive = False
    if valid_pred:
        is_passive = any(
            c.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass", "expl:pass")
            for c in doc[pred_idx].children
        )

    sub_info = _compute_pred_subtree_info(doc, pred_idx) if valid_pred else {}
    top_deps = [sub_info[i][0] if i in sub_info else "NONE" for i in range(n)]

    sent_feats: List[Dict[str, object]] = []
    for i in range(n):
        word = words[i]
        w_low = word.lower()
        pos = _safe(pos_tags, i, "X")
        fine_tag = _safe(tag_tags, i, "X")
        lemma = _safe(lemmas, i, w_low)
        dep = _safe(deps, i, "NONE")
        dist = i - pred_idx
        abs_d = abs(dist)

        feats: Dict[str, object] = {
            "bias": 1.0,
            "lower": w_low,
            "lemma": lemma,
            "shape": _word_shape(word),
            "pos": pos,
            "tag": fine_tag,
            "prefix2": w_low[:2],
            "prefix3": w_low[:3],
            "suffix2": w_low[-2:],
            "suffix3": w_low[-3:],
            "is_upper": word.isupper(),
            "is_title": word.istitle(),
            "is_digit": word.isdigit(),
            "is_punct": not word.isalnum(),
            "is_pred": i == pred_idx,
            "dist_bin": (
                "eq" if dist == 0 else
                "close" if abs_d <= 2 else
                "mid" if abs_d <= 5 else
                "far" if abs_d <= 10 else "very_far"
            ),
            "dist_sign": "before" if dist < 0 else "after" if dist > 0 else "eq",
            "pred_lemma": pred_lemma,
            "pred_pos": pred_pos,
            "is_passive": is_passive,
            "dep": dep,
            "dep+voice": f"{dep}|{'pass' if is_passive else 'act'}",
            "pred+dep": f"{pred_lemma}|{dep}",
            "pred+pos": f"{pred_lemma}|{pos}",
        }

        if valid_pred and i < len(doc):
            tok = doc[i]
            head = tok.head
            path_str, path_len = get_lca_path(i, pred_idx, doc)
            feats["lca_path"] = path_str
            feats["lca_len"] = min(path_len, 8)
            feats["lca_path+voice"] = f"{path_str}|{'P' if is_passive else 'A'}"
            feats["is_pred_child"] = head.i == pred_idx and i != pred_idx
            feats["head_pos"] = head.pos_
            feats["head_dep"] = head.dep_
            feats["head_lemma"] = head.lemma_.lower() if head.lemma_ else head.lower_

            if i in sub_info:
                top_dep, prep, is_left, is_right = sub_info[i]
                feats["in_pred_subtree"] = True
                feats["top_arg_dep"] = top_dep
                feats["top_arg_dep+voice"] = f"{top_dep}|{'P' if is_passive else 'A'}"
                feats["top_arg_dep+sign"] = f"{top_dep}|{feats['dist_sign']}"
                if prep:
                    feats["top_arg_prep"] = prep
                    feats["top_arg_dep+prep"] = f"{top_dep}|{prep}"
                feats["is_sub_start"] = is_left
                feats["is_sub_end"] = is_right
            else:
                feats["in_pred_subtree"] = False
                feats["top_arg_dep"] = "NONE"

        # Context window (-2, -1, +1, +2)
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < n:
                feats[f"w{off:+d}_lower"] = words[j].lower()
                feats[f"w{off:+d}_pos"] = _safe(pos_tags, j)
                feats[f"w{off:+d}_dep"] = _safe(deps, j)
                if off in (-1, 1):
                    feats[f"w{off:+d}_top_dep"] = top_deps[j]
                    feats[f"same_top_dep{off:+d}"] = top_deps[j] == top_deps[i]
            else:
                feats[f"w{off:+d}_EDGE"] = True

        if i == 0:
            feats["BOS"] = True
        if i == n - 1:
            feats["EOS"] = True
        sent_feats.append(feats)
    return sent_feats


def parse_aligned_docs(instances: Sequence[dict], nlp_model, batch_size: int = 256) -> List[Doc]:
    """Run spaCy pipeline on pre-tokenized words using Doc(vocab, words=w).

    Deduplicates identical sentences across multiple predicates (~3x speedup on PropBank!).
    """
    unique_keys: Dict[Tuple[str, ...], int] = {}
    unique_words: List[List[str]] = []
    inst_to_uid: List[int] = []

    for inst in instances:
        key = tuple(inst["words"])
        uid = unique_keys.get(key)
        if uid is None:
            uid = len(unique_words)
            unique_keys[key] = uid
            unique_words.append(inst["words"])
        inst_to_uid.append(uid)

    parsed_unique: List[Doc] = []
    for st in range(0, len(unique_words), batch_size):
        batch = [Doc(nlp_model.vocab, words=w) for w in unique_words[st : st + batch_size]]
        for _, comp in nlp_model.pipeline:
            batch = list(comp.pipe(batch))
        parsed_unique.extend(batch)

    return [parsed_unique[uid] for uid in inst_to_uid]


def batch_sent2features(
    instances: Sequence[dict],
    nlp_model,
    docs: Optional[Sequence[Doc]] = None,
    batch_size: int = 256,
) -> List[List[Dict[str, object]]]:
    if docs is None:
        docs = parse_aligned_docs(instances, nlp_model, batch_size=batch_size)
    return [
        sent2features(inst["words"], inst["predicate_idx"], doc)
        for inst, doc in zip(instances, docs)
    ]


class CRFModel:
    """Linear-chain CRF for SRL with fast convergence and valid-only transition filtering."""

    def __init__(
        self,
        algorithm: str = "l2sgd",
        max_iterations: int = 80,
        c1: float = 0.1,
        c2: float = 0.5,
        min_freq: int = 5,
        all_possible_transitions: bool = False,
        verbose: bool = True,
    ):
        kwargs = dict(
            algorithm=algorithm,
            c2=c2,
            min_freq=min_freq,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
            verbose=verbose,
        )
        if algorithm == "lbfgs":
            kwargs["c1"] = c1
        self.model = sklearn_crfsuite.CRF(**kwargs)
        self._fitted = False

    def fit(
        self,
        train_instances: Sequence[dict],
        nlp_model,
        docs: Optional[Sequence[Doc]] = None,
    ) -> None:
        X = batch_sent2features(train_instances, nlp_model, docs=docs)
        y = [inst["labels"] for inst in train_instances]
        self.model.fit(X, y)
        self._fitted = True

    def predict(
        self,
        instances: Sequence[dict],
        nlp_model,
        docs: Optional[Sequence[Doc]] = None,
    ) -> List[List[str]]:
        assert self._fitted, "Call fit() before predict()."
        X = batch_sent2features(instances, nlp_model, docs=docs)
        return self.model.predict(X)
