"""Dependency-based Rule-Based SRL (v2) for PropBank-EWT (EN) and PropBank-ES (AnCora)."""

from typing import Dict, List, Optional, Sequence
from spacy.tokens import Doc

# Core grammatical relations -> PropBank roles (EN)
_DEP_TO_ROLE_EN: Dict[str, str] = {
    "nsubj": "ARG0",
    "csubj": "ARG0",
    "agent": "ARG0",
    "nsubjpass": "ARG1",
    "nsubj:pass": "ARG1",
    "csubjpass": "ARG1",
    "csubj:pass": "ARG1",
    "dobj": "ARG1",
    "obj": "ARG1",
    "attr": "ARG1",
    "oprd": "ARG1",
    "xcomp": "ARG1",
    "ccomp": "ARG1",
    "iobj": "ARG2",
    "dative": "ARG2",
    "acomp": "ARG2",
}

# Preposition -> ARGM suffix (prefix ARGM- or AM- is applied dynamically!)
_PREP_TO_SUFFIX_EN: Dict[str, str] = {
    "in": "LOC", "at": "LOC", "on": "LOC", "under": "LOC", "over": "LOC",
    "near": "LOC", "beside": "LOC", "along": "LOC", "above": "LOC", "below": "LOC",
    "behind": "LOC", "inside": "LOC", "outside": "LOC", "between": "LOC", "across": "LOC",
    "to": "DIR", "into": "DIR", "onto": "DIR", "toward": "DIR", "towards": "DIR", "from": "DIR",
    "after": "TMP", "before": "TMP", "during": "TMP", "since": "TMP", "until": "TMP",
    "till": "TMP", "within": "TMP",
    "by": "MNR", "with": "MNR", "without": "MNR", "via": "MNR", "through": "MNR", "as": "MNR",
    "because": "CAU", "due": "CAU",
    "for": "PRP",
    "of": "ADV", "about": "ADV", "despite": "ADV", "against": "ADV",
}

_PREP_TO_SUFFIX_ES: Dict[str, str] = {
    "en": "LOC", "sobre": "LOC", "bajo": "LOC", "cerca": "LOC", "junto": "LOC",
    "dentro": "LOC", "fuera": "LOC", "entre": "LOC", "ante": "LOC", "frente": "LOC",
    "a": "DIR", "hacia": "DIR", "desde": "DIR",
    "hasta": "TMP", "antes": "TMP", "después": "TMP", "despues": "TMP",
    "durante": "TMP", "tras": "TMP",
    "por": "MNR", "con": "MNR", "sin": "MNR", "mediante": "MNR", "según": "MNR", "segun": "MNR",
    "porque": "CAU",
    "para": "PRP",
    "de": "ADV", "contra": "ADV",
}

_MODAL_LEMMAS_EN = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}
_MODAL_LEMMAS_ES = {"poder", "deber", "soler"}

_NEG_WORDS_EN = {"not", "n't", "never", "no", "neither", "nor", "hardly", "barely"}
_NEG_WORDS_ES = {"no", "nunca", "jamás", "jamas", "tampoco"}

_TMP_WORDS_EN = {
    "now", "then", "today", "yesterday", "tomorrow", "always", "often", "still",
    "already", "soon", "recently", "ever", "again", "later", "early", "late",
    "immediately", "currently", "finally", "eventually", "sometimes", "usually",
    "first", "last", "once", "twice", "tonight", "nowadays", "ago", "yet",
}
_TMP_WORDS_ES = {
    "ahora", "hoy", "ayer", "mañana", "siempre", "ya", "todavía", "todavia",
    "aún", "aun", "luego", "después", "despues", "antes", "pronto", "tarde",
    "temprano", "nunca", "entonces", "finalmente", "actualmente", "recientemente",
}

_LOC_WORDS_EN = {"here", "there", "where", "everywhere", "somewhere", "anywhere", "nowhere", "abroad", "home", "away", "back"}
_LOC_WORDS_ES = {"aquí", "aqui", "allí", "alli", "ahí", "ahi", "acá", "allá", "donde", "dónde", "lejos", "cerca", "fuera", "dentro"}

_DIS_WORDS_EN = {"however", "well", "anyway", "therefore", "moreover", "meanwhile", "instead", "otherwise", "furthermore", "actually", "basically", "so", "oh", "yes", "yeah", "please"}
_DIS_WORDS_ES = {"además", "ademas", "así", "entonces", "igualmente", "asimismo", "claro", "bueno"}

_EXT_WORDS_EN = {"very", "too", "more", "most", "much", "less", "almost", "quite", "nearly", "completely", "totally", "entirely", "slightly", "partly", "far", "way"}
_EXT_WORDS_ES = {"muy", "más", "mas", "menos", "mucho", "poco", "casi", "bastante", "demasiado", "totalmente", "completamente"}

_TMP_NOUNS_EN = {"today", "tomorrow", "yesterday", "day", "week", "month", "year", "morning", "afternoon", "evening", "night", "time", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
_TMP_NOUNS_ES = {"hoy", "ayer", "mañana", "día", "dia", "semana", "mes", "año", "ano", "noche", "tarde", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"}

_CLAUSE_MARK_EN = {
    "when": "TMP", "while": "TMP", "before": "TMP", "after": "TMP", "until": "TMP", "since": "TMP", "once": "TMP",
    "because": "CAU", "as": "CAU",
    "if": "ADV", "unless": "ADV", "although": "ADV", "though": "ADV", "whereas": "ADV", "whether": "ADV",
    "so": "PRP", "to": "PRP",
}
_CLAUSE_MARK_ES = {
    "cuando": "TMP", "mientras": "TMP", "antes": "TMP", "después": "TMP", "hasta": "TMP",
    "porque": "CAU", "pues": "CAU", "como": "CAU",
    "si": "ADV", "aunque": "ADV",
    "para": "PRP",
}

_PUNCT_CHARS = set(".,;:!?\"'()[]{}")


def _to_ancora(role: str) -> str:
    if role.startswith("ARG"):
        return "A" + role[3:]
    return role


def _is_punct_token(tok) -> bool:
    return tok.is_punct or (len(tok.text) > 0 and all(ch in _PUNCT_CHARS for ch in tok.text))


class RuleBasedSRL:
    """Syntax-driven rule-based SRL baseline supporting English (ClearNLP/UD) and Spanish (UD)."""

    def __init__(self, nlp, mode: str = "propbank", batch_size: int = 256):
        self.nlp = nlp
        self.mode = mode
        self.batch_size = batch_size
        self.is_es = mode == "propbank_es"

        if self.is_es:
            self._dep_map = {k: _to_ancora(v) for k, v in _DEP_TO_ROLE_EN.items()}
            self._dep_map.update({
                "obl:agent": "A0",
                "obl:arg": "A2",
                "iobj": "A2",
            })
            self._prep_map = _PREP_TO_SUFFIX_ES
            self._argm_prefix = "AM-"
            self._modal_lemmas = _MODAL_LEMMAS_ES
            self._neg_words = _NEG_WORDS_ES
            self._tmp_words = _TMP_WORDS_ES
            self._loc_words = _LOC_WORDS_ES
            self._dis_words = _DIS_WORDS_ES
            self._ext_words = _EXT_WORDS_ES
            self._tmp_nouns = _TMP_NOUNS_ES
            self._clause_mark = _CLAUSE_MARK_ES
        else:
            self._dep_map = dict(_DEP_TO_ROLE_EN)
            self._prep_map = _PREP_TO_SUFFIX_EN
            self._argm_prefix = "ARGM-"
            self._modal_lemmas = _MODAL_LEMMAS_EN
            self._neg_words = _NEG_WORDS_EN
            self._tmp_words = _TMP_WORDS_EN
            self._loc_words = _LOC_WORDS_EN
            self._dis_words = _DIS_WORDS_EN
            self._ext_words = _EXT_WORDS_EN
            self._tmp_nouns = _TMP_NOUNS_EN
            self._clause_mark = _CLAUSE_MARK_EN

    def _argm(self, suffix: str) -> str:
        return f"{self._argm_prefix}{suffix}"

    def parse_docs(self, instances: Sequence[dict]) -> List[Doc]:
        """Parse pre-tokenized instances without re-tokenization, deduplicating shared sentences."""
        unique_keys = {}
        unique_words = []
        inst_to_uid = []
        for inst in instances:
            key = tuple(inst["words"])
            uid = unique_keys.get(key)
            if uid is None:
                uid = len(unique_words)
                unique_keys[key] = uid
                unique_words.append(inst["words"])
            inst_to_uid.append(uid)

        parsed_unique: List[Doc] = []
        for st in range(0, len(unique_words), self.batch_size):
            batch = [Doc(self.nlp.vocab, words=w) for w in unique_words[st : st + self.batch_size]]
            for _, comp in self.nlp.pipeline:
                batch = list(comp.pipe(batch))
            parsed_unique.extend(batch)
        return [parsed_unique[uid] for uid in inst_to_uid]

    def predict_batch(self, instances: Sequence[dict], docs: Optional[Sequence[Doc]] = None) -> List[List[str]]:
        if docs is None:
            docs = self.parse_docs(instances)
        return [
            self._predict_from_doc(doc, inst["predicate_idx"], len(inst["words"]))
            for doc, inst in zip(docs, instances)
        ]

    def _classify_advmod(self, tok) -> str:
        w = tok.lower_
        if w in self._neg_words:
            return self._argm("NEG")
        if w in self._tmp_words:
            return self._argm("TMP")
        if w in self._loc_words:
            return self._argm("LOC")
        if w in self._dis_words:
            return self._argm("DIS")
        if w in self._ext_words:
            return self._argm("EXT")
        if (not self.is_es and w.endswith("ly")) or (self.is_es and w.endswith("mente")):
            return self._argm("MNR")
        if w in ("well", "fast", "hard", "bien", "mal", "rápido", "despacio"):
            return self._argm("MNR")
        return self._argm("ADV")

    def _classify_obl(self, tok, is_passive: bool) -> Optional[str]:
        """In UD (especially Spanish), prepositions sit as 'case' children under 'obl'."""
        case_children = [c.lower_ for c in tok.children if c.dep_ == "case"]
        if case_children:
            prep = case_children[0]
            if is_passive and prep in ("by", "por"):
                return "A0" if self.is_es else "ARG0"
            if self.is_es and prep == "a" and tok.pos_ in ("PROPN", "PRON"):
                return "A2"
            suffix = self._prep_map.get(prep)
            if suffix:
                return self._argm(suffix)
            return self._argm("ADV")
        # Bare oblique (often temporal noun phrase like "last week" / "el lunes")
        if tok.lower_ in self._tmp_nouns or any(c.lower_ in self._tmp_nouns for c in tok.children):
            return self._argm("TMP")
        return self._argm("ADV")

    def _role_for(self, tok, is_passive: bool) -> Optional[str]:
        dep = tok.dep_
        if dep in self._dep_map:
            role = self._dep_map[dep]
            # Passive subject mapped to ARG1/A1 even if parser only labeled nsubj
            if is_passive and dep in ("nsubj", "csubj"):
                return "A1" if self.is_es else "ARG1"
            return role

        # Modals -> ARGM-MOD / AM-MOD
        if dep == "aux":
            if (not self.is_es and tok.tag_ == "MD") or tok.lemma_.lower() in self._modal_lemmas or tok.lower_ in self._modal_lemmas:
                return self._argm("MOD")
            return None

        # Negation
        if dep == "neg":
            return self._argm("NEG")

        # English ClearNLP prepositional phrase child (prep)
        if dep == "prep":
            prep = tok.lower_
            if is_passive and prep == "by":
                return "A0" if self.is_es else "ARG0"
            suffix = self._prep_map.get(prep)
            return self._argm(suffix) if suffix else self._argm("ADV")

        # UD oblique argument (crucial for Spanish AnCora!)
        if dep == "obl" or dep.startswith("obl:"):
            return self._classify_obl(tok, is_passive)

        # Adverbs
        if dep == "advmod":
            return self._classify_advmod(tok)

        # Adverbial clauses (when/because/if/although...)
        if dep == "advcl":
            marks = [c.lower_ for c in tok.children if c.dep_ == "mark"]
            if marks:
                suffix = self._clause_mark.get(marks[0], "ADV")
                return self._argm(suffix)
            return self._argm("ADV")

        # Bare NP adverbials (usually temporal: "last year", "yesterday")
        if dep == "npadvmod":
            if tok.lower_ in self._tmp_nouns or any(c.lower_ in self._tmp_nouns for c in tok.subtree):
                return self._argm("TMP")
            return self._argm("EXT")

        # Discourse markers / interjections
        if dep in ("intj", "discourse"):
            return self._argm("DIS")

        return None

    def _trim_span(self, doc: Doc, idxs: List[int]) -> List[int]:
        """Strip leading/trailing punctuation tokens from a dependency subtree span."""
        left = 0
        right = len(idxs) - 1
        while left <= right and _is_punct_token(doc[idxs[left]]):
            left += 1
        while right >= left and _is_punct_token(doc[idxs[right]]):
            right -= 1
        return idxs[left : right + 1]

    def _write_span(self, labels: List[str], doc: Doc, tok, role: str, predicate_idx: int, n: int) -> None:
        raw_idxs = sorted(s.i for s in tok.subtree if 0 <= s.i < n and s.i != predicate_idx)
        if not raw_idxs:
            return
        span_idxs = self._trim_span(doc, raw_idxs)
        if not span_idxs:
            return
        # If non-contiguous (e.g. predicate splits the subtree), keep contiguous block around tok
        if span_idxs[-1] - span_idxs[0] + 1 != len(span_idxs):
            if labels[tok.i] == "O":
                labels[tok.i] = f"B-{role}"
            return
        if any(labels[i] != "O" for i in span_idxs):
            return
        labels[span_idxs[0]] = f"B-{role}"
        for idx in span_idxs[1:]:
            labels[idx] = f"I-{role}"

    def _predict_from_doc(self, doc: Doc, predicate_idx: int, n: int) -> List[str]:
        labels = ["O"] * n
        if predicate_idx < 0 or predicate_idx >= n or predicate_idx >= len(doc):
            return labels

        labels[predicate_idx] = "B-V"
        pred_tok = doc[predicate_idx]

        is_passive = any(
            c.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass", "expl:pass")
            for c in pred_tok.children
        )

        has_subject = False
        for tok in pred_tok.children:
            if tok.i == predicate_idx or tok.i >= n:
                continue
            role = self._role_for(tok, is_passive)
            if role is None:
                continue
            if tok.dep_ in ("nsubj", "nsubjpass", "nsubj:pass", "csubj"):
                has_subject = True
            self._write_span(labels, doc, tok, role, predicate_idx, n)

        # Recover shared subject for infinitives / coordinated verbs (xcomp, conj)
        if not has_subject and pred_tok.dep_ in ("xcomp", "conj") and pred_tok.head != pred_tok:
            head_tok = pred_tok.head
            pred_subtree = {s.i for s in pred_tok.subtree}
            for sib in head_tok.children:
                if sib.i in pred_subtree or sib.i >= n:
                    continue
                if sib.dep_ in ("nsubj", "nsubjpass", "nsubj:pass"):
                    subj_role = ("A1" if self.is_es else "ARG1") if is_passive else ("A0" if self.is_es else "ARG0")
                    self._write_span(labels, doc, sib, subj_role, predicate_idx, n)
                    break

        return labels
