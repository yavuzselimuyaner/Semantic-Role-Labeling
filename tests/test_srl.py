"""Unit & integration tests for SRL v2 components (runs fast on CPU without internet)."""

import os
import sys
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from srl.crf_model import CRFModel, get_lca_path, sent2features
from srl.decoding import ConstrainedLinearCRF, build_bio_constraint_masks, viterbi_decode_bio
from srl.metrics import bio_to_spans, compute_metrics, count_illegal_bio, error_analysis
from srl.rule_based import RuleBasedSRL
from srl.transformer_model import PredicateConditionedSRLNet, make_collate_fn


# --- Lightweight mock spaCy Doc / Token for deterministic syntax tests ---
class MockToken:
    def __init__(self, i, text, dep, pos="NOUN", tag="NN", lemma=None, head_i=None):
        self.i = i
        self.text = text
        self.lower_ = text.lower()
        self.dep_ = dep
        self.pos_ = pos
        self.tag_ = tag
        self.lemma_ = lemma or text.lower()
        self.is_punct = text in ".,;:!?"
        self._head_i = i if head_i is None else head_i
        self.head = self
        self.children = []
        self.subtree = [self]


class MockDoc(list):
    def __init__(self, tokens):
        super().__init__(tokens)
        for t in self:
            t.head = self[t._head_i]
        for t in self:
            if t.head is not t:
                t.head.children.append(t)
        # Build subtrees recursively
        def collect(tok):
            res = [tok]
            for c in tok.children:
                res.extend(collect(c))
            return sorted(res, key=lambda x: x.i)

        for t in self:
            t.subtree = collect(t)


def test_rule_based_english_prefixes_and_modals():
    # "The chef will quickly cook the meal in the kitchen ."
    #  0   1    2    3       4    5   6    7  8   9       10
    toks = [
        MockToken(0, "The", "det", "DET", "DT", head_i=1),
        MockToken(1, "chef", "nsubj", "NOUN", "NN", head_i=4),
        MockToken(2, "will", "aux", "AUX", "MD", lemma="will", head_i=4),
        MockToken(3, "quickly", "advmod", "ADV", "RB", head_i=4),
        MockToken(4, "cook", "ROOT", "VERB", "VB", head_i=4),
        MockToken(5, "the", "det", "DET", "DT", head_i=6),
        MockToken(6, "meal", "dobj", "NOUN", "NN", head_i=4),
        MockToken(7, "in", "prep", "ADP", "IN", head_i=4),
        MockToken(8, "the", "det", "DET", "DT", head_i=9),
        MockToken(9, "kitchen", "pobj", "NOUN", "NN", head_i=7),
        MockToken(10, ".", "punct", "PUNCT", ".", head_i=4),
    ]
    doc = MockDoc(toks)
    rb = RuleBasedSRL(nlp=None, mode="propbank")
    preds = rb._predict_from_doc(doc, predicate_idx=4, n=len(toks))
    spans = set(bio_to_spans(preds))

    assert ("ARG0", 0, 2) in spans, f"Missing ARG0: {spans}"
    assert ("ARGM-MOD", 2, 3) in spans, f"Missing ARGM-MOD: {spans}"
    assert ("ARGM-MNR", 3, 4) in spans, f"Missing ARGM-MNR: {spans}"
    assert ("V", 4, 5) in spans, f"Missing V: {spans}"
    assert ("ARG1", 5, 7) in spans, f"Missing ARG1: {spans}"
    # Crucial bug fix check: must be ARGM-LOC, NOT bare LOC!
    assert ("ARGM-LOC", 7, 10) in spans, f"Missing ARGM-LOC (prefix bug!): {spans}"
    assert count_illegal_bio([preds]) == 0
    print("[PASS] test_rule_based_english_prefixes_and_modals")


def test_rule_based_spanish_obl_case_traversal():
    # "El chef cocinó la comida en la cocina ."
    #  0  1    2      3  4      5  6  7      8
    # UD Spanish: cocina (obl) -> en (case), la (det)
    toks = [
        MockToken(0, "El", "det", "DET", "DET", head_i=1),
        MockToken(1, "chef", "nsubj", "NOUN", "NOUN", head_i=2),
        MockToken(2, "cocinó", "ROOT", "VERB", "VERB", lemma="cocinar", head_i=2),
        MockToken(3, "la", "det", "DET", "DET", head_i=4),
        MockToken(4, "comida", "obj", "NOUN", "NOUN", head_i=2),
        MockToken(5, "en", "case", "ADP", "ADP", head_i=7),
        MockToken(6, "la", "det", "DET", "DET", head_i=7),
        MockToken(7, "cocina", "obl", "NOUN", "NOUN", head_i=2),
        MockToken(8, ".", "punct", "PUNCT", "PUNCT", head_i=2),
    ]
    doc = MockDoc(toks)
    rb = RuleBasedSRL(nlp=None, mode="propbank_es")
    preds = rb._predict_from_doc(doc, predicate_idx=2, n=len(toks))
    spans = set(bio_to_spans(preds))

    assert ("A0", 0, 2) in spans, f"Missing A0: {spans}"
    assert ("A1", 3, 5) in spans, f"Missing A1: {spans}"
    # Crucial Spanish UD bug fix: obl -> case('en') must emit AM-LOC!
    assert ("AM-LOC", 5, 8) in spans, f"Missing AM-LOC from Spanish obl+case: {spans}"
    assert count_illegal_bio([preds]) == 0
    print("[PASS] test_rule_based_spanish_obl_case_traversal")


def test_crf_lca_path_and_features():
    # "John wants to eat pizza"
    #  0    1     2  3   4
    toks = [
        MockToken(0, "John", "nsubj", "PROPN", "NNP", head_i=1),
        MockToken(1, "wants", "ROOT", "VERB", "VBZ", lemma="want", head_i=1),
        MockToken(2, "to", "aux", "PART", "TO", head_i=3),
        MockToken(3, "eat", "xcomp", "VERB", "VB", lemma="eat", head_i=1),
        MockToken(4, "pizza", "dobj", "NOUN", "NN", head_i=3),
    ]
    doc = MockDoc(toks)
    # Direct child of predicate 'eat' (idx 3): 'pizza' (idx 4) -> must NOT be "none"!
    path_obj, dist_obj = get_lca_path(4, 3, doc)
    assert path_obj == "^dobj" and dist_obj == 1, f"Got {path_obj}, {dist_obj}"
    # Shared matrix subject 'John' (idx 0) to 'eat' (idx 3) via LCA 'wants' (idx 1)
    path_subj, dist_subj = get_lca_path(0, 3, doc)
    assert path_subj == "^nsubjvxcomp" and dist_subj == 2, f"Got {path_subj}, {dist_subj}"

    feats = sent2features([t.text for t in toks], pred_idx=3, doc=doc)
    assert len(feats) == len(toks)
    assert feats[4]["is_pred_child"] is True
    assert feats[4]["top_arg_dep"] == "dobj"
    assert feats[4]["is_sub_start"] is True
    print("[PASS] test_crf_lca_path_and_features")


def test_bio_constrained_crf_and_viterbi():
    label_set = ["O", "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1", "B-V"]
    id2label = {i: l for i, l in enumerate(label_set)}
    start_ok, trans_ok = build_bio_constraint_masks(id2label)

    # Sequence cannot start with I-ARG0 (idx 2) or I-ARG1 (idx 4)
    assert not start_ok[2] and not start_ok[4]
    # O (0) -> I-ARG0 (2) is forbidden
    assert not trans_ok[0, 2]
    # B-ARG0 (1) -> I-ARG1 (4) is forbidden
    assert not trans_ok[1, 4]
    # B-ARG0 (1) -> I-ARG0 (2) is allowed
    assert trans_ok[1, 2]

    # Construct adversarial emissions that would cause argmax to output illegal [I-ARG0, I-ARG1, O]
    emissions = torch.tensor(
        [
            [
                [0.0, 1.0, 9.0, 0.0, 0.0, 0.0],  # argmax = I-ARG0 (illegal at pos 0!)
                [0.0, 0.0, 1.5, 0.5, 9.0, 0.0],  # argmax = I-ARG1 (illegal after ARG0!)
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax = O
            ]
        ]
    )
    mask = torch.tensor([[True, True, True]])
    decoded = viterbi_decode_bio(emissions, mask, id2label)
    pred_labels = [[id2label[i] for i in decoded[0]]]
    assert count_illegal_bio(pred_labels) == 0, f"Illegal BIO decoded: {pred_labels}"

    # Verify CRF NLL loss is finite, positive, and backpropagates cleanly
    crf = ConstrainedLinearCRF(id2label, enforce_bio=True)
    emissions_req = emissions.clone().requires_grad_(True)
    tags = torch.tensor([[1, 2, 0]])
    loss = crf(emissions_req, tags, mask)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert emissions_req.grad is not None and torch.isfinite(emissions_req.grad).all()

    # Verify noisy gold BIO (O -> I-ARG0) does NOT cause a -1e4 loss spike during training
    noisy_tags = torch.tensor([[0, 2, 0]])
    noisy_loss = crf(emissions, noisy_tags, mask)
    assert noisy_loss.item() < 100.0, f"Noisy gold BIO exploded loss: {noisy_loss.item()}"
    print("[PASS] test_bio_constrained_crf_and_viterbi")


def test_predicate_conditioned_net_forward_backward_decode():
    class TinyConfig:
        hidden_size = 32

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TinyConfig()
            self.emb = nn.Embedding(100, 32)

        def forward(self, input_ids, attention_mask=None):
            class Out:
                pass
            o = Out()
            o.last_hidden_state = self.emb(input_ids)
            return o

    label_set = ["O", "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1", "B-V"]
    id2label = {i: l for i, l in enumerate(label_set)}
    net = PredicateConditionedSRLNet(
        model_name="tiny",
        id2label=id2label,
        vocab_size=100,
        use_crf=True,
        enforce_bio=True,
        encoder=TinyEncoder(),
    )

    collate = make_collate_fn(pad_token_id=0)
    batch = collate(
        [
            {
                "input_ids": [1, 10, 11, 2, 12, 3, 13, 4],
                "attention_mask": [1] * 8,
                "word_subword_idx": [1, 2, 4, 6],
                "word_labels": [1, 2, 5, 3],
                "rel_pos": [0, 0, 1, 2],
                "pred_word_pos": 2,
                "orig_len": 4,
            },
            {
                "input_ids": [1, 20, 2, 21, 3, 4],
                "attention_mask": [1] * 6,
                "word_subword_idx": [1, 3],
                "word_labels": [1, 5],
                "rel_pos": [0, 1],
                "pred_word_pos": 1,
                "orig_len": 2,
            },
        ]
    )

    out = net(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        word_subword_idx=batch["word_subword_idx"],
        rel_pos=batch["rel_pos"],
        pred_word_pos=batch["pred_word_pos"],
        word_mask=batch["word_mask"],
        word_labels=batch["word_labels"],
    )
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()

    decoded = net.decode(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        word_subword_idx=batch["word_subword_idx"],
        rel_pos=batch["rel_pos"],
        pred_word_pos=batch["pred_word_pos"],
        word_mask=batch["word_mask"],
    )
    assert len(decoded) == 2 and len(decoded[0]) == 4 and len(decoded[1]) == 2
    decoded_lbls = [[id2label[i] for i in seq] for seq in decoded]
    assert count_illegal_bio(decoded_lbls) == 0
    print("[PASS] test_predicate_conditioned_net_forward_backward_decode")


def test_error_analysis_decomposition():
    gold = [["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "I-ARG1", "B-ARGM-LOC", "I-ARGM-LOC"]]
    # ARG0 exact, ARG1 boundary short by 1, ARGM-LOC confused as ARGM-ADV, plus spurious ARGM-TMP? No, clean test:
    pred = [["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "O", "B-ARGM-ADV", "I-ARGM-ADV"]]
    ea = error_analysis(gold, pred)
    assert ea["exact_match"] == 1
    assert ea["boundary_only"] == 1
    assert ea["role_confusion"] == 1
    assert ea["top_confusions"][0] == (("ARGM-LOC", "ARGM-ADV"), 1)
    print("[PASS] test_error_analysis_decomposition")


if __name__ == "__main__":
    test_rule_based_english_prefixes_and_modals()
    test_rule_based_spanish_obl_case_traversal()
    test_crf_lca_path_and_features()
    test_bio_constrained_crf_and_viterbi()
    test_predicate_conditioned_net_forward_backward_decode()
    test_error_analysis_decomposition()
    print("\nALL 6 SRL v2 TESTS PASSED!")
