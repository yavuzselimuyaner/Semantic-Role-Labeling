"""Unit & integration tests for SRL v3: unified schema, Span-Biaffine, global decoding, LLM JSON parser."""

import os
import sys
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from srl.llm_srl import build_chat_messages, instance_to_target_json, parse_llm_response
from srl.metrics import bio_to_spans, count_illegal_bio
from srl.schema import (
    get_role_inventory,
    group_by_sentence,
    normalize_bio_label,
    spans_to_bio,
    unify_instances,
)
from srl.span_biaffine import (
    SpanBiaffineSRLNet,
    decode_predicate_spans,
    make_grouped_collate_fn,
)


def test_unified_schema_and_sentence_grouping():
    assert normalize_bio_label("B-A0", "propbank_es") == "B-ARG0"
    assert normalize_bio_label("I-AM-LOC", "propbank_es") == "I-ARGM-LOC"
    assert normalize_bio_label("B-C-A1", "propbank_es") == "B-C-ARG1"
    assert normalize_bio_label("B-AM-PNC", "propbank_es") == "B-ARGM-PRP"
    assert normalize_bio_label("B-V", "propbank_es") == "B-V"

    # Two predicates sharing the exact same sentence -> must collapse into 1 grouped sentence!
    raw_es = [
        {
            "words": ["Juan", "quiere", "comer", "pan"],
            "predicate_idx": 1,
            "labels": ["B-A0", "B-V", "B-A1", "I-A1"],
        },
        {
            "words": ["Juan", "quiere", "comer", "pan"],
            "predicate_idx": 2,
            "labels": ["B-A0", "O", "B-V", "B-A1"],
        },
    ]
    unified = unify_instances(raw_es, language="propbank_es")
    grouped = group_by_sentence(unified)
    assert len(grouped) == 1, f"Expected 1 grouped sentence, got {len(grouped)}"
    assert grouped[0]["predicates"] == [1, 2]
    assert grouped[0]["orig_indices"] == [0, 1]
    assert ("ARG0", 0, 1) in grouped[0]["gold_spans"][0]
    assert ("ARG1", 2, 4) in grouped[0]["gold_spans"][0]
    assert ("ARG1", 3, 4) in grouped[0]["gold_spans"][1]

    inv = get_role_inventory(unified)
    assert inv[0] == "NULL" and "ARG0" in inv and "ARG1" in inv and "V" not in inv
    print("[PASS] test_unified_schema_and_sentence_grouping")


def test_global_unique_core_and_non_overlap_decoding():
    id2role = {0: "NULL", 1: "ARG0", 2: "ARG1", 3: "ARGM-LOC"}
    # 3 candidate spans:
    # k=0: (0,1) -> ARG0 with high margin (+5.0)
    # k=1: (1,2) -> ARG1 overlapping (0,1) -> must be rejected due to overlap!
    # k=2: (3,4) -> second ARG0 (+3.0) -> must be rejected due to unique core role ARG0!
    # k=3: (4,5) -> ARGM-LOC (+2.0) -> accepted!
    cand_starts = [0, 1, 3, 4]
    cand_ends = [1, 2, 4, 5]
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]
    )
    decoded = decode_predicate_spans(logits, cand_starts, cand_ends, pred_pos=2, id2role=id2role, enforce_unique_core=True)
    roles = [r for r, _, _ in decoded]
    assert ("ARG0", 0, 2) in decoded
    assert ("ARGM-LOC", 4, 6) in decoded
    assert roles.count("ARG0") == 1, f"Duplicate core ARG0 allowed: {decoded}"
    assert len(decoded) == 2, f"Overlap or duplicate slipped through: {decoded}"
    bio = spans_to_bio(6, predicate_idx=2, spans=decoded)
    assert count_illegal_bio([bio]) == 0
    print("[PASS] test_global_unique_core_and_non_overlap_decoding")


def test_span_biaffine_forward_backward():
    class TinyCfg:
        hidden_size = 32

    class TinyEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TinyCfg()
            self.emb = nn.Embedding(100, 32)

        def forward(self, input_ids, attention_mask=None):
            class O:
                pass
            o = O()
            o.last_hidden_state = self.emb(input_ids)
            return o

    id2role = {0: "NULL", 1: "ARG0", 2: "ARG1", 3: "ARGM-LOC"}
    net = SpanBiaffineSRLNet(
        model_name="tiny",
        id2role=id2role,
        max_width=10,
        biaffine_dim=24,
        encoder=TinyEnc(),
    )

    collate = make_grouped_collate_fn(pad_token_id=0)
    batch = collate(
        [
            {
                "input_ids": [1, 10, 11, 12, 13, 14, 2],
                "attention_mask": [1] * 7,
                "word_subword_idx": [1, 2, 3, 4, 5],
                "kept_orig_indices": [0, 1, 2, 3, 4],
                "pred_positions": [1, 3],
                "gold_maps": [{(0, 0): 1, (2, 4): 2}, {(0, 0): 1, (4, 4): 2}],
                "orig_indices": [0, 1],
                "orig_len": 5,
            }
        ]
    )

    net.train()
    out = net(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        word_subword_idx=batch["word_subword_idx"],
        word_lengths=batch["word_lengths"],
        pred_positions=batch["pred_positions"],
        gold_maps=batch["gold_maps"],
    )
    loss = out["loss"]
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert net.biaffine_U.grad is not None and torch.isfinite(net.biaffine_U.grad).all()
    assert len(out["outputs"][0]) == 2  # 2 predicates scored in 1 encoder pass!
    print("[PASS] test_span_biaffine_forward_backward")


def test_llm_json_parsing_and_alignment_fallback():
    words = ["The", "chef", "cooked", "the", "meal", "in", "the", "kitchen", "."]
    inst = {
        "words": words,
        "predicate_idx": 2,
        "labels": ["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "I-ARG1", "B-ARGM-LOC", "I-ARGM-LOC", "I-ARGM-LOC", "O"],
    }
    target_json = instance_to_target_json(inst)
    bio, ok = parse_llm_response(target_json, words, predicate_idx=2, align_text_fallback=False)
    assert ok and bio == inst["labels"]

    # Simulate off-by-1 LLM index error with fenced markdown, rescued by text alignment!
    drifted = """```json
    {"arguments": [
        {"role": "ARG0", "start": 0, "end": 2, "text": "The chef"},
        {"role": "ARG1", "start": 4, "end": 6, "text": "the meal"},
        {"role": "ARGM-LOC", "start": 5, "end": 8, "text": "in the kitchen"}
    ]}
    ```"""
    bio_raw, ok_raw = parse_llm_response(drifted, words, predicate_idx=2, align_text_fallback=False)
    bio_aln, ok_aln = parse_llm_response(drifted, words, predicate_idx=2, align_text_fallback=True)
    assert ok_raw and ok_aln
    assert ("ARG1", 3, 5) not in set(bio_to_spans(bio_raw))  # drifted raw index
    assert ("ARG1", 3, 5) in set(bio_to_spans(bio_aln))      # rescued by alignment!

    # Invalid garbage output -> graceful fallback, ok=False
    bad_bio, bad_ok = parse_llm_response("Sorry, I cannot parse this.", words, predicate_idx=2)
    assert not bad_ok and bad_bio[2] == "B-V"

    msgs = build_chat_messages(inst, few_shot_examples=[inst])
    assert len(msgs) == 4  # system + 1*(user+assistant) + query user
    print("[PASS] test_llm_json_parsing_and_alignment_fallback")


if __name__ == "__main__":
    test_unified_schema_and_sentence_grouping()
    test_global_unique_core_and_non_overlap_decoding()
    test_span_biaffine_forward_backward()
    test_llm_json_parsing_and_alignment_fallback()
    print("\nALL 4 SRL v3 RESEARCH TESTS PASSED!")
