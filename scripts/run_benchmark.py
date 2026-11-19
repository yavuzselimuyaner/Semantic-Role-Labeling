"""CLI runner for SRL v2 benchmarks (Rule-based, CRF, Predicate-Aware Transformer+CRF)."""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import spacy

# Allow running directly from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from srl.crf_model import CRFModel, parse_aligned_docs
from srl.data import dataset_stats, get_label_set, load_dataset, truncate
from srl.metrics import compute_metrics, error_analysis, print_error_analysis, print_results
from srl.rule_based import RuleBasedSRL
from srl.transformer_model import RECOMMENDED_MODELS, TransformerSRLModel


def load_spacy_model(language: str, prefer_trf: bool = True):
    candidates = (
        ["es_dep_news_trf", "es_core_news_lg", "es_core_news_md", "es_core_news_sm"]
        if language == "propbank_es"
        else ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
    )
    if not prefer_trf:
        candidates = [c for c in candidates if not c.endswith("_trf")] + [
            c for c in candidates if c.endswith("_trf")
        ]
    for name in candidates:
        try:
            nlp = spacy.load(name, disable=["ner"])
            print(f"  Loaded spaCy model: {name}")
            return nlp
        except OSError:
            continue
    raise RuntimeError(
        f"No spaCy model found for {language}. Install one of: {', '.join(candidates)}"
    )


def run_dataset(
    dataset_name: str,
    data_path: str,
    methods: List[str],
    out_dir: str,
    max_instances: int = None,
    model_name: str = None,
    epochs: int = 8,
    batch_size: int = 32,
    use_crf_head: bool = True,
    prefer_trf_spacy: bool = True,
) -> Dict[str, dict]:
    print(f"\n{'=' * 70}\n  DATASET: {dataset_name.upper()} ({data_path})\n{'=' * 70}")
    train, dev, test = load_dataset(data_path)
    if max_instances:
        train = truncate(train, max_instances)
        dev = truncate(dev, max(max_instances // 5, 100))
        test = truncate(test, max(max_instances // 5, 100))

    stats = dataset_stats(test, name=f"{dataset_name}_test")
    print(
        f"  Splits: Train={len(train):,}  Dev={len(dev):,}  Test={len(test):,}\n"
        f"  Test stats: spans={stats['num_spans']:,}, avg_span_len={stats['avg_span_len']}, "
        f"single_tok_ratio={stats['single_token_span_ratio']:.2%}, "
        f"max_sent_len={stats['max_sent_len']}"
    )

    gold_test = [inst["labels"] for inst in test]
    label_set = get_label_set(train, dev, test)
    print(f"  Label inventory size: {len(label_set)}")

    results: Dict[str, dict] = {}
    nlp = None
    test_docs = None

    if "rule" in methods or "crf" in methods:
        nlp = load_spacy_model(dataset_name, prefer_trf=prefer_trf_spacy)
        print("  Parsing aligned test docs once for Rule-based / CRF...")
        test_docs = parse_aligned_docs(test, nlp)

    if "rule" in methods:
        t0 = time.time()
        rb = RuleBasedSRL(nlp=nlp, mode=dataset_name)
        preds = rb.predict_batch(test, docs=test_docs)
        dt = round(time.time() - t0, 2)
        m = compute_metrics(gold_test, preds)
        ea = error_analysis(gold_test, preds)
        print_results(f"Rule-Based v2 [{dataset_name}]", m)
        print_error_analysis(f"Rule-Based v2 [{dataset_name}]", ea)
        results["rule_based_v2"] = {**m, "error_analysis": ea, "time_s": dt}

    if "crf" in methods:
        t0 = time.time()
        print("  Parsing aligned train docs for CRF...")
        train_docs = parse_aligned_docs(train, nlp)
        crf = CRFModel(max_iterations=200, c1=0.1, c2=0.1, min_freq=3, all_possible_transitions=False)
        crf.fit(train, nlp, docs=train_docs)
        preds = crf.predict(test, nlp, docs=test_docs)
        dt = round(time.time() - t0, 2)
        m = compute_metrics(gold_test, preds)
        ea = error_analysis(gold_test, preds)
        print_results(f"CRF v2 [{dataset_name}]", m)
        print_error_analysis(f"CRF v2 [{dataset_name}]", ea)
        results["crf_v2"] = {**m, "error_analysis": ea, "time_s": dt}

    if "transformer" in methods:
        t0 = time.time()
        chosen_model = model_name or RECOMMENDED_MODELS.get(dataset_name, "bert-base-cased")
        ckpt_path = os.path.join(
            out_dir, f"{dataset_name}_{chosen_model.replace('/', '_')}_crf{int(use_crf_head)}.pt"
        )
        trainer = TransformerSRLModel(
            label_set=label_set,
            model_name=chosen_model,
            language=dataset_name,
            max_length=256,
            batch_size=batch_size,
            epochs=epochs,
            use_crf=use_crf_head,
            enforce_bio=True,
        )
        history = trainer.fit(train, dev_instances=dev, save_path=ckpt_path)
        preds = trainer.predict(test)
        dt = round(time.time() - t0, 2)
        m = compute_metrics(gold_test, preds)
        ea = error_analysis(gold_test, preds)
        tag = f"transformer_{chosen_model.split('/')[-1]}_crf{int(use_crf_head)}"
        print_results(f"{tag} [{dataset_name}]", m)
        print_error_analysis(f"{tag} [{dataset_name}]", ea)
        results[tag] = {
            **m,
            "model_name": chosen_model,
            "use_crf": use_crf_head,
            "history": history,
            "error_analysis": ea,
            "time_s": dt,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run SRL v2 benchmarks.")
    parser.add_argument("--dataset", choices=["propbank", "propbank_es"], required=True)
    parser.add_argument("--data-path", required=True, help="Path to preprocessed JSON.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["rule", "crf", "transformer"],
        choices=["rule", "crf", "transformer"],
    )
    parser.add_argument("--model-name", default=None, help="HuggingFace backbone override.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-crf-head", action="store_true", help="Disable CRF head (ablation).")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--cpu-spacy", action="store_true", help="Prefer fast non-trf spaCy model.")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "SRL_data", "results_v2"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    res = run_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        methods=args.methods,
        out_dir=args.out_dir,
        max_instances=args.max_instances,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_crf_head=not args.no_crf_head,
        prefer_trf_spacy=not args.cpu_spacy,
    )

    json_out = os.path.join(args.out_dir, f"{args.dataset}_results_v2.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    csv_out = os.path.join(args.out_dir, f"{args.dataset}_summary_v2.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "method", "precision", "recall", "f1", "illegal_bio", "time_s"]
        )
        writer.writeheader()
        for method, r in res.items():
            writer.writerow(
                {
                    "dataset": args.dataset,
                    "method": method,
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "illegal_bio": r.get("illegal_bio", 0),
                    "time_s": r.get("time_s", 0),
                }
            )
    print(f"\nSaved JSON -> {json_out}\nSaved CSV  -> {csv_out}")


if __name__ == "__main__":
    main()
