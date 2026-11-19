"""Semantic Role Labeling (SRL) benchmark package: Rule-based, CRF, and Neural Transformer+CRF."""

from .data import load_dataset, truncate, get_label_set, dataset_stats
from .metrics import bio_to_spans, compute_metrics, error_analysis, print_results, print_error_analysis

__all__ = [
    "load_dataset",
    "truncate",
    "get_label_set",
    "dataset_stats",
    "bio_to_spans",
    "compute_metrics",
    "error_analysis",
    "print_results",
    "print_error_analysis",
]
