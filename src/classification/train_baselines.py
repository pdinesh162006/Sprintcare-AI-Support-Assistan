"""
train_baselines.py - Phase 3 Baseline Classifiers for SprintCare AI Agent

Trains two classical NLP / dense embedding baseline classifiers:
1. TF-IDF + LinearSVC (calibrated with CalibratedClassifierCV)
2. Dense Embedding / BERT baseline (fastembed BGE-small + Logistic Regression)

Splits labeled_intents.jsonl into stratified train and held-out test splits,
and reports Macro-F1, Weighted Precision, and Weighted Recall (not raw accuracy).
Saves held_out_test.jsonl for downstream LLM classifier benchmarking.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from fastembed import TextEmbedding

from src.classification.taxonomy import get_all_intents

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_labeled_data(path: str) -> Tuple[List[str], List[str], List[Dict]]:
    """Loads labeled examples from JSONL."""
    texts = []
    labels = []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["intent"])
            records.append(item)
    return texts, labels, records


def create_stratified_split(
    records: List[Dict],
    test_size: float = 0.20,
    random_state: int = 42,
    train_out_path: str = "data/processed/train_split.jsonl",
    test_out_path: str = "data/processed/held_out_test.jsonl",
) -> Tuple[List[Dict], List[Dict]]:
    """Creates a deterministic stratified train/test split and persists held-out test split."""
    y = [r["intent"] for r in records]
    train_records, test_records = train_test_split(
        records, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
    with open(train_out_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_out_path, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Stratified split complete: {len(train_records)} train, {len(test_records)} held-out test.")
    print(f"Held-out test split saved to: {test_out_path}")
    return train_records, test_records


def train_tfidf_svm(
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
) -> Tuple[Any, Dict[str, Any]]:
    """Trains TF-IDF + Calibrated LinearSVC classifier."""
    print("\n" + "=" * 50)
    print("Training Baseline 1: TF-IDF + Calibrated LinearSVC")
    print("=" * 50)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_svc = LinearSVC(C=1.0, random_state=42, dual="auto")
    clf = CalibratedClassifierCV(estimator=base_svc, cv=3)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_p:.4f}")
    print(f"Weighted Recall   : {weighted_r:.4f}")

    metrics = {
        "model_name": "TF-IDF + LinearSVC",
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "detailed_report": report,
    }

    pipeline = {
        "vectorizer": vectorizer,
        "classifier": clf,
    }
    return pipeline, metrics


def run_bert_baseline(
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Implements Dense Embedding / BERT baseline:
    Uses BGE-small-en dense sentence embeddings + Logistic Regression classifier.
    """
    print("\n" + "=" * 50)
    print("Training Baseline 2: Dense Embedding / BERT + Logistic Regression")
    print("=" * 50)

    print("Generating dense embeddings with fastembed (BAAI/bge-small-en-v1.5)...")
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    X_train_emb = np.array(list(embedder.embed(X_train)))
    X_test_emb = np.array(list(embedder.embed(X_test)))

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_p:.4f}")
    print(f"Weighted Recall   : {weighted_r:.4f}")

    metrics = {
        "model_name": "Dense Embedding / BERT + LogisticRegression",
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "detailed_report": report,
    }

    pipeline = {
        "embedder_name": "BAAI/bge-small-en-v1.5",
        "classifier": clf,
    }
    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline intent classifiers.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/labeled_intents.jsonl",
        help="Path to labeled_intents.jsonl",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/processed/models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="report",
        help="Directory to save baseline comparison report",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        sys.exit(1)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print(f"Loading data from {args.data}...")
    texts, labels, records = load_labeled_data(args.data)
    print(f"Loaded {len(records)} examples across {len(set(labels))} classes.")

    train_records, test_records = create_stratified_split(records)

    X_train = [r["text"] for r in train_records]
    y_train = [r["intent"] for r in train_records]
    X_test = [r["text"] for r in test_records]
    y_test = [r["intent"] for r in test_records]

    # Baseline 1: TF-IDF + SVM
    tfidf_pipeline, tfidf_metrics = train_tfidf_svm(X_train, y_train, X_test, y_test)
    joblib.dump(tfidf_pipeline, os.path.join(args.models_dir, "tfidf_svm_model.joblib"))

    # Baseline 2: Dense Embedding / BERT + Logistic Regression
    bert_pipeline, bert_metrics = run_bert_baseline(X_train, y_train, X_test, y_test)
    joblib.dump(bert_pipeline, os.path.join(args.models_dir, "bert_lr_model.joblib"))

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Model':<42} | {'Macro-F1':<8} | {'Weighted P':<10} | {'Weighted R':<10}")
    print("-" * 65)
    print(f"{tfidf_metrics['model_name']:<42} | {tfidf_metrics['macro_f1']:<8.4f} | {tfidf_metrics['weighted_precision']:<10.4f} | {tfidf_metrics['weighted_recall']:<10.4f}")
    print(f"{bert_metrics['model_name']:<42} | {bert_metrics['macro_f1']:<8.4f} | {bert_metrics['weighted_precision']:<10.4f} | {bert_metrics['weighted_recall']:<10.4f}")
    print("=" * 65)

    # Save metrics
    output_summary = {
        "dataset_size": len(records),
        "train_size": len(train_records),
        "held_out_test_size": len(test_records),
        "baselines": [tfidf_metrics, bert_metrics],
    }

    summary_path = os.path.join(args.report_dir, "baseline_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, ensure_ascii=False)

    print(f"\nBaseline metrics saved to {summary_path}")
    print("Phase 3 Baseline Classifiers completed successfully!")


if __name__ == "__main__":
    main()
