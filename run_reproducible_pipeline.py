"""
run_reproducible_pipeline.py - Master Reproducibility Runner for SprintCare AI Agent

Executes all project phases end-to-end in under 15 minutes:
1. Environment & Dependency Validation
2. Phase 1: Data Pipeline & PII Redaction
3. Phase 2: Intent Taxonomy Curation
4. Phase 3: Baseline Classifier Training (TF-IDF + SVM & Dense BGE + LR)
5. Phase 4: LLM Few-Shot Classifier Evaluation
6. Phase 5: RAG Vector Store & Positive Resolution Indexing
7. Phase 6: Grounded Twitter-Compliant Reply Generation
8. Phase 7: Handler Contract & Escalation Policy Verification
9. Phase 8: Golden Evaluation Set Curation & Anti-Leakage Audit
10. Phase 9: End-to-End Evaluation Harness (Faithfulness, RAGAS, G-Eval, Kappa)

Usage:
    python run_reproducible_pipeline.py [--rebuild-threads]
"""

import os
import sys
import time
import argparse
import subprocess
from typing import List, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_step(step_number: int, step_title: str, module_cmd: List[str]):
    """Executes a pipeline step and measures execution duration."""
    print("\n" + "=" * 75)
    print(f"STEP {step_number}: {step_title}")
    print("=" * 75)
    start_time = time.time()

    cmd = [sys.executable, "-m"] + module_cmd
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_number} failed with exit code {result.returncode} after {elapsed:.2f}s")
        sys.exit(result.returncode)

    print(f"[SUCCESS] Step {step_number} completed in {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run SprintCare AI Agent pipeline end-to-end.")
    parser.add_argument(
        "--rebuild-threads",
        action="store_true",
        help="Force re-extraction of 5,000 threads from raw twcs.csv",
    )
    args = parser.parse_args()

    pipeline_start = time.time()
    print("*" * 75)
    print("STARTING END-TO-END REPRODUCIBILITY PASS: SPRINTCARE AI AGENT")
    print(f"Python: {sys.version.split()[0]} on {sys.platform}")
    print("*" * 75)

    step_timings = {}

    # Step 1: Data Pipeline (Skip rebuild if processed threads already exist, unless requested)
    if args.rebuild_threads or not os.path.exists("data/processed/threads.jsonl"):
        t = run_step(1, "Data Pipeline (Brand Filtering, Graph Walk, PII Redaction)", [
            "src.pipeline.build_threads",
            "--raw-csv", "data/raw/twcs.csv",
            "--output", "data/processed/threads.jsonl",
            "--sample-size", "5000",
        ])
        step_timings["Phase 1: Data Pipeline"] = t
    else:
        print("\n[INFO] data/processed/threads.jsonl already exists. Skipping raw CSV re-extraction (use --rebuild-threads to force).")
        step_timings["Phase 1: Data Pipeline"] = 0.0

    # Step 2: Intent Taxonomy Curation
    step_timings["Phase 2: Intent Taxonomy"] = run_step(
        2, "Intent Taxonomy Curation (labeled_intents.jsonl)",
        ["src.classification.generate_labeled_data"]
    )

    # Step 3: Train Baseline Classifiers
    step_timings["Phase 3: Baseline Classifiers"] = run_step(
        3, "Train Baseline Classifiers (TF-IDF+SVM and Dense BGE+LR)",
        ["src.classification.train_baselines"]
    )

    # Step 4: LLM Few-Shot Classifier
    step_timings["Phase 4: LLM Classifier"] = run_step(
        4, "LLM Classifier Evaluation (Held-Out Test Split)",
        ["src.classification.llm_classifier"]
    )

    # Step 5: Build RAG Index
    step_timings["Phase 5: RAG Vector Store"] = run_step(
        5, "Build FAISS Vector Store (Positive Resolution Filtering)",
        ["src.rag.build_index"]
    )

    # Step 6: Grounded Generation
    step_timings["Phase 6: Grounded Generation"] = run_step(
        6, "Test Grounded Generation Suite across Customer Queries",
        ["src.rag.generate_replies"]
    )

    # Step 7: Escalation Handler Contract
    step_timings["Phase 7: Handler Contract"] = run_step(
        7, "Verify Safety Thresholds & Escalation Policies",
        ["src.escalation.handler_contract"]
    )

    # Step 8: Golden Evaluation Set Curation & Anti-Leakage Audit
    step_timings["Phase 8: Golden Set Curation"] = run_step(
        8, "Curate Golden Evaluation Set & Run Anti-Leakage Audit",
        ["eval.golden_set.build_golden_set"]
    )

    # Step 9: Evaluation Harness
    step_timings["Phase 9: Evaluation Harness"] = run_step(
        9, "Run Evaluation Harness (Faithfulness, RAGAS, G-Eval, Kappa)",
        ["eval.run_harness"]
    )

    total_time = time.time() - pipeline_start

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION SUMMARY & TIMINGS")
    print("=" * 75)
    for name, dur in step_timings.items():
        print(f"{name:<45} : {dur:>7.2f}s")
    print("-" * 75)
    print(f"{'TOTAL PIPELINE EXECUTION TIME':<45} : {total_time:>7.2f}s ({total_time/60:.2f} mins)")
    print("=" * 75)
    print(f"Under 15-Minute Requirement: {'PASSED' if total_time < 900 else 'FAILED'}")
    print("ALL PHASES REPRODUCED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
