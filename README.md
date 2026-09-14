# SprintCare AI Support Assistant

A production-grade conversational AI customer service agent for Sprint (`@sprintcare`) built using Kaggle's Twitter Customer Support dataset (`thoughtvector/customer-support-on-twitter`).

## Architecture & Roadmap
- **Phase 0 — Setup**: Project structure, pinned dependencies, dataset placement (`twcs.csv`).
- **Phase 1 — Data pipeline**: Brand filtering (`@sprintcare`), directed reply graph reconstruction, PII masking, casual tokenization.
- **Phase 2 — Intent taxonomy**: 7 MECE intent classes, risk-tiering, hand-labeled baseline dataset.
- **Phase 3 — Baseline classifiers**: TF-IDF + SVM, Dense Embedding / BERT classification.
- **Phase 4 — LLM classifier**: Few-shot in-context learning with structured JSON schema.
- **Phase 5 — RAG index**: Resolution-filtered FAISS vector store over historical successful resolutions.
- **Phase 6 — Grounded generation**: Twitter character limit (<280 chars), factual grounding, empathy tone.
- **Phase 7 — Handler contract / escalation**: Risk-tiered confidence thresholds, slot-filling, sentiment-drift detection.
- **Phase 8 — Golden evaluation set**: Stratified buckets (production, adversarial, edge-case, historical failure).
- **Phase 9 — Evaluation harness**: Faithfulness, G-Eval judge LLM, Cohen's Kappa human agreement.
- **Phase 10 — Report and decision log**: Production trade-offs, failure modes, headline metric caveats.
- **Phase 11 — Reproducibility**: End-to-end quickstart script.

## Setup & Quickstart
```bash
pip install -r requirements.txt
```
