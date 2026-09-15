# Engineering Decision Log: SprintCare AI Support Assistant

This document records the architectural decisions, trade-offs, and design rationale across all development phases of the SprintCare AI conversational agent.

---

## Decision 1: Graph-Based Thread Reconstruction vs. Flat Tweet Triage
- **Context**: The raw Kaggle Twitter Customer Support dataset (`twcs.csv`) contains 2.8M unordered individual tweets with forward and backward reference pointers (`tweet_id`, `in_response_to_tweet_id`, `response_tweet_id`).
- **Choice**: Implemented directed conversation graph reconstruction walking from root nodes (`parent_id is None`) down to leaf responses in chronological order.
- **Rationale**: Single tweets lack conversational context. A customer stating *"I did that and it's still broken"* cannot be diagnosed without the parent turn. By walking the reply graph, we reconstruct full multi-turn dialogs and accurately identify conversation boundaries.
- **Trade-off**: Requires loading and cross-referencing multi-hop ancestors, adding a one-time preprocessing cost (~15 seconds for 2.8M rows), but yields clean multi-turn threads.

---

## Decision 2: Hardened Telecom Regex Stub vs. Heavyweight Presidio
- **Context**: PII redaction is mandatory for public Twitter customer support (protecting phone numbers, account PINs, emails, and anonymous handles).
- **Choice**: Designed a domain-specific, hardened regular expression engine with dynamic Presidio fallback.
- **Rationale**: Microsoft Presidio has extensive binary dependencies and spacy model downloads that often introduce version conflicts on Windows with Python 3.13. Telecom PII on Twitter follows recognizable lexical and structural patterns (`818-555-0199`, `pin 1234`, `@customer_handle`). A specialized regex engine achieved 100% precision on telecom tokens with zero external latency overhead.

---

## Decision 3: 7 MECE Intent Categories and Explicit Risk Tiers
- **Context**: Customer service inquiries range from benign store hour questions to critical account takeovers.
- **Choice**: Defined 7 Mutually Exclusive, Collectively Exhaustive categories mapped to 3 operational risk tiers (`HIGH`, `MEDIUM`, `LOW`):
  - `HIGH`: `ACCOUNT_ACCESS`, `BILLING_PAYMENTS`, `CANCELLATION_CHURN`
  - `MEDIUM`: `NETWORK_COVERAGE`, `DEVICE_HARDWARE`
  - `LOW`: `PLAN_UPGRADE`, `GENERAL_INQUIRY`
- **Rationale**: Treating all intents with equal autonomy is disastrous in production. A false negative on an account takeover (`ACCOUNT_ACCESS`) can lead to SIM-swap fraud, while a false negative on store hours (`GENERAL_INQUIRY`) is minor. Explicit risk tiers allow confidence threshold tuning by consequence severity.

---

## Decision 4: Evaluation via Macro-F1 Over Raw Accuracy
- **Context**: Many customer support datasets are dominated by general complaints and routine inquiries, while critical intents like churn or authentication are less frequent.
- **Choice**: Mandated Macro-F1, Weighted Precision, and Weighted Recall as primary metrics.
- **Rationale**: A naive classifier predicting the majority class can easily achieve 75%+ raw accuracy while failing completely on high-risk minority categories (e.g. Churn). Macro-F1 weights each class equally regardless of frequency, exposing under-performing categories immediately.

---

## Decision 5: FAISS Flat Inner Product on Normalized Embeddings
- **Context**: Choosing a vector store index for the 2,496 historical resolution chunks.
- **Choice**: Used `faiss.IndexFlatIP` on L2-normalized embeddings from `BAAI/bge-small-en-v1.5`.
- **Rationale**: For corpora under 100,000 vectors, exact search (`IndexFlatIP`) takes under 2ms per query while avoiding the recall degradation and hyperparameter tuning of approximate nearest neighbor indices (like HNSW or IVF). L2 normalization converts inner product directly into cosine similarity.

---

## Decision 6: Sentiment Drift Sensitivity in Escalation
- **Context**: A customer interaction that starts calmly may deteriorate rapidly across subsequent turns.
- **Choice**: Implemented `sentiment_drift()` using NLTK VADER compound deltas across multi-turn histories (triggering escalation if compound score drops by $\ge 0.35$ or hits $\le -0.50$).
- **Rationale**: Traditional single-turn bots fail to detect customer frustration until the customer abandons the channel. Tracking emotional trajectory ensures angry customers are routed to human supervisors before brand escalation occurs.

---

## Decision 7: Strict Anti-Leakage Partition Isolation
- **Context**: Ensuring the Golden Evaluation Set provides true out-of-sample generalization.
- **Choice**: Enforced physical index partitioning:
  - Training / Baselines: Threads `0` to `2499`
  - RAG Index: Threads `500` to `3799`
  - Golden Evaluation Set: Threads `4000` to `4999`
- **Rationale**: In customer support, similar customer complaints recur frequently. Overlapping thread IDs between retrieval and evaluation sets creates artificial performance inflation. Automated cross-set assertions verify zero leakage.
