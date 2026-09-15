# References & Citations: SprintCare AI Support Assistant

The following academic papers, datasets, software libraries, and industry methodologies were utilized in the development and evaluation of this system:

---

## 1. Primary Dataset
- **Customer Support on Twitter Corpus**:
  - Author: Thought Vector (Kaggle Dataset, 2017)
  - URL: [https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  - Identifier: `thoughtvector/customer-support-on-twitter`
  - Description: 2.8+ million tweets and replies between consumers and prominent brands on Twitter, filtered down to 48,970 `@sprintcare` interactions.

---

## 2. Representation & Dense Vector Retrieval
- **BGE Dense Embeddings**:
  - Citation: Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). *C-Pack: Packaged Resources to Advance General Chinese and English Embedding*. arXiv preprint arXiv:2309.07597.
  - Model: `BAAI/bge-small-en-v1.5` (via `fastembed` ONNX execution engine).
- **FAISS (Facebook AI Similarity Search)**:
  - Citation: Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 7(3), 535-547.
  - Implementation: `faiss-cpu>=1.8.0`, utilizing `IndexFlatIP` with L2-normalized cosine similarity.

---

## 3. Sentiment Analysis & NLP Tokenization
- **VADER Sentiment Analysis**:
  - Citation: Hutto, C., & Gilbert, E. (2014). *VADER: A parsimonious rule-based model for sentiment analysis of social media text*. In Proceedings of the International AAAI Conference on Web and Social Media (Vol. 8, No. 1, pp. 216-225).
  - Implementation: `nltk.sentiment.vader.SentimentIntensityAnalyzer`.
- **Casual Tokenization**:
  - Citation: Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python: analyzing text with the natural language toolkit*. O'Reilly Media.
  - Implementation: `nltk.tokenize.casual_tokenize` (optimized for social media text with emoji and punctuation handling).

---

## 4. Evaluation Frameworks & Statistical Metrics
- **G-Eval (LLM-as-a-Judge Multi-Criteria Evaluation)**:
  - Citation: Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). *G-Eval: NLG evaluation using GPT-4 with better human alignment*. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2511–2522.
- **RAGAS (Retrieval Augmented Generation Assessment)**:
  - Citation: Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv preprint arXiv:2309.15217.
  - Metrics adapted: Context Precision@K and Context Recall.
- **Cohen's Kappa & The Kappa Paradox**:
  - Citation: Cohen, J. (1960). *A coefficient of agreement for nominal scales*. Educational and Psychological Measurement, 20(1), 37-46.
  - Citation: Feinstein, A. R., & Cicchetti, D. V. (1990). *High agreement but low kappa: I. The problems of two paradoxes*. Journal of Clinical Epidemiology, 43(6), 543-549.
  - Implementation: `sklearn.metrics.cohen_kappa_score`.

---

## 5. Software Stack & Open Source Libraries
- **Scikit-Learn**: Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
- **FastEmbed**: Qdrant Team (2023). *FastEmbed: Fast and lightweight text embedding library*.
- **Emoji**: Carpedm20 & Kim, T. (2023). *Emoji for Python*.
