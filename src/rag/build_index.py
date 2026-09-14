"""
build_index.py - Phase 5 RAG Vector Store & Positive Resolution Filtering

Implements filter_positive_resolution() using VADER sentiment scoring and resolution
satisfaction markers on historical customer turns.
Constructs semantic issue-resolution chunks from successfully resolved @sprintcare threads,
generates normalized dense vector embeddings (BAAI/bge-small-en-v1.5), and builds a FAISS
IndexFlatIP vector store saved to data/processed/vector_store/.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import faiss
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fastembed import TextEmbedding

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class PositiveResolutionFilter:
    """
    Evaluates whether a customer service thread concluded with a positive resolution:
    - Analyzes final customer turn(s) with VADER compound sentiment scoring
    - Checks for explicit satisfaction cues ('thank you', 'appreciate', 'fixed', 'works now')
    - Filters out unresolved frustration, sarcastic rebuttals, or churn statements.
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.positive_cues = {
            "thank", "thanks", "appreciate", "helpful", "awesome", "great", "perfect",
            "works now", "it worked", "fixed", "good job", "much better", "resolved",
            "solved", "glad to hear", "all set", "good to know"
        }
        self.negative_cues = {
            "no thanks", "still not", "terrible", "useless", "waste", "cancel", "worst",
            "hate", "switching", "ridiculous", "not working", "fail", "bullshit", "horrible"
        }

    def score_turn(self, text: str) -> float:
        """Returns VADER compound score between -1.0 and 1.0."""
        return float(self.sia.polarity_scores(text)["compound"])

    def is_positive_resolution(self, thread: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Determines if thread represents a successful positive resolution.
        Returns (is_resolved, sentiment_score, reason).
        """
        turns = thread.get("turns", [])
        if len(turns) < 2:
            return False, 0.0, "insufficient_turns"

        # Separate customer and agent turns
        customer_turns = [t for t in turns if t["role"] == "customer"]
        agent_turns = [t for t in turns if t["role"] == "agent"]

        if not agent_turns:
            return False, 0.0, "no_agent_response"

        # Case 1: Multi-turn interaction with customer follow-up
        if len(customer_turns) > 1:
            last_cust_turn = customer_turns[-1]["clean_text"].lower()
            compound = self.score_turn(last_cust_turn)

            # Check for negative disqualifiers first
            if any(neg in last_cust_turn for neg in self.negative_cues):
                return False, compound, "negative_followup"

            # Check for explicit satisfaction cues
            if any(pos in last_cust_turn for pos in self.positive_cues):
                return True, compound, "explicit_satisfaction"

            # Positive sentiment threshold
            if compound >= 0.20:
                return True, compound, "positive_sentiment"

            return False, compound, "neutral_or_ambiguous_resolution"

        # Case 2: Clean 2-turn Q&A (Customer Query -> Agent Resolution)
        agent_resp = agent_turns[0]["clean_text"]
        # Retain informative resolutions with actionable guidance (URL, steps, or policy)
        if len(agent_resp.split()) >= 8 and not any(neg in agent_resp.lower() for neg in ["sorry, we cannot", "outage"]):
            return True, 0.15, "clean_uncontested_resolution"

        return False, 0.0, "unverified"


def build_semantic_chunks(threads: List[Dict[str, Any]], res_filter: PositiveResolutionFilter) -> List[Dict[str, Any]]:
    """
    Transforms resolved threads into semantic issue-resolution chunks
    optimized for RAG retrieval.
    """
    chunks = []
    for t in threads:
        is_pos, score, reason = res_filter.is_positive_resolution(t)
        if not is_pos:
            continue

        turns = t.get("turns", [])
        customer_issue = turns[0]["clean_text"]
        
        # Find first comprehensive agent response
        agent_turns = [t["clean_text"] for t in turns if t["role"] == "agent"]
        if not agent_turns:
            continue
        agent_resolution = agent_turns[0]

        # Contextual chunk representation
        chunk_text = (
            f"Customer Inquiry: {customer_issue}\n"
            f"SprintCare Resolution: {agent_resolution}"
        )

        chunk_id = f"rag_{t['thread_id']}"
        chunks.append({
            "chunk_id": chunk_id,
            "thread_id": t["thread_id"],
            "customer_issue": customer_issue,
            "agent_resolution": agent_resolution,
            "chunk_text": chunk_text,
            "resolution_score": round(score, 3),
            "resolution_reason": reason,
            "turn_count": len(turns),
        })

    return chunks


class RAGRetriever:
    """
    FAISS-backed Dense Vector Retriever for historical grounded resolutions.
    """

    def __init__(self, vector_store_dir: str = "data/processed/vector_store"):
        self.index_path = os.path.join(vector_store_dir, "index.faiss")
        self.meta_path = os.path.join(vector_store_dir, "metadata.jsonl")

        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Vector store not found at {vector_store_dir}")

        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        print(f"Index loaded. Total vectors: {self.index.ntotal}")

        self.metadata = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant historical resolution chunks for a customer query."""
        q_emb = np.array(list(self.embedder.embed([query])), dtype=np.float32)
        faiss.normalize_L2(q_emb)

        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["similarity_score"] = round(float(dist), 4)
            results.append(item)

        return results


def build_vector_store(
    threads_path: str = "data/processed/threads.jsonl",
    output_dir: str = "data/processed/vector_store",
    start_index: int = 500,
    end_index: int = 3800,
):
    """
    Builds the complete FAISS vector store over positively resolved historical threads
    within the historical partition [start_index, end_index], strictly isolating the
    evaluation golden set partition (>= 4000).
    """
    print(f"Loading conversation threads from {threads_path}...")
    with open(threads_path, "r", encoding="utf-8") as f:
        all_threads = [json.loads(line) for line in f]

    historical_threads = all_threads[start_index:end_index]
    print(f"Filtering positive resolutions from partition [{start_index}:{end_index}] (count: {len(historical_threads)})...")

    res_filter = PositiveResolutionFilter()
    chunks = build_semantic_chunks(historical_threads, res_filter)
    print(f"Extracted {len(chunks)} positively resolved semantic resolution chunks.")

    if not chunks:
        print("Error: No valid chunks found.")
        sys.exit(1)

    print("Generating dense vector embeddings using BAAI/bge-small-en-v1.5...")
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    chunk_texts = [c["chunk_text"] for c in chunks]
    embeddings = np.array(list(embedder.embed(chunk_texts)), dtype=np.float32)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape} (Dimension: {dim})")

    # Build FAISS Flat Inner Product index
    print("Building FAISS IndexFlatIP (cosine similarity)...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} vectors in FAISS store.")

    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, "index.faiss")
    meta_file = os.path.join(output_dir, "metadata.jsonl")

    faiss.write_index(index, index_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Saved FAISS index to {index_file}")
    print(f"Saved metadata to {meta_file}")

    # Test sample retrieval
    print("\n" + "=" * 50)
    print("Verifying RAG Retrieval with Sample Query:")
    test_query = "My LTE data is not working in Chicago, having no service"
    print(f"Query: '{test_query}'")
    retriever = RAGRetriever(vector_store_dir=output_dir)
    hits = retriever.retrieve(test_query, top_k=2)
    for i, h in enumerate(hits, 1):
        print(f"\n[Hit {i}] Score: {h['similarity_score']:.4f} | Thread: {h['thread_id']}")
        print(f"Customer Issue : {h['customer_issue'][:100]}...")
        print(f"Sprint Response: {h['agent_resolution'][:100]}...")
    print("=" * 50)
    print("Phase 5 RAG Index completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Build RAG Index over resolved historical threads.")
    parser.add_argument(
        "--threads",
        type=str,
        default="data/processed/threads.jsonl",
        help="Path to threads.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/vector_store",
        help="Directory to save index.faiss and metadata.jsonl",
    )
    args = parser.parse_args()

    build_vector_store(threads_path=args.threads, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
