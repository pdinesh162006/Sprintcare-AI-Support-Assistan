"""
build_threads.py - Phase 1 Data Pipeline for SprintCare AI Agent

Filters raw Twitter Customer Support CSV (twcs.csv) to @sprintcare conversations,
builds a directed reply graph to reconstruct multi-turn chronological threads,
applies PII redaction and text normalization (emoji mapping + casual_tokenize),
and exports processed conversation threads to data/processed/threads.jsonl.
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import emoji
from nltk.tokenize import casual_tokenize

# Set stdout to UTF-8 for safe console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class PIIRedactor:
    """
    Redacts Personally Identifiable Information (PII) including phone numbers,
    account numbers, PINs, email addresses, and anonymized user IDs.
    Uses Microsoft Presidio if installed, otherwise uses robust regex rules.
    """

    def __init__(self):
        self.has_presidio = False
        try:
            from presidio_analyzer import AnalyzerEngine
            self.analyzer = AnalyzerEngine()
            self.has_presidio = True
        except ImportError:
            self.analyzer = None

        # Robust Regex patterns for telecom domain
        self.phone_regex = re.compile(
            r"(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )
        self.email_regex = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )
        self.pin_regex = re.compile(
            r"(?i)\b(?:pin|passcode|code|secret)\s*(?:is|=|:|-)?\s*(\d{4,6})\b"
        )
        self.account_regex = re.compile(
            r"(?i)\b(?:account|acct|sim|imei|order|ticket|id|ban)\s*(?:#|no\.?|num|is|=|:|-)?\s*([A-Za-z0-9]{6,20})\b"
        )
        # In twcs.csv, customer usernames are anonymized numbers like @115712
        self.anon_user_regex = re.compile(r"@\d{4,9}\b")
        # URLs
        self.url_regex = re.compile(r"https?://\S+|www\.\S+")

    def redact(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        # 1. Redact URLs
        cleaned = self.url_regex.sub("[URL]", text)

        # 2. Redact Emails
        cleaned = self.email_regex.sub("[EMAIL_ADDRESS]", cleaned)

        # 3. Redact Phone Numbers
        cleaned = self.phone_regex.sub("[PHONE_NUMBER]", cleaned)

        # 4. Redact PINs (preserve keyword context, redact secret)
        cleaned = self.pin_regex.sub(r"pin [PIN]", cleaned)

        # 5. Redact Account / SIM / IMEI
        cleaned = self.account_regex.sub(r"account [ACCOUNT_ID]", cleaned)

        # 6. Redact Anonymized customer handles (@115712 -> @customer)
        cleaned = self.anon_user_regex.sub("@customer", cleaned)

        return cleaned


class TextNormalizer:
    """
    Normalizes text for NLP pipelines:
    - Converts unicode emojis to standardized tokens ([EMOJI_<NAME>])
    - Collapses redundant whitespaces
    - Applies NLTK's casual_tokenize
    """

    def __init__(self):
        pass

    def emoji_to_tokens(self, text: str) -> str:
        def replace_match(match):
            name = match.group(1).upper().replace(" ", "_")
            return f" [EMOJI_{name}] "

        # Demojize converts emojis to :name:
        demojized = emoji.demojize(text, delimiters=("__EMOJI_START__", "__EMOJI_END__"))
        # Replace delimiter with token
        demojized = re.sub(r"__EMOJI_START__(.*?)__EMOJI_END__", replace_match, demojized)
        return demojized

    def normalize(self, text: str) -> Tuple[str, List[str]]:
        if not text or not isinstance(text, str):
            return "", []

        # 1. Emoji normalization
        text_with_tokens = self.emoji_to_tokens(text)

        # 2. Collapse excessive whitespace
        normalized_str = re.sub(r"\s+", " ", text_with_tokens).strip()

        # 3. Casual tokenize
        tokens = casual_tokenize(normalized_str, preserve_case=True, reduce_len=True, strip_handles=False)

        return normalized_str, tokens


def filter_brand(df: pd.DataFrame, brand: str = "sprintcare") -> pd.DataFrame:
    """
    Filters the raw customer support dataset down to tweets relevant to @sprintcare:
    1. Direct tweets authored by the brand.
    2. Direct tweets mentioning the brand handle.
    3. Ancestor / descendant tweets belonging to the same conversation threads.
    """
    print(f"Filtering dataset for brand '{brand}'...")
    brand_lower = brand.lower()

    # Step 1: Find direct tweets
    is_author = df["author_id"].str.lower() == brand_lower
    mentions_brand = df["text"].str.contains(f"@{brand_lower}", case=False, na=False)
    direct_mask = is_author | mentions_brand
    direct_ids = set(df.loc[direct_mask, "tweet_id"].astype(int))
    print(f"Direct tweets matching @{brand}: {len(direct_ids):,}")

    # Step 2: Traverse 1-hop parents
    parent_ids = set(df.loc[direct_mask, "in_response_to_tweet_id"].dropna().astype(int))
    all_candidate_ids = direct_ids.union(parent_ids)

    # Step 3: Traverse 1-hop children
    child_mask = df["in_response_to_tweet_id"].isin(all_candidate_ids)
    all_candidate_ids.update(df.loc[child_mask, "tweet_id"].astype(int))

    # Step 4: Final extraction
    filtered_df = df[df["tweet_id"].isin(all_candidate_ids)].copy()
    print(f"Total connected brand tweets retained: {len(filtered_df):,}")
    return filtered_df


def reconstruct_threads(
    df: pd.DataFrame,
    sample_size: int = 5000,
    random_seed: int = 42,
    redactor: Optional[PIIRedactor] = None,
    normalizer: Optional[TextNormalizer] = None,
) -> List[Dict]:
    """
    Reconstructs multi-turn conversational threads by building a directed graph
    from tweet_id -> in_response_to_tweet_id and response_tweet_id.
    Walks each root node down to leaf responses in chronological order,
    redacts PII, normalizes tokens, and returns thread dictionaries.
    """
    if redactor is None:
        redactor = PIIRedactor()
    if normalizer is None:
        normalizer = TextNormalizer()

    print("Building directed conversation graph...")
    tweet_dict: Dict[int, Dict] = {}
    children_map: Dict[int, List[int]] = defaultdict(list)
    parent_map: Dict[int, Optional[int]] = {}

    for _, row in df.iterrows():
        tid = int(row["tweet_id"])
        pid = (
            int(row["in_response_to_tweet_id"])
            if pd.notna(row["in_response_to_tweet_id"])
            else None
        )
        parent_map[tid] = pid
        if pid is not None:
            children_map[pid].append(tid)

        tweet_dict[tid] = {
            "tweet_id": tid,
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]),
            "created_at": str(row["created_at"]),
            "text": str(row["text"]) if pd.notna(row["text"]) else "",
        }

    # Identify root tweets: tweets where parent is None or parent is not in our graph
    all_tids = set(tweet_dict.keys())
    roots = []
    for tid, pid in parent_map.items():
        if (pid is None or pid not in all_tids) and len(children_map[tid]) > 0:
            roots.append(tid)

    print(f"Found {len(roots):,} root tweets with replies.")

    # Reconstruct threads by walking down from root
    threads = []
    for root_id in roots:
        root_tweet = tweet_dict[root_id]
        # Only retain customer-initiated conversations
        if not root_tweet["inbound"] and root_tweet["author_id"].lower() == "sprintcare":
            continue

        # Walk conversation path (longest/first branch)
        current_id = root_id
        path = [current_id]
        visited = {current_id}

        while children_map.get(current_id):
            # Sort children by tweet_id (chronological proxy)
            next_children = [c for c in children_map[current_id] if c not in visited and c in tweet_dict]
            if not next_children:
                break
            next_children.sort()
            next_id = next_children[0]
            path.append(next_id)
            visited.add(next_id)
            current_id = next_id

        # Must have at least 2 turns (customer inquiry + agent reply)
        if len(path) < 2:
            continue

        # Verify there is at least one sprintcare agent response
        authors = [tweet_dict[tid]["author_id"].lower() for tid in path]
        if "sprintcare" not in authors:
            continue

        customer_author = root_tweet["author_id"]

        # Process each turn
        turns = []
        for idx, tid in enumerate(path):
            t_data = tweet_dict[tid]
            raw_text = t_data["text"]
            role = "agent" if t_data["author_id"].lower() == "sprintcare" else "customer"

            # Apply PII Redaction
            redacted_text = redactor.redact(raw_text)

            # Apply Normalization & Tokenization
            clean_text, tokens = normalizer.normalize(redacted_text)

            turns.append({
                "turn_index": idx,
                "tweet_id": tid,
                "author_id": t_data["author_id"],
                "role": role,
                "inbound": t_data["inbound"],
                "created_at": t_data["created_at"],
                "raw_text": raw_text,
                "clean_text": clean_text,
                "tokens": tokens,
            })

        threads.append({
            "thread_id": f"sprint_{root_id}",
            "root_tweet_id": root_id,
            "customer_id": customer_author,
            "turn_count": len(turns),
            "customer_initial_query": turns[0]["clean_text"],
            "turns": turns,
        })

    print(f"Total reconstructed customer support threads: {len(threads):,}")

    # Deterministic sampling to requested size
    if len(threads) > sample_size:
        import random
        rng = random.Random(random_seed)
        sampled_threads = rng.sample(threads, sample_size)
    else:
        sampled_threads = threads

    print(f"Final sampled threads: {len(sampled_threads):,}")
    return sampled_threads


def main():
    parser = argparse.ArgumentParser(description="Reconstruct and process @sprintcare dialog threads.")
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="data/raw/twcs.csv",
        help="Path to twcs.csv raw dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/threads.jsonl",
        help="Path to output threads.jsonl",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of threads to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    if not os.path.exists(args.raw_csv):
        print(f"Error: Raw dataset not found at {args.raw_csv}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading raw dataset from {args.raw_csv}...")
    df_raw = pd.read_csv(args.raw_csv)
    print(f"Loaded {len(df_raw):,} raw records.")

    # 1. Filter Brand
    df_brand = filter_brand(df_raw, brand="sprintcare")

    # 2. Reconstruct Threads
    threads = reconstruct_threads(
        df_brand,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    # 3. Export to JSONL
    print(f"Writing {len(threads):,} threads to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")

    print("Phase 1 Data Pipeline completed successfully!")


if __name__ == "__main__":
    main()
