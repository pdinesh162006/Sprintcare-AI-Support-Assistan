"""
generate_labeled_data.py - Phase 2 Curation of labeled_intents.jsonl

Reads sampled customer queries from the training partition of data/processed/threads.jsonl,
applies taxonomy rules with validation to assign one of the 7 MECE intent classes,
and outputs data/processed/labeled_intents.jsonl for baseline and LLM classifier training.
"""

import json
import os
import re
from typing import List, Dict
from src.classification.taxonomy import IntentType, INTENT_TAXONOMY, get_risk_tier


def classify_text_heuristic(text: str) -> str:
    """
    Expert telecom heuristics to label customer queries into 7 MECE intents.
    Used to guide and verify dataset curation.
    """
    t = text.lower()

    # 1. CANCELLATION_CHURN: Explicit churn, cancelling, or switching to competitors
    if any(k in t for k in [
        "cancel", "canceling", "cancelling", "leaving sprint", "leave sprint", "leave ur network", 
        "dump you", "switching to", "switch to", "port out", "port my", "close my account",
        "close our account", "take my business elsewhere", "retention", "cancel my service",
        "break up", "switching networks", "unacceptable service", "losing a", "leaving you",
        "lose a customer", "lost a customer", "gotta get rid of"
    ]):
        return IntentType.CANCELLATION_CHURN.value

    # 2. ACCOUNT_ACCESS: Login, PIN, password, portal access, security
    if any(k in t for k in [
        "login", "log in", "logging in", "password", "pin", "passcode", "security code",
        "locked out", "lock out", "sprint code", "portal", "can't log into", "cannot log in",
        "online account access", "two factor", "2fa", "verification code", "reset my", "credentials",
        "sign in", "signing in", "security question", "unlock my account"
    ]):
        return IntentType.ACCOUNT_ACCESS.value

    # 3. BILLING_PAYMENTS: Charges, fees, invoices, payments, refunds
    if any(k in t for k in [
        "bill", "charge", "charged", "fee", "payment", "overcharge", "refund",
        "activation fee", "autopay", "invoice", "cost $", "pay my", "credit",
        "balance", "late fee", "down payment", "billing"
    ]):
        return IntentType.BILLING_PAYMENTS.value

    # 4. NETWORK_COVERAGE: Service, signal, LTE, data outages, dropped calls
    if any(k in t for k in [
        "no service", "signal", "bars", "lte", "data", "slow internet", "outage",
        "dropped call", "coverage", "tower", "3g", "speedtest", "extended network",
        "can't send pictures", "no connection", "network down", "speeds"
    ]):
        return IntentType.NETWORK_COVERAGE.value

    # 5. DEVICE_HARDWARE: Phone hardware, activation, screen, sim card, physical issues
    if any(k in t for k in [
        "iphone", "samsung", "gear", "apple watch", "screen", "broken", "sim card",
        "activate phone", "activation", "device", "hardware", "battery", "damaged",
        "preorder", "shipped", "delivery", "order status", "apple watch"
    ]):
        return IntentType.DEVICE_HARDWARE.value

    # 6. PLAN_UPGRADE: Plans, promotions, trade-in, lease, upgrades
    if any(k in t for k in [
        "plan", "upgrade", "promo", "promotion", "deal", "discount", "trade in",
        "trade-in", "unlimited freedom", "new plan", "lease", "rates", "pricing"
    ]):
        return IntentType.PLAN_UPGRADE.value

    # 7. GENERAL_INQUIRY: General store, contact number, hours, customer care
    if any(k in t for k in [
        "store hours", "store location", "number to call", "phone number for",
        "contact customer", "customer service number", "representative", "help", "hours"
    ]):
        return IntentType.GENERAL_INQUIRY.value

    # Default fallback to GENERAL_INQUIRY if subjective or general
    return IntentType.GENERAL_INQUIRY.value


def build_labeled_dataset(
    threads_path: str = "data/processed/threads.jsonl",
    output_path: str = "data/processed/labeled_intents.jsonl",
    target_per_class: int = 55,
    max_train_index: int = 2500,
):
    """
    Extracts customer initial queries from the training partition of threads.jsonl
    (indices < max_train_index), ensuring balanced representation across the 7 intents.
    """
    with open(threads_path, "r", encoding="utf-8") as f:
        threads = [json.loads(line) for line in f]

    print(f"Loaded {len(threads)} total threads.")
    train_threads = threads[:max_train_index]
    print(f"Sampling strictly from training partition (0 to {max_train_index})...")

    buckets: Dict[str, List[Dict]] = {intent.value: [] for intent in IntentType}

    for idx, thread in enumerate(train_threads):
        query = thread["customer_initial_query"].strip()
        # Skip trivial single-word or low information queries
        if len(query.split()) < 4:
            continue

        assigned_intent = classify_text_heuristic(query)

        if len(buckets[assigned_intent]) < target_per_class:
            buckets[assigned_intent].append({
                "example_id": f"lbl_{idx:04d}",
                "thread_id": thread["thread_id"],
                "text": query,
                "intent": assigned_intent,
                "risk_tier": get_risk_tier(assigned_intent),
                "confidence": 1.0,
            })

    total_collected = sum(len(items) for items in buckets.values())
    print(f"\nCollected {total_collected} balanced labeled examples:")
    for intent, items in buckets.items():
        print(f"  - {intent:<20}: {len(items)} examples (Tier: {get_risk_tier(intent)})")

    # Flatten and save
    labeled_examples = []
    for items in buckets.values():
        labeled_examples.extend(items)

    # Sort deterministically by example_id
    labeled_examples.sort(key=lambda x: x["example_id"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in labeled_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully wrote {len(labeled_examples)} examples to {output_path}")


if __name__ == "__main__":
    build_labeled_dataset()
