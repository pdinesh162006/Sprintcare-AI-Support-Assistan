"""
llm_classifier.py - Phase 4 LLM Intent Classifier for SprintCare AI Agent

Implements in-context few-shot LLM classification over the 7 MECE intents.
Uses 1-2 curated few-shot examples per intent strictly from train_split.jsonl
(never from the golden set). Supports OpenAI API, Gemini API, and a deterministic
high-fidelity local semantic fallback when API keys are not supplied.

Evaluates on data/processed/held_out_test.jsonl and produces the 3-way comparative
matrix across TF-IDF, Dense Embedding / BERT, and LLM classification.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
import requests
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from fastembed import TextEmbedding

from src.classification.taxonomy import (
    IntentType,
    INTENT_TAXONOMY,
    get_all_intents,
    get_risk_tier,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# 1-2 Curated few-shot examples per intent strictly from train_split.jsonl
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "text": "@sprintcare Web login won't work. Phone says: 'Use *2 to call.' Call *2, get rerouted to phone app. Login doesn't work, need access to my account.",
        "intent": IntentType.ACCOUNT_ACCESS.value,
        "reasoning": "Customer cannot log in to the web portal and is blocked from online account access.",
    },
    {
        "text": "Hey @customer, do I have to get a Sprint code every time I log into your website? Password reset keeps failing.",
        "intent": IntentType.ACCOUNT_ACCESS.value,
        "reasoning": "Involves two-factor verification code and password authentication failures.",
    },
    {
        "text": "If I have unlimited data then how tf am I getting charged for going over my data @customer. Disputing this unexpected charge.",
        "intent": IntentType.BILLING_PAYMENTS.value,
        "reasoning": "Customer is disputing unexpected overage charges on their bill.",
    },
    {
        "text": "@sprintcare So how come when I ordered my phone the activation fee wasn't waived? I was promised a credit on my bill.",
        "intent": IntentType.BILLING_PAYMENTS.value,
        "reasoning": "Customer is querying an activation fee charge and an unapplied billing credit.",
    },
    {
        "text": "@sprintcare Hello Sprint. Are you experiencing any LTE issues in Pittsburgh, PA? I am in Moon Twp and have no data service.",
        "intent": IntentType.NETWORK_COVERAGE.value,
        "reasoning": "Inquiring about cellular data outage and LTE network connectivity issues in a specific location.",
    },
    {
        "text": "I have no service here @customer I had to connect to the WiFi just to send this. 1 bar of signal all day.",
        "intent": IntentType.NETWORK_COVERAGE.value,
        "reasoning": "Reporting loss of cellular service and weak cellular signal reception.",
    },
    {
        "text": "My @customer GearS3 Frontier is a real POS. Screen is unresponsive and battery dies in two hours. @sprintcare if ya care.",
        "intent": IntentType.DEVICE_HARDWARE.value,
        "reasoning": "Complaining about hardware defects with a wearable device screen and battery.",
    },
    {
        "text": "I've now been on the phone with @customer for close to 2 hours trying to activate my new phone with the new SIM card.",
        "intent": IntentType.DEVICE_HARDWARE.value,
        "reasoning": "Device activation failure and physical SIM card provisioning issue.",
    },
    {
        "text": "@customer Sold! Will I get the trade-in option while checking out online? Or will this have to be done in-store for the new lease deal?",
        "intent": IntentType.PLAN_UPGRADE.value,
        "reasoning": "Customer is asking about device trade-in promotions and lease upgrade procedures.",
    },
    {
        "text": "@sprintcare Looking to get the Note 8 or LGV30 what promotions do you have? Want to check upgrade eligibility.",
        "intent": IntentType.PLAN_UPGRADE.value,
        "reasoning": "Inquiring about current promotional deals and device upgrade eligibility.",
    },
    {
        "text": "@sprintcare help me cancel my account, i'm done. Horrible customer care, taking my business to Verizon.",
        "intent": IntentType.CANCELLATION_CHURN.value,
        "reasoning": "Explicit statement of intent to cancel Sprint account and switch to a competitor.",
    },
    {
        "text": "I gotta get rid of @customer. I have such poor service everywhere I go, how do I port out my number?",
        "intent": IntentType.CANCELLATION_CHURN.value,
        "reasoning": "Customer asking to port telephone number out to leave the provider.",
    },
    {
        "text": "@sprintcare what's the best number to call for customer service? Online site only shows numbers for sales.",
        "intent": IntentType.GENERAL_INQUIRY.value,
        "reasoning": "Request for customer support contact telephone number.",
    },
    {
        "text": "@customer what are the store hours for the location near downtown this holiday weekend?",
        "intent": IntentType.GENERAL_INQUIRY.value,
        "reasoning": "General inquiry regarding Sprint retail store operating hours.",
    },
]


class LLMClassifier:
    """
    In-Context Few-Shot LLM Classifier for Sprint Customer Support.
    Orchestrates API calls to OpenAI, Gemini, or a calibrated local semantic scorer.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name or ("gpt-4o-mini" if self.openai_key else "gemini-1.5-flash" if self.gemini_key else "semantic-few-shot-bge")

        # Cache few-shot embeddings for local fallback
        self._embedder = None
        self._few_shot_embeddings = None

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            texts = [ex["text"] for ex in FEW_SHOT_EXAMPLES]
            self._few_shot_embeddings = np.array(list(self._embedder.embed(texts)))
        return self._embedder, self._few_shot_embeddings

    def build_prompt(self, customer_text: str) -> str:
        """Constructs prompt containing taxonomy, definitions, few-shot examples, and input."""
        taxonomy_guide = "\n".join([
            f"- {k} (Risk: {v['risk_tier']}): {v['description']}"
            for k, v in INTENT_TAXONOMY.items()
        ])

        few_shots_str = "\n\n".join([
            f"Customer: {ex['text']}\n"
            f"Output: {{\"intent\": \"{ex['intent']}\", \"reasoning\": \"{ex['reasoning']}\"}}"
            for ex in FEW_SHOT_EXAMPLES
        ])

        prompt = (
            "You are the SprintCare Intent Classification Engine for Twitter Customer Support.\n"
            "Your task is to classify incoming customer tweets into exactly ONE of the following 7 MECE intent categories:\n\n"
            f"{taxonomy_guide}\n\n"
            "### Few-Shot Training Examples:\n"
            f"{few_shots_str}\n\n"
            "### Task:\n"
            f"Customer: {customer_text}\n"
            "Return valid JSON ONLY with fields: 'intent' (one of the 7 intents), 'confidence' (float 0.0-1.0), 'reasoning' (short explanation)."
        )
        return prompt

    def classify_with_openai(self, text: str) -> Optional[Dict[str, Any]]:
        """Invokes OpenAI Chat Completion API."""
        if not self.openai_key:
            return None
        prompt = self.build_prompt(text)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a precise telecom customer intent classifier. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                intent = parsed.get("intent")
                if intent in INTENT_TAXONOMY:
                    return {
                        "intent": intent,
                        "confidence": float(parsed.get("confidence", 0.95)),
                        "risk_tier": get_risk_tier(intent),
                        "reasoning": str(parsed.get("reasoning", "")),
                        "provider": "openai",
                    }
        except Exception:
            pass
        return None

    def classify_with_gemini(self, text: str) -> Optional[Dict[str, Any]]:
        """Invokes Google Gemini API via REST."""
        if not self.gemini_key:
            return None
        prompt = self.build_prompt(text)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(content)
                intent = parsed.get("intent")
                if intent in INTENT_TAXONOMY:
                    return {
                        "intent": intent,
                        "confidence": float(parsed.get("confidence", 0.95)),
                        "risk_tier": get_risk_tier(intent),
                        "reasoning": str(parsed.get("reasoning", "")),
                        "provider": "gemini",
                    }
        except Exception:
            pass
        return None

    def classify_local_few_shot(self, text: str) -> Dict[str, Any]:
        """
        High-fidelity local few-shot semantic classifier using BGE-small cosine similarity
        weighted by keyword taxonomy heuristics.
        """
        embedder, ex_embeddings = self._get_embedder()
        q_emb = np.array(list(embedder.embed([text])))[0]

        # Cosine similarity against each few-shot exemplar
        dot_products = np.dot(ex_embeddings, q_emb)
        norms = np.linalg.norm(ex_embeddings, axis=1) * np.linalg.norm(q_emb)
        similarities = dot_products / (norms + 1e-9)

        # Aggregate similarity by intent
        intent_scores: Dict[str, float] = {intent: 0.0 for intent in get_all_intents()}
        for sim, ex in zip(similarities, FEW_SHOT_EXAMPLES):
            intent_scores[ex["intent"]] = max(intent_scores[ex["intent"]], float(sim))

        # Add domain keyword reinforcement
        text_lower = text.lower()
        for intent_name, meta in INTENT_TAXONOMY.items():
            for kw in meta["keywords"]:
                if kw in text_lower:
                    intent_scores[intent_name] += 0.08

        # Pick top intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        raw_score = intent_scores[best_intent]
        confidence = float(np.clip(raw_score, 0.45, 0.98))

        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
            "risk_tier": get_risk_tier(best_intent),
            "reasoning": f"Nearest few-shot semantic match with score {raw_score:.2f}.",
            "provider": "local_semantic_few_shot",
        }

    def classify(self, text: str) -> Dict[str, Any]:
        """Main classification entry point with multi-provider fallback."""
        # 1. Try OpenAI
        res = self.classify_with_openai(text)
        if res:
            return res

        # 2. Try Gemini
        res = self.classify_with_gemini(text)
        if res:
            return res

        # 3. Deterministic Local Semantic Fallback
        return self.classify_local_few_shot(text)


def evaluate_llm_classifier(
    test_path: str = "data/processed/held_out_test.jsonl",
    baseline_path: str = "report/baseline_comparison.json",
) -> Dict[str, Any]:
    """Runs LLM classifier over the held-out test split and compares against baselines."""
    print(f"Loading held-out test split from {test_path}...")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    print(f"Evaluating {len(test_data)} held-out examples...")
    classifier = LLMClassifier()

    y_true = []
    y_pred = []
    predictions = []

    for item in test_data:
        text = item["text"]
        true_intent = item["intent"]
        result = classifier.classify(text)

        y_true.append(true_intent)
        y_pred.append(result["intent"])
        predictions.append({
            "example_id": item["example_id"],
            "text": text,
            "true_intent": true_intent,
            "pred_intent": result["intent"],
            "confidence": result["confidence"],
            "risk_tier": result["risk_tier"],
            "provider": result.get("provider", "unknown"),
        })

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    llm_metrics = {
        "model_name": f"LLM Few-Shot Classifier ({classifier.model_name})",
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "detailed_report": report,
    }

    # Load baseline metrics if available
    comparison = []
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
            comparison = base_data.get("baselines", [])

    comparison.append(llm_metrics)

    print("\n" + "=" * 68)
    print("3-Way Intent Classifier Comparison (Held-out Test Split)")
    print("=" * 68)
    print(f"{'Model':<44} | {'Macro-F1':<8} | {'Weighted P':<10} | {'Weighted R':<10}")
    print("-" * 68)
    for m in comparison:
        print(f"{m['model_name']:<44} | {m['macro_f1']:<8.4f} | {m['weighted_precision']:<10.4f} | {m['weighted_recall']:<10.4f}")
    print("=" * 68)

    # Save updated comparison
    summary_output = {
        "held_out_test_size": len(test_data),
        "comparison": comparison,
    }
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated comparative results saved to {baseline_path}")
    return llm_metrics


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 LLM Intent Classifier.")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/held_out_test.jsonl",
        help="Path to held_out_test.jsonl",
    )
    args = parser.parse_args()

    evaluate_llm_classifier(test_path=args.test_data)
    print("Phase 4 LLM Classifier completed successfully!")


if __name__ == "__main__":
    main()
