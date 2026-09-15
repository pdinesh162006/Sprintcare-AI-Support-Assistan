"""
generate_replies.py - Phase 6 Grounded Reply Generation for SprintCare AI Agent

Retrieves relevant historical resolution context from the FAISS vector store
and generates grounded, empathetic, Twitter-compliant customer support replies.

Operational Constraints:
1. Under 280 characters (strict Twitter limit).
2. Factual grounding: Grounded strictly in retrieved historical resolutions.
3. Appropriate tone: Reassuring, empathetic, and professional with agent sign-off.
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, List, Any, Optional
import requests
from dotenv import load_dotenv
load_dotenv()

from src.rag.build_index import RAGRetriever

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class GroundedReplyGenerator:
    """
    Grounded conversational generation engine for @sprintcare on Twitter.
    """

    def __init__(
        self,
        vector_store_dir: str = "data/processed/vector_store",
        model_name: Optional[str] = None,
    ):
        self.retriever = RAGRetriever(vector_store_dir=vector_store_dir)
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.model_name = model_name or (
            "gemini-1.5-flash" if self.gemini_key else
            "groq-llama-3.1-8b" if self.groq_key else
            "gpt-4o-mini" if self.openai_key else
            "grounded-local-synthesizer"
        )

    def format_retrieved_context(self, hits: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into grounding context."""
        context_blocks = []
        for i, h in enumerate(hits, 1):
            context_blocks.append(
                f"[Resolution {i}] (Similarity: {h.get('similarity_score', 0.0):.3f})\n"
                f"Historical Issue: {h['customer_issue']}\n"
                f"SprintCare Solution: {h['agent_resolution']}"
            )
        return "\n\n".join(context_blocks)

    def build_prompt(self, customer_query: str, customer_handle: str, context_str: str) -> str:
        """Constructs prompt with constraints and grounding context."""
        prompt = (
            "You are an official SprintCare Customer Support agent on Twitter (@sprintcare).\n"
            "Generate a tweet reply addressing the customer's issue based ONLY on the provided historical resolutions.\n\n"
            "### Strict Operational Rules:\n"
            "1. Character Limit: The entire response MUST be under 280 characters (including handle and sign-off).\n"
            f"2. Mention Handle: Start your response with '{customer_handle}'.\n"
            "3. Grounding: Rely ONLY on the facts, guidance, or actions shown in the historical resolutions. Do NOT invent policies or phone numbers.\n"
            "4. Tone: Empathetic, polite, concise, professional.\n"
            "5. Escalation: If account details or a DM is required according to context, invite them to send a Direct Message with their phone number and PIN.\n"
            "6. Sign-off: End with '^Care' or '^AI'.\n\n"
            f"### Historical Resolved Context:\n{context_str}\n\n"
            f"### Customer Tweet:\n{customer_query}\n\n"
            "Output the exact reply text only:"
        )
        return prompt

    def generate_with_openai(self, prompt: str) -> Optional[str]:
        """Calls OpenAI Chat Completions API."""
        if not self.openai_key:
            return None
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a customer support agent. Obey the 280 character limit strictly."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 120,
            "temperature": 0.3,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"].strip()
                return reply
        except Exception:
            pass
        return None

    def generate_with_gemini(self, prompt: str) -> Optional[str]:
        """Calls Google Gemini API."""
        if not self.gemini_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 120},
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                reply = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                return reply
        except Exception:
            pass
        return None

    def generate_with_groq(self, prompt: str) -> Optional[str]:
        """Calls Groq Cloud API with Llama-3.1."""
        if not self.groq_key:
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a customer support agent. Obey the 280 character limit strictly."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 120,
            "temperature": 0.3,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"].strip()
                return reply
        except Exception:
            pass
        return None

    def generate_local_grounded(
        self, customer_query: str, customer_handle: str, hits: List[Dict[str, Any]]
    ) -> str:
        """
        Deterministic local grounded synthesizer: adapts the highest-scoring historical
        resolution to the current customer query while strictly enforcing Twitter conventions.
        """
        if not hits:
            return f"{customer_handle} We want to help! Please send us a DM with your phone number and ZIP code so we can look into this. ^Care"

        top_hit = hits[0]
        historical_solution = top_hit["agent_resolution"]

        # Clean up historical agent solution: remove historical agent sign-offs and old handles
        clean_sol = re.sub(r"@\w+\s*", "", historical_solution)
        clean_sol = re.sub(r"-[A-Z]{2,3}$|\^[A-Za-z]+$", "", clean_sol).strip()

        # Format into concise, grounded response
        reply = f"{customer_handle} {clean_sol}"
        if not reply.endswith((".", "!", "?")):
            reply += "."

        # Append sign-off
        reply = f"{reply} ^Care"

        # Character budget guardrail: trim if exceeds 280 chars
        if len(reply) > 280:
            truncated = reply[:240].rsplit(" ", 1)[0]
            reply = f"{customer_handle} {truncated}... Please DM us for help. ^Care"

        return reply

    def generate_reply(
        self,
        customer_query: str,
        customer_handle: str = "@customer",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieves relevant historical resolutions,
        generates grounded reply, and validates constraints.
        """
        # 1. Retrieve historical resolutions
        hits = self.retriever.retrieve(customer_query, top_k=top_k)
        context_str = self.format_retrieved_context(hits)

        # 2. Build prompt
        prompt = self.build_prompt(customer_query, customer_handle, context_str)

        # 3. Generate via multi-provider (Gemini -> Groq -> OpenAI -> Local Grounded Synthesizer)
        raw_reply = None
        provider = "grounded_synthesizer"

        if self.gemini_key:
            raw_reply = self.generate_with_gemini(prompt)
            if raw_reply:
                provider = "gemini-1.5-flash"

        if not raw_reply and self.groq_key:
            raw_reply = self.generate_with_groq(prompt)
            if raw_reply:
                provider = "groq-llama-3.1"

        if not raw_reply and self.openai_key:
            raw_reply = self.generate_with_openai(prompt)
            if raw_reply:
                provider = "openai"

        if not raw_reply:
            raw_reply = self.generate_local_grounded(customer_query, customer_handle, hits)
            provider = "grounded_synthesizer"

        # 4. Clean and enforce <= 280 chars
        reply = raw_reply.strip().replace("\n", " ")
        if not reply.startswith(customer_handle):
            reply = f"{customer_handle} {reply}"

        if len(reply) > 280:
            reply = reply[:277] + "..."

        char_count = len(reply)
        is_under_280 = char_count <= 280

        # Assess grounding score from retrieval similarity
        top_sim = hits[0]["similarity_score"] if hits else 0.0

        return {
            "query": customer_query,
            "handle": customer_handle,
            "reply": reply,
            "char_count": char_count,
            "under_280": is_under_280,
            "grounding_similarity": top_sim,
            "provider": provider,
            "retrieved_hits": hits,
        }


def run_test_queries():
    """Evaluates grounded generation across representative customer care scenarios."""
    generator = GroundedReplyGenerator()

    test_cases = [
        {
            "intent": "NETWORK_COVERAGE",
            "query": "I have zero bars and no LTE data in downtown Dallas today, what's going on?",
        },
        {
            "intent": "BILLING_PAYMENTS",
            "query": "Why was I charged an extra $30 activation fee on my latest bill when I was told it's waived?",
        },
        {
            "intent": "DEVICE_HARDWARE",
            "query": "My new phone won't activate with the SIM card I received in the mail, keeps showing error.",
        },
        {
            "intent": "PLAN_UPGRADE",
            "query": "What are the latest trade-in promotions and lease deals for the iPhone X?",
        },
        {
            "intent": "ACCOUNT_ACCESS",
            "query": "I am locked out of my online account and not getting the security verification code.",
        },
        {
            "intent": "CANCELLATION_CHURN",
            "query": "I'm sick of this terrible network, how do I cancel my service and switch to Verizon?",
        },
        {
            "intent": "GENERAL_INQUIRY",
            "query": "What are your customer service chat hours and contact phone number?",
        },
    ]

    print("\n" + "=" * 70)
    print("Evaluating Grounded Generation across @sprintcare Test Queries")
    print("=" * 70)

    results = []
    all_under_280 = True

    for idx, tc in enumerate(test_cases, 1):
        output = generator.generate_reply(tc["query"])
        results.append(output)

        print(f"\n--- [Test {idx}] Intent: {tc['intent']} ---")
        print(f"Customer Query: \"{tc['query']}\"")
        print(f"Top Hit Sim   : {output['grounding_similarity']:.4f} (Thread: {output['retrieved_hits'][0]['thread_id'] if output['retrieved_hits'] else 'N/A'})")
        print(f"Generated Reply: \"{output['reply']}\"")
        print(f"Char Count    : {output['char_count']} / 280 (Valid: {output['under_280']})")
        print(f"Provider      : {output['provider']}")

        if not output["under_280"]:
            all_under_280 = False

    print("\n" + "=" * 70)
    print(f"Summary: All {len(test_cases)} test replies under 280 characters: {all_under_280}")
    print("Phase 6 Grounded Generation completed successfully!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate grounded customer care replies.")
    parser.add_argument("--test", action="store_true", default=True, help="Run test suite across intents")
    args = parser.parse_args()

    if args.test:
        run_test_queries()


if __name__ == "__main__":
    main()
