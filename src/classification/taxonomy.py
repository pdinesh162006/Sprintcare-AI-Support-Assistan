"""
taxonomy.py - Intent Taxonomy and Risk Tiers for SprintCare AI Agent

Defines the 7 MECE (Mutually Exclusive, Collectively Exhaustive) customer intent
categories for Sprint customer care on Twitter, mapped to operational risk tiers.
"""

from enum import Enum
from typing import Dict, List, Any


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IntentType(str, Enum):
    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"
    BILLING_PAYMENTS = "BILLING_PAYMENTS"
    NETWORK_COVERAGE = "NETWORK_COVERAGE"
    DEVICE_HARDWARE = "DEVICE_HARDWARE"
    PLAN_UPGRADE = "PLAN_UPGRADE"
    CANCELLATION_CHURN = "CANCELLATION_CHURN"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"


INTENT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    IntentType.ACCOUNT_ACCESS.value: {
        "intent": IntentType.ACCOUNT_ACCESS.value,
        "display_name": "Account Access & Security",
        "description": (
            "Issues with Sprint account credentials, website login, two-factor authentication, "
            "security PIN reset, account lockouts, or unauthorized account modifications."
        ),
        "risk_tier": RiskTier.HIGH.value,
        "escalation_priority": "P1",
        "keywords": ["login", "password", "pin", "code", "lock", "portal", "website", "verification", "access", "account", "security"],
        "required_slots": ["phone_number", "account_pin"],
    },
    IntentType.BILLING_PAYMENTS.value: {
        "intent": IntentType.BILLING_PAYMENTS.value,
        "display_name": "Billing, Fees & Payments",
        "description": (
            "Disputed charges, unexpected fees (e.g. activation fee, roaming charge, protection fee), "
            "payment arrangements, invoice clarification, autopay issues, or refund requests."
        ),
        "risk_tier": RiskTier.HIGH.value,
        "escalation_priority": "P1",
        "keywords": ["bill", "charge", "fee", "payment", "overcharge", "refund", "credit", "activation fee", "cost", "invoice", "paid", "due"],
        "required_slots": ["account_number", "billing_zip"],
    },
    IntentType.NETWORK_COVERAGE.value: {
        "intent": IntentType.NETWORK_COVERAGE.value,
        "display_name": "Network Coverage & Outages",
        "description": (
            "Cellular service issues, dropped calls, slow or unavailable LTE/3G data, "
            "lack of signal/bars, local tower outages, or roaming connection issues."
        ),
        "risk_tier": RiskTier.MEDIUM.value,
        "escalation_priority": "P2",
        "keywords": ["signal", "bars", "lte", "data", "service", "slow", "network", "tower", "outage", "coverage", "internet", "dropped call", "speed"],
        "required_slots": ["zip_code", "device_model"],
    },
    IntentType.DEVICE_HARDWARE.value: {
        "intent": IntentType.DEVICE_HARDWARE.value,
        "display_name": "Device & Hardware Troubleshooting",
        "description": (
            "Problems with phone, watch, or tablet hardware: SIM card errors, device activation, "
            "screen damage, battery failure, operating system bugs, or shipping delays of new phones."
        ),
        "risk_tier": RiskTier.MEDIUM.value,
        "escalation_priority": "P2",
        "keywords": ["phone", "iphone", "samsung", "activation", "activate", "sim", "screen", "broken", "battery", "apple watch", "hardware", "shipped", "delivery"],
        "required_slots": ["device_model", "order_number"],
    },
    IntentType.PLAN_UPGRADE.value: {
        "intent": IntentType.PLAN_UPGRADE.value,
        "display_name": "Plans, Promos & Upgrades",
        "description": (
            "Inquiries regarding trade-in deals, phone upgrades, rate plan changes, "
            "data plan packages, lease terms, promotional pricing, or eligibility."
        ),
        "risk_tier": RiskTier.LOW.value,
        "escalation_priority": "P3",
        "keywords": ["plan", "upgrade", "promo", "deal", "discount", "trade-in", "lease", "pricing", "unlimited", "switch", "offer", "eligible"],
        "required_slots": ["current_plan"],
    },
    IntentType.CANCELLATION_CHURN.value: {
        "intent": IntentType.CANCELLATION_CHURN.value,
        "display_name": "Cancellation & Churn Risk",
        "description": (
            "Customers explicitly stating desire to cancel service, switch to a competitor "
            "(Verizon, AT&T, T-Mobile), port out their telephone numbers, or terminate contract."
        ),
        "risk_tier": RiskTier.HIGH.value,
        "escalation_priority": "P1",
        "keywords": ["cancel", "leave", "switch to", "dump", "port", "close account", "switching", "leaving", "retention", "cancel service", "contract"],
        "required_slots": ["account_number", "cancellation_reason"],
    },
    IntentType.GENERAL_INQUIRY.value: {
        "intent": IntentType.GENERAL_INQUIRY.value,
        "display_name": "General Inquiry & Feedback",
        "description": (
            "General brand questions, customer support telephone numbers, store locations, "
            "policy clarifications, or customer sentiment/feedback without a specific technical request."
        ),
        "risk_tier": RiskTier.LOW.value,
        "escalation_priority": "P3",
        "keywords": ["hours", "store", "number", "contact", "support", "help", "chat", "representative", "feedback", "question", "info"],
        "required_slots": [],
    },
}


def get_all_intents() -> List[str]:
    """Returns list of valid intent string names."""
    return [intent.value for intent in IntentType]


def get_intent_metadata(intent_name: str) -> Dict[str, Any]:
    """Retrieves taxonomy metadata for a given intent string."""
    return INTENT_TAXONOMY.get(
        intent_name,
        {
            "intent": intent_name,
            "display_name": "Unknown",
            "description": "Unknown or out-of-domain category.",
            "risk_tier": RiskTier.LOW.value,
            "escalation_priority": "P3",
            "keywords": [],
            "required_slots": [],
        },
    )


def get_risk_tier(intent_name: str) -> str:
    """Returns the risk tier (LOW, MEDIUM, HIGH) for an intent."""
    meta = get_intent_metadata(intent_name)
    return meta["risk_tier"]
