"""
Lab 11 — Optional enrichment: Human-in-the-Loop Design
  (Không chấm — tham khảo. Tóm tắt nộp do scripts/grade.py tự sinh,
   không viết report/*.md tay.)
  - Confidence Router
  - 3 HITL decision points
"""
from dataclasses import dataclass


# ============================================================
# Optional enrichment: ConfidenceRouter (không chấm)
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        # 1. High-risk banking actions always need a human, whatever the score.
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        # 2. Confidence bands.
        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )
        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )
        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


# ============================================================
# Optional enrichment: 3 HITL decision points (không chấm)
# Không bắt buộc điền. Tóm tắt bài nộp: chạy scripts/grade.py
# (tự sinh lab_report.md) — không viết report tay.
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "High-risk money movement approval",
        "trigger": (
            "Any transfer_money / close_account / change_password / delete_data / "
            "update_personal_info action, regardless of model confidence or claimed "
            "authority."
        ),
        "hitl_model": "human-in-the-loop (blocking approval before the sink runs)",
        "context_needed": (
            "Customer ID, source and destination account, amount, currency, the "
            "originating message thread, and the model's proposed action as a diff."
        ),
        "example": (
            "Customer asks to transfer 500,000,000 VND to a new payee. The agent "
            "proposes the transfer; a human agent must approve before the egress "
            "call to api.vinbank.example."
        ),
        "approval_path": (
            "Approve → action executes with the approval id. Reject → action dropped "
            "and customer told it needs branch verification. Timeout (e.g. 15 min) → "
            "treated as reject and escalated to the fraud queue."
        ),
        "audit_fields": (
            "correlation_id, user_id, intent, proposed_action_diff, reviewer_id, "
            "approval_id (HITL-XXXXXXXX), decision, decided_at."
        ),
    },
    {
        "id": 2,
        "name": "Secret / PII leak containment",
        "trigger": (
            "Output guardrail finds a password, sk-* API key, *.internal host, "
            "national ID, phone or email in a draft reply."
        ),
        "hitl_model": "human-on-the-loop (auto-redact now, review the incident after)",
        "context_needed": (
            "Redacted vs original draft, which pattern matched, the prompt that "
            "produced it, and the target agent (Blue / Red / Red Advance)."
        ),
        "example": (
            "A jailbreak coaxes the model into echoing the admin password; the "
            "output filter replaces the reply and files a security incident."
        ),
        "approval_path": (
            "Auto-redact ships immediately; a security reviewer confirms the "
            "incident, tunes the rule, and may raise a hardening ticket. No release "
            "of the raw secret at any point."
        ),
        "audit_fields": (
            "correlation_id, matched_pattern, original_preview_hash, redacted_reply, "
            "reviewer_id, incident_status."
        ),
    },
    {
        "id": 3,
        "name": "Low-confidence answer escalation",
        "trigger": (
            "ConfidenceRouter returns queue_review (0.7–0.9) or escalate (<0.7), e.g. "
            "ambiguous product/rate questions or hallucination risk."
        ),
        "hitl_model": "human-as-tiebreaker (only when the router is unsure)",
        "context_needed": (
            "The customer question, the draft answer, the confidence score, and the "
            "ground-truth reference from data/pii_hallucination_samples.json."
        ),
        "example": (
            "Customer asks for a rate that is not in ground truth; the model is 0.6 "
            "confident. The answer is queued so a human confirms the real figure "
            "before it reaches the customer."
        ),
        "approval_path": (
            "Approve → answer sent as-is. Edit → reviewer corrects the figure and "
            "sends the corrected reply. Reject/timeout → a safe holding message is "
            "sent and a human follows up."
        ),
        "audit_fields": (
            "correlation_id, intent, confidence, draft_answer, ground_truth_ref, "
            "reviewer_id, decision, final_answer."
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
