"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Classification Schema ──────────────────────────────────────────
class ClassificationResult(BaseModel):
    route: str
    reasoning: str


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")

    llm = get_llm()
    classifier = llm.with_structured_output(ClassificationResult)

    classification_prompt = f"""Classify this support ticket into exactly one route:

Routes (priority order - choose the HIGHEST applicable):
1. "risky" - Actions with side effects: refunds, deletions, cancellations
2. "tool" - Information lookups: order status, tracking, account info
3. "missing_info" - Vague/incomplete queries lacking actionable context
4. "error" - System failures: timeouts, crashes, service unavailable
5. "simple" - General questions answerable without tools or actions

Query to classify: "{query}"

Return the route and your reasoning."""

    result = classifier.invoke(classification_prompt)
    route = result.route
    risk_level = "high" if route == "risky" else "low"

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                classification=result.model_dump(),
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    # Simulate transient failures for error route
    if route == "error" and attempt < 2:
        result = f"ERROR: Transient failure on attempt {attempt + 1} - connection timeout"
        events = [make_event("tool", "error", f"simulated error on attempt {attempt + 1}")]
    else:
        # Mock successful tool results based on query type
        if route == "tool":
            result = (
                "Order status for order 12345: Shipped, "
                "expected delivery in 3-5 business days."
            )
        elif route == "risky":
            result = "Action prepared: Refund processed for customer."
        else:
            result = f"Tool execution completed successfully for: {query[:50]}"

        events = [make_event("tool", "completed", "tool execution successful")]

    return {
        "tool_results": [result],
        "events": events,
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    tool_results = state.get("tool_results", [])
    if not tool_results:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "no tool results, needs retry")],
        }

    latest_result = tool_results[-1]

    # Heuristic: check for ERROR substring
    if "ERROR" in latest_result:
        evaluation_result = "needs_retry"
        error_preview = latest_result[:50]
        events = [
            make_event("evaluate", "retry_needed", f"tool result contains error: {error_preview}")
        ]
    else:
        evaluation_result = "success"
        events = [make_event("evaluate", "success", "tool result satisfactory")]

    return {
        "evaluation_result": evaluation_result,
        "events": events,
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    proposed_action = state.get("proposed_action", "")

    # Build context for the LLM
    context_parts = [f"User query: {query}"]

    if tool_results:
        context_parts.append(f"Tool results: {tool_results[-1]}")

    if approval:
        context_parts.append(f"Approval: {getattr(approval, 'comment', 'Approved')}")

    if proposed_action:
        context_parts.append(f"Action taken: {proposed_action}")

    context = "\n".join(context_parts)

    llm = get_llm(temperature=0.3)

    answer_prompt = (
        "You are a helpful customer support agent. "
        "Generate a concise, helpful response based on the following context:\n\n"
        f"{context}\n\nProvide a clear, helpful answer to the user's question."
    )

    response = llm.invoke(answer_prompt)
    final_answer = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")

    llm = get_llm(temperature=0.3)

    clarification_prompt = f"""The user sent this vague or incomplete support request:
"{query}"

Generate ONE specific clarification question that would help address their needs.
The question should be polite, clear, and actionable.

Return only the question, nothing else."""

    response = llm.invoke(clarification_prompt)
    pending_question = response.content if hasattr(response, "content") else str(response)

    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "events": [make_event("clarify", "completed", "clarification question generated")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")

    llm = get_llm(temperature=0.3)

    action_prompt = (
        f'The user has requested the following action:\n"{query}"\n\n'
        "This action has been classified as RISKY because it involves "
        "side effects like refunds, deletions, or modifications.\n\n"
        "Describe:\n"
        "1. What action will be taken\n"
        "2. Why human approval is required\n"
        "3. Any potential implications\n\n"
        "Keep it concise and professional."
    )

    response = llm.invoke(action_prompt)
    proposed_action = response.content if hasattr(response, "content") else str(response)

    return {
        "proposed_action": proposed_action,
        "events": [
            make_event("risky_action", "prepared", "risky action prepared for approval")
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return:
        {"approval": {"approved": bool, "reviewer": str, "comment": str},
         "events": [make_event(...)]}
    """
    proposed_action = state.get("proposed_action", "")

    # Check for real HITL mode
    is_interrupt = os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true"
    if is_interrupt:
        from langgraph.types import interrupt

        interrupt_result = interrupt(f"Approval required for: {proposed_action[:100]}")
        # In real HITL, this would be set by external review
        approved = (
            interrupt_result.get("approved", False)
            if isinstance(interrupt_result, dict)
            else False
        )
    else:
        # Mock approval for testing/CI
        approved = True

    reviewer = "human-reviewer" if is_interrupt else "mock-reviewer"
    comment = "Action rejected" if not approved else "Mock approved for testing"

    approval = ApprovalDecision(approved=approved, reviewer=reviewer, comment=comment)

    status = "approved" if approved else "rejected"
    return {
        "approval": approval,
        "events": [make_event("approval", "completed", f"action {status}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    current_attempt = state.get("attempt", 0)
    new_attempt = current_attempt + 1

    error_msg = f"Transient failure on attempt {new_attempt}, retrying..."

    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "events": [make_event("retry", "attempt", f"retry attempt {new_attempt}")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)

    final_answer = (
        f"I apologize, but I was unable to complete your request after {attempt} attempts. "
        f"Your request for '{query}' has been escalated to our support team for manual review. "
        f"They will contact you within 24-48 hours to resolve the issue."
    )

    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "dead_letter", "escalated", f"max retries ({max_attempts}) exceeded"
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
