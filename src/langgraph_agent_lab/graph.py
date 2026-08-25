"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from .routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from .state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow.

    Architecture:
    START → intake → classify → [conditional: route_after_classify]
      simple       → answer → finalize → END
      tool         → tool → evaluate → [conditional: route_after_evaluate]
                                          success → answer → finalize → END
                                          needs_retry → retry → [conditional: route_after_retry]
                                                                  tool (retry)
                                                                  dead_letter → finalize → END
      missing_info → clarify → finalize → END
      risky        → risky_action → approval → [conditional: route_after_approval]
                                                  approved → tool → evaluate → ...
                                                  rejected → clarify → finalize → END
      error        → retry → [conditional: route_after_retry] → ...
    """
    from langgraph.graph import START

    # Create the StateGraph
    graph = StateGraph(AgentState)

    # Register all 11 nodes
    graph.add_node("intake", intake_node)
    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("tool", tool_node)
    graph.add_node("clarify", ask_clarification_node)
    graph.add_node("risky_action", risky_action_node)
    graph.add_node("approval", approval_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("retry", retry_or_fallback_node)
    graph.add_node("dead_letter", dead_letter_node)
    graph.add_node("finalize", finalize_node)

    # Add fixed edges
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")

    # Conditional edge after classify
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )

    # Simple path: answer → finalize → END
    graph.add_edge("answer", "finalize")
    graph.add_edge("finalize", END)

    # Missing info path: clarify → finalize → END
    graph.add_edge("clarify", "finalize")

    # Tool path: tool → evaluate → [conditional]
    graph.add_edge("tool", "evaluate")

    # Conditional edge after evaluate
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "answer": "answer",
            "retry": "retry",
        },
    )

    # Retry loop: retry → [conditional] → tool or dead_letter
    graph.add_conditional_edges(
        "retry",
        route_after_retry,
        {
            "tool": "tool",
            "dead_letter": "dead_letter",
        },
    )

    # Dead letter path: dead_letter → finalize → END
    graph.add_edge("dead_letter", "finalize")

    # Risky path: risky_action → approval → [conditional]
    graph.add_edge("risky_action", "approval")

    # Conditional edge after approval
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "tool": "tool",
            "clarify": "clarify",
        },
    )

    # Error path goes to retry (already wired above via classify → retry)

    # Compile with optional checkpointer
    return graph.compile(checkpointer=checkpointer)
