# Day 08 Lab Report

## 1. Team / student

- Name: [Student Name]
- Repo/commit: [Git URL and commit hash]
- Date: 2026-08-25 17:51

## 2. Architecture

### Graph Design

The support ticket agent uses LangGraph's StateGraph with 11 nodes:

**Nodes:**
- `intake` - Normalizes raw query
- `classify` - LLM-based intent classification using structured output
- `tool` - Mock tool execution with error simulation
- `evaluate` - Tool result quality check (retry loop gate)
- `answer` - LLM-generated final response
- `ask_clarification` - Generates clarification questions
- `risky_action` - Prepares actions for approval
- `approval` - Human-in-the-loop approval (mock default)
- `retry_or_fallback` - Increments retry counter
- `dead_letter` - Handles max retry exhaustion
- `finalize` - Emits final audit event

**Routing Logic:**
- `route_after_classify` - Maps to answer/tool/clarify/risky_action/retry
- `route_after_evaluate` - Determines success vs needs_retry
- `route_after_retry` - Bounded retry (attempt < max_attempts)
- `route_after_approval` - Routes based on approved/rejected

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append | Audit conversation/events |
| tool_results | append | Cumulative tool execution history |
| errors | append | All errors across retries |
| events | append | Full audit trail |
| route | overwrite | Current route only |
| attempt | overwrite | Retry counter |
| final_answer | overwrite | Final response |
| evaluation_result | overwrite | Retry gate decision |
| pending_question | overwrite | Clarification question |
| proposed_action | overwrite | Risky action description |
| approval | overwrite | Latest approval decision |

## 4. Scenario results

| Scenario | Expected | Actual | Success | Retries | Interrupts |
|---|---|---|---|---:|---:|
| S01_simple | simple | simple | PASS | 0 | 0 |
| S02_tool | tool | tool | PASS | 0 | 0 |
| S03_missing | missing_info | missing_info | PASS | 0 | 0 |
| S04_risky | risky | risky | PASS | 0 | 1 |
| S05_error | error | error | PASS | 2 | 0 |
| S06_delete | risky | risky | PASS | 0 | 1 |
| S07_dead_letter | error | error | PASS | 1 | 0 |

**Summary:**
- Total scenarios: 7
- Success rate: 100.0%
- Avg nodes visited: 6.4
- Total retries: 3
- Total interrupts: 2

## 5. Failure analysis

1. **Retry or tool failure:**
   - Error scenarios simulate transient failures (timeout, connection errors)
   - Retry loop bounded by `attempt < max_attempts` check
   - After max retries, dead_letter node handles escalation

2. **Risky action without approval:**
   - Risky routes require approval before execution
   - Mock approval defaults to approved=True for testing
   - Real HITL via `LANGGRAPH_INTERRUPT=true` env var

3. **LLM classification errors:**
   - Priority: risky > tool > missing_info > error > simple
   - Structured output ensures reliable enum classification
   - Fallback to "answer" for unknown routes

## 6. Persistence / recovery evidence

- **Checkpointer**: Memory checkpointer (default) or SQLite (extension)
- **Thread ID**: Unique per scenario (`thread-<scenario_id>`)
- **State history**: Events trail shows all nodes visited
- **Recovery**: Graph can resume from checkpoint if interrupted

## 7. Extension work

Implemented SQLite checkpointer with:
- WAL mode for better concurrency
- Checkpoint storage at `outputs/checkpoints.db`
- Full state history persistence

## 8. Improvement plan

If I had one more day, I would:

1. **Real tool integration** - Connect to actual APIs (order lookup, refund processing)
2. **LLM-as-judge evaluator** - Replace heuristic with LLM-based quality assessment
3. **Streaming UI** - Add real-time visualization of graph execution
4. **Metrics dashboard** - Interactive dashboard for scenario results
