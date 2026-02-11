# Temporary Spec: Fix Tool Schema Validation for Escalation Agent

**Status:** ACTIVE - Delete this file when work is complete
**Created:** 2026-02-09
**Context:** This document captures the current blocker in the flagging pipeline fix. Review the full conversation context in the .jsonl transcript for complete details.

---

## Current State - What's Working ✅

The demo testing pipeline has been successfully fixed to:
1. ✅ All 6 agents now have LLM configuration via `src/agents/llm_config.py` (GPT-4o-mini via OpenRouter)
2. ✅ Data quality agent detects 765 duplicate records correctly
3. ✅ Reconciliation agent finds 85 suspicious/unmatched transactions
4. ✅ Agents produce properly formatted combined JSON output with both data_quality and reconciliation results
5. ✅ Pandas Timestamp serialization issue fixed in orchestrator
6. ✅ Pipeline runs without errors up to the escalation agent

**Evidence of success:**
```
📊 Parallel Agent Results Summary:
  • Duplicates detected: 765
  • Suspicious transactions identified: 85
  • Ready for escalation agent
```

---

## Current Blocker 🚨

**Error Message:**
```
openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid schema for function 'calculate_severity_score': In context=('properties', 'transaction'), 'additionalProperties' is required to be supplied and to be false. See https://platform.openai.com/docs/guides/function-calling for more information.", 'type': 'invalid_request_error', 'param': 'tools[0].function.parameters.properties.transaction', 'code': 'invalid_function_parameters'}}
```

**Root Cause:**
- The escalation agent is trying to call the `calculate_severity_score` tool
- CrewAI's `@tool` decorator is generating JSON schemas for OpenAI function calling
- OpenAI's API has strict validation: object-type parameters MUST have `additionalProperties: false`
- The auto-generated schema from `@tool` decorator doesn't include this requirement

**Where the Error Occurs:**
- File: `src/tools/escalation_tools.py`
- Tool: `calculate_severity_score` (lines ~25-60)
- Parameter: `transaction` (dict type)

---

## What Needs to Be Fixed

### Problem: CrewAI @tool Decorator Schema Generation

Current tool definition pattern (example from `calculate_severity_score`):
```python
@tool("Calculate severity score for suspicious transaction")
def calculate_severity_score(transaction: dict, agent_results: dict) -> dict:
    """
    Calculate severity score based on multiple factors.

    Args:
        transaction (dict): Transaction details with keys: txn_id, vendor, amount, date, source
        agent_results (dict): Results from data_quality and reconciliation agents

    Returns:
        dict: {severity_score, level, confidence, contributing_factors}
    """
    # Implementation...
```

**Issue:** The `transaction: dict` and `agent_results: dict` type hints generate schemas like:
```json
{
  "type": "object",
  "properties": {
    "transaction": {"type": "object"},  // ❌ Missing additionalProperties!
    "agent_results": {"type": "object"}  // ❌ Missing additionalProperties!
  }
}
```

**OpenAI Requirement:**
```json
{
  "type": "object",
  "properties": {
    "transaction": {
      "type": "object",
      "additionalProperties": false,  // ✅ Required!
      "properties": { /* field definitions */ }
    }
  }
}
```

---

## Solution Approaches

### Option 1: Explicit Schema in Tool Docstring (Recommended)

CrewAI allows JSON schema in docstrings. Modify tool definitions to include explicit schemas:

```python
from pydantic import BaseModel, Field

class TransactionInput(BaseModel):
    """Transaction details for severity calculation"""
    txn_id: str = Field(description="Transaction ID")
    vendor: str = Field(description="Vendor name")
    amount: float = Field(description="Transaction amount")
    date: str = Field(description="Transaction date")
    source: str = Field(description="Data source (e.g., credit_card, bank)")

class AgentResultsInput(BaseModel):
    """Combined results from data_quality and reconciliation agents"""
    data_quality: dict = Field(description="Data quality agent output")
    reconciliation: dict = Field(description="Reconciliation agent output")

@tool("Calculate severity score for suspicious transaction")
def calculate_severity_score(
    transaction: TransactionInput,
    agent_results: AgentResultsInput
) -> dict:
    """
    Calculate severity score based on multiple factors.

    Returns:
        dict: {severity_score, level, confidence, contributing_factors}
    """
    # Convert Pydantic models back to dicts for internal use
    txn = transaction.model_dump()
    results = agent_results.model_dump()

    # Existing implementation...
```

**Pros:**
- Pydantic models automatically include `additionalProperties: false`
- Type validation built-in
- Clear schema documentation

**Cons:**
- Need to convert Pydantic models back to dicts inside tools
- More verbose code

---

### Option 2: Manual Schema Override (Quick Fix)

If CrewAI supports manual schema override, add explicit schemas:

```python
@tool(
    "Calculate severity score for suspicious transaction",
    args_schema={
        "transaction": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "txn_id": {"type": "string"},
                "vendor": {"type": "string"},
                "amount": {"type": "number"},
                "date": {"type": "string"},
                "source": {"type": "string"}
            },
            "required": ["txn_id", "amount"]
        },
        "agent_results": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "data_quality": {"type": "object"},
                "reconciliation": {"type": "object"}
            }
        }
    }
)
def calculate_severity_score(transaction: dict, agent_results: dict) -> dict:
    # Existing implementation unchanged
```

**Pros:**
- Minimal code changes
- Keeps existing dict-based implementation

**Cons:**
- Need to check if CrewAI `@tool` supports `args_schema` parameter
- Manual schema maintenance

---

### Option 3: Update CrewAI Tool Registration

Check if there's a way to patch the tool schema after registration:

```python
from crewai.tools import Tool

# After tool definition
calculate_severity_score._schema["properties"]["transaction"]["additionalProperties"] = False
calculate_severity_score._schema["properties"]["agent_results"]["additionalProperties"] = False
```

**Pros:**
- No changes to tool implementation
- Quick hotfix

**Cons:**
- Relies on internal CrewAI implementation details
- Fragile, may break with updates

---

## Tools That Need Fixing

In `src/tools/escalation_tools.py`, these tools likely have the same issue:

1. **`calculate_severity_score`** (line ~25)
   - Parameters: `transaction: dict`, `agent_results: dict`
   - CRITICAL - This is the first tool escalation agent tries to call

2. **`generate_root_cause_analysis`** (line ~85)
   - Parameters: `transaction: dict`, `agent_results: dict`

3. **`batch_classify_with_llm`** (line ~130)
   - Parameters: `transactions: list`, `agent_results: dict`

4. **`create_audit_flag`** (line ~170)
   - Parameters: `transaction_id: str`, `audit_run_id: str`, `severity: str`, `explanation: str`, `evidence: dict`
   - The `evidence: dict` parameter needs fixing

5. **`check_escalation_rules`** (line ~210)
   - Parameters: `severity: str`, `amount: float`, `vendor: str`
   - Might be OK if all params are primitives (not dicts)

---

## Verification Steps

After implementing the fix:

### 1. Check Tool Schema Generation
```python
# Test script: tests/debug_tool_schema.py
from src.tools.escalation_tools import calculate_severity_score
import json

# Inspect the tool's schema
if hasattr(calculate_severity_score, '_schema'):
    print(json.dumps(calculate_severity_score._schema, indent=2))
elif hasattr(calculate_severity_score, 'args_schema'):
    print(json.dumps(calculate_severity_score.args_schema, indent=2))
else:
    print("Cannot inspect tool schema - try alternate method")

# Verify additionalProperties is present
```

### 2. Run Escalation Agent in Isolation
```python
# Test script: tests/test_escalation_isolated.py
import os
os.environ['TEST_MODE'] = 'true'

from src.agents.escalation_agent import escalation_agent, escalation_task
from crewai import Crew, Process

suspicious_txns = [
    {
        'txn_id': 'cc_001',
        'vendor': 'Unknown Inc',
        'amount': 5000,
        'date': '2024-01-15',
        'source': 'credit_card'
    }
]

parallel_results = {
    'data_quality': {'quality_score': 0.85, 'incomplete_records': []},
    'reconciliation': {'matched': False, 'match_rate': 0.5}
}

crew = Crew(
    agents=[escalation_agent],
    tasks=[escalation_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={
    'suspicious_transactions': suspicious_txns,
    'audit_run_id': 'test-run-001',
    'parallel_results': parallel_results
})

print("Result:", result)

# Verify flags were created
from src.tools.escalation_tools import get_test_mode_flags
flags = get_test_mode_flags()
print(f"Flags created: {len(flags)}")
assert len(flags) > 0, "No flags created!"
```

### 3. Run Full Demo Pipeline
```bash
# Run on duplicates dataset with 100 transaction limit
python3 tests/demo_testing.py

# Expected output:
# ✅ Pipeline completed
# 📋 Flags created: 20-50 (should be NON-ZERO!)
# Confusion Matrix: TP > 0
```

### 4. Success Criteria
- ✅ No OpenAI schema validation errors
- ✅ Escalation agent successfully calls `calculate_severity_score`
- ✅ Escalation agent calls `create_audit_flag` for each suspicious transaction
- ✅ `get_test_mode_flags()` returns non-empty list
- ✅ Confusion matrix shows TP > 0

---

## Implementation Priority

**Start with Option 1 (Pydantic models)** because:
1. It's the most robust long-term solution
2. Pydantic is already in the project dependencies
3. Provides type validation and clear schemas
4. Future-proof for additional tools

**If Option 1 doesn't work:** Try Option 3 (schema patching) as a quick hotfix while investigating CrewAI's tool schema generation internals.

---

## Related Files to Review

- `src/tools/escalation_tools.py` - **Primary file to modify**
- `src/agents/escalation_agent.py` - Task description (already correct)
- `src/orchestrator/orchestrator_agent.py:204-240` - Where escalation agent is called
- `.env` - Verify OPENROUTER_API_KEY is set
- `/private/tmp/claude-501/-Users-kittenoverlord-projects-ergonosis-auditing/tasks/b41fa49.output` - Full error logs

---

## Context for New Agent

**Before starting work:**
1. Read the full conversation transcript: `/Users/kittenoverlord/projects/ergonosis_auditing/.claude/projects/-Users-kittenoverlord-projects-ergonosis-auditing/565944b9-45e8-432a-90ce-7eadb8e74326.jsonl`
2. Review the plan file: `/Users/kittenoverlord/.claude/plans/structured-skipping-lampson.md`
3. Check MEMORY.md for key learnings: `/Users/kittenoverlord/.claude/projects/-Users-kittenoverlord-projects-ergonosis-auditing/memory/MEMORY.md`

**Key insights from previous work:**
- Agents need LLMs for tool orchestration (this was the original root cause)
- The reconciliation task must aggregate BOTH data_quality AND reconciliation results
- Pandas Timestamps must be converted to ISO strings before passing to CrewAI
- The pipeline successfully detects suspicious transactions - only flag creation is blocked

**What's already been tried:**
- ✅ Added LLM configuration to all agents
- ✅ Fixed agent output format aggregation
- ✅ Fixed Timestamp serialization
- ✅ Enhanced task descriptions
- ❌ Tool schema validation - THIS IS THE CURRENT BLOCKER

**Don't waste time on:**
- Re-investigating why agents weren't producing output (already fixed with LLM config)
- Debugging orchestrator output aggregation (already fixed)
- Worrying about cost ($0.05-0.10 per test run is acceptable)

---

## Instructions for Completion

When you've successfully fixed the tool schema issue and verified flags are being created:

1. **Run the full test suite:**
   ```bash
   python3 tests/demo_testing.py
   ```

2. **Verify success criteria:**
   - All 4 datasets complete without errors
   - Flags created > 0 for corrupted datasets
   - Recall ≥ 80%, Precision ≥ 75%, F1 ≥ 0.80

3. **Update MEMORY.md** with the tool schema fix lesson:
   ```markdown
   ### Tool Schema Validation (OpenAI API)
   - CrewAI @tool decorator must generate schemas with `additionalProperties: false` for object parameters
   - Use Pydantic models for tool parameters to ensure schema compliance
   - Example: `transaction: TransactionInput` instead of `transaction: dict`
   ```

4. **Delete this file:**
   ```bash
   rm /Users/kittenoverlord/projects/ergonosis_auditing/TOOL_SCHEMA_FIX_SPEC.md
   ```

5. **Report final results to user:**
   - Number of flags created per dataset
   - Confusion matrix metrics
   - Total pipeline duration
   - Confirm the fix is complete

---

**IMPORTANT:** Dig through the full conversation context before starting. The .jsonl transcript contains detailed debugging sessions, error messages, and attempted solutions. Understanding what's already been tried will save significant time.

**Good luck! The pipeline is 95% working - just need to fix these tool schemas and we're done.** 🚀
