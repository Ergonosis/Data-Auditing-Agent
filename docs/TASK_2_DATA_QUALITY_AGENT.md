# TASK 2: Data Quality Agent & Tools

## Objective
Implement the Data Quality Agent with its 5 deterministic tools that validate data completeness, schema conformity, duplicates, domain inference, and quality gates.

## Context
This is the **first agent** in the parallel execution phase. It runs BEFORE Reconciliation and Anomaly agents. The Data Quality Agent is primarily deterministic (SQL-based) with ONE optional LLM call for domain inference if no manual config exists.

**Critical**: This agent should complete in ~30 seconds for 1000 transactions.

## Architecture Overview
```
Data Quality Agent
├─ Tool 1: check_data_completeness (SQL query)
├─ Tool 2: validate_schema_conformity (Pandas validation)
├─ Tool 3: detect_duplicate_records (SQL GROUP BY)
├─ Tool 4: infer_domain_freshness (LLM if no manual config)
└─ Tool 5: check_data_quality_gates (Rule-based threshold)
```

---

## Files to Create

### 1. `/src/tools/__init__.py`
```python
"""Tool modules"""
```

### 2. `/src/tools/data_quality_tools.py` (~250 lines)

**Purpose**: 5 tools for data quality validation.

**Key Requirements**:
- All tools must be decorated with `@tool` from `crewai_tools`
- Tools 1-3, 5 are pure SQL/Python (NO LLM)
- Tool 4 uses LLM only if domain not in config
- Return structured dictionaries, NOT strings

**Implementation**:

```python
"""Data Quality validation tools"""

from crewai_tools import tool
import pandas as pd
from typing import Dict, Any, List
from src.tools.databricks_client import query_gold_tables
from src.tools.llm_client import call_llm
from src.utils.config_loader import load_config, get_domain_config
from src.utils.logging import get_logger
import json
import os

logger = get_logger(__name__)

@tool("check_data_completeness")
def check_data_completeness(table_name: str) -> dict:
    """
    Validate data completeness - check if required fields are populated

    Args:
        table_name: Name of table to check (e.g., 'gold.recent_transactions')

    Returns:
        Dictionary with completeness metrics:
        {
            'total_records': int,
            'missing_vendor': int,
            'missing_amount': int,
            'missing_date': int,
            'missing_source': int,
            'completeness_score': float  # 0-1
        }
    """
    logger.info(f"Checking data completeness for {table_name}")

    try:
        df = query_gold_tables(f"SELECT * FROM {table_name}")

        if df.empty:
            logger.warning(f"No data found in {table_name}")
            return {
                'total_records': 0,
                'missing_vendor': 0,
                'missing_amount': 0,
                'missing_date': 0,
                'missing_source': 0,
                'completeness_score': 0.0
            }

        total_records = len(df)
        required_fields = ['vendor', 'amount', 'date', 'source']

        missing_counts = {}
        for field in required_fields:
            if field in df.columns:
                missing_counts[f'missing_{field}'] = int(df[field].isnull().sum())
            else:
                missing_counts[f'missing_{field}'] = total_records

        # Calculate completeness score
        total_cells = total_records * len(required_fields)
        missing_cells = sum(missing_counts.values())
        completeness_score = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

        result = {
            'total_records': total_records,
            **missing_counts,
            'completeness_score': round(completeness_score, 3)
        }

        logger.info(f"Completeness check complete", **result)
        return result

    except Exception as e:
        logger.error(f"Completeness check failed: {e}")
        raise


@tool("validate_schema_conformity")
def validate_schema_conformity(table_name: str, expected_schema: dict) -> list:
    """
    Validate schema conformity - check data types and structure

    Args:
        table_name: Name of table to validate
        expected_schema: Expected schema {'field': 'type', ...}
                        e.g., {'amount': 'float', 'vendor': 'str', 'date': 'datetime'}

    Returns:
        List of validation errors (empty if all valid)
        [
            {'field': 'amount', 'error': 'Invalid type', 'details': '...'},
            ...
        ]
    """
    logger.info(f"Validating schema for {table_name}")

    try:
        df = query_gold_tables(f"SELECT * FROM {table_name}")

        if df.empty:
            return []

        errors = []

        for field, expected_type in expected_schema.items():
            if field not in df.columns:
                errors.append({
                    'field': field,
                    'error': 'Missing field',
                    'details': f'Field {field} not found in table'
                })
                continue

            # Check data type
            actual_type = str(df[field].dtype)

            type_map = {
                'float': ['float64', 'float32'],
                'int': ['int64', 'int32'],
                'str': ['object'],
                'datetime': ['datetime64[ns]', 'datetime64']
            }

            if expected_type in type_map:
                if actual_type not in type_map[expected_type]:
                    errors.append({
                        'field': field,
                        'error': 'Type mismatch',
                        'details': f'Expected {expected_type}, got {actual_type}'
                    })

        logger.info(f"Schema validation complete: {len(errors)} errors found")
        return errors

    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise


@tool("detect_duplicate_records")
def detect_duplicate_records(table_name: str, key_fields: list) -> dict:
    """
    Detect duplicate records based on key fields

    Args:
        table_name: Name of table to check
        key_fields: List of fields to check for duplicates
                   e.g., ['txn_id'] or ['vendor', 'amount', 'date']

    Returns:
        Dictionary with duplicate info:
        {
            'duplicate_count': int,
            'duplicate_groups': [
                {'key': 'value', 'count': 3, 'ids': ['id1', 'id2', 'id3']},
                ...
            ]
        }
    """
    logger.info(f"Detecting duplicates in {table_name} on fields {key_fields}")

    try:
        df = query_gold_tables(f"SELECT * FROM {table_name}")

        if df.empty or not all(field in df.columns for field in key_fields):
            return {'duplicate_count': 0, 'duplicate_groups': []}

        # Find duplicates
        duplicates = df[df.duplicated(subset=key_fields, keep=False)]

        if duplicates.empty:
            logger.info("No duplicates found")
            return {'duplicate_count': 0, 'duplicate_groups': []}

        # Group duplicates
        duplicate_groups = []
        for key_values, group in duplicates.groupby(key_fields):
            duplicate_groups.append({
                'key': dict(zip(key_fields, key_values if isinstance(key_values, tuple) else [key_values])),
                'count': len(group),
                'ids': group.get('txn_id', group.index).tolist()[:5]  # Limit to 5 IDs
            })

        result = {
            'duplicate_count': len(duplicates),
            'duplicate_groups': duplicate_groups[:10]  # Limit to 10 groups
        }

        logger.info(f"Found {result['duplicate_count']} duplicates in {len(duplicate_groups)} groups")
        return result

    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        raise


@tool("infer_domain_freshness")
def infer_domain_freshness(transaction_pattern: dict) -> dict:
    """
    Infer domain-specific freshness requirements

    Args:
        transaction_pattern: Transaction characteristics
        {
            'frequency': 'daily' | 'weekly' | 'monthly',
            'vendor_type': 'supplier' | 'service' | 'subscription',
            'avg_amount': float,
            'domain': str (optional - if known)
        }

    Returns:
        Domain configuration:
        {
            'domain': 'inventory_management' | 'senior_living' | 'business_operations' | 'default',
            'max_age_hours': int,
            'confidence': float,
            'source': 'manual_config' | 'inferred'
        }
    """
    logger.info(f"Inferring domain freshness for pattern: {transaction_pattern}")

    try:
        # Load configuration
        config = load_config()

        # Check if domain is manually specified
        if 'domain' in transaction_pattern:
            domain = transaction_pattern['domain']
            domain_config = get_domain_config(config, domain)

            if domain_config:
                logger.info(f"Using manual domain config for {domain}")
                return {
                    'domain': domain,
                    'max_age_hours': domain_config.get('max_age_hours', 48),
                    'confidence': 1.0,
                    'source': 'manual_config',
                    'critical_amount_threshold': domain_config.get('critical_amount_threshold', 1000)
                }

        # No manual config - use LLM for inference
        logger.info("No manual config found, using LLM for domain inference")

        prompt = f"""
Given this transaction pattern, infer the business domain and data freshness requirements:
- Frequency: {transaction_pattern.get('frequency', 'unknown')}
- Vendor type: {transaction_pattern.get('vendor_type', 'unknown')}
- Average amount: ${transaction_pattern.get('avg_amount', 0)}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "domain": "inventory_management" or "senior_living" or "business_operations" or "default",
  "max_age_hours": 24 or 48 or 168,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}
"""

        response = call_llm(prompt, agent_name="DataQuality")

        # Parse JSON response
        try:
            result = json.loads(response.strip())
            result['source'] = 'inferred'
            logger.info(f"Domain inferred: {result['domain']} (confidence: {result.get('confidence', 0)})")
            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response, using default")
            return {
                'domain': 'default',
                'max_age_hours': 48,
                'confidence': 0.5,
                'source': 'fallback'
            }

    except Exception as e:
        logger.error(f"Domain inference failed: {e}")
        # Fallback to default
        return {
            'domain': 'default',
            'max_age_hours': 48,
            'confidence': 0.0,
            'source': 'error_fallback'
        }


@tool("check_data_quality_gates")
def check_data_quality_gates(quality_metrics: dict, thresholds: dict) -> bool:
    """
    Check if data quality meets minimum thresholds

    Args:
        quality_metrics: Output from check_data_completeness
        thresholds: Threshold configuration
                   {'completeness_threshold': 0.90, ...}

    Returns:
        True if quality passes gates, False if audit should halt
    """
    logger.info("Checking data quality gates")

    try:
        completeness_threshold = thresholds.get('completeness_threshold', 0.90)
        completeness_score = quality_metrics.get('completeness_score', 0)

        if completeness_score < completeness_threshold:
            logger.warning(
                f"Data quality gate FAILED: completeness {completeness_score} < {completeness_threshold}"
            )
            return False

        logger.info(f"Data quality gate PASSED: completeness {completeness_score} >= {completeness_threshold}")
        return True

    except Exception as e:
        logger.error(f"Quality gate check failed: {e}")
        return False
```

---

### 3. `/src/agents/__init__.py`
```python
"""Agent modules"""
```

### 4. `/src/agents/data_quality_agent.py` (~100 lines)

**Purpose**: CrewAI agent definition with task.

**Implementation**:

```python
"""Data Quality Agent - validates data before auditing"""

from crewai import Agent, Task
from src.tools.data_quality_tools import (
    check_data_completeness,
    validate_schema_conformity,
    detect_duplicate_records,
    infer_domain_freshness,
    check_data_quality_gates
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Agent definition
data_quality_agent = Agent(
    role="Data Quality Specialist",
    goal="Validate data completeness, schema conformity, and freshness with 95%+ accuracy before audit begins",
    backstory="""You are an expert data engineer with 10 years of experience in data validation.
    You ensure that all transaction data meets quality standards before the audit process begins.
    You are meticulous about data integrity and never let bad data pass through.""",

    tools=[
        check_data_completeness,
        validate_schema_conformity,
        detect_duplicate_records,
        infer_domain_freshness,
        check_data_quality_gates
    ],

    verbose=True,
    allow_delegation=False,
    llm=None  # No LLM needed for agent reasoning (tools handle LLM calls)
)

# Task definition
data_quality_task = Task(
    description="""
    Given a table of transactions from Databricks Gold layer:

    1. **Check data completeness** - ensure required fields (vendor, amount, date, source) are populated
    2. **Validate schema conformity** - verify data types match expected schema
    3. **Detect duplicate records** - find duplicate transactions based on txn_id
    4. **Infer domain freshness** - determine business domain and max data age (use config if available, else infer via LLM)
    5. **Apply quality gates** - check if completeness score meets threshold (default 90%)

    **Input**: transactions table name (e.g., 'gold.recent_transactions')
    **Output**: Structured report with:
    - quality_score (0-1)
    - incomplete_records (list of IDs)
    - schema_violations (list of errors)
    - duplicates (list of duplicate groups)
    - domain_config (dict with domain, max_age_hours)
    - freshness_violations (list of stale transactions)
    - gate_passed (bool - True if quality acceptable, False if audit should halt)

    **Important**:
    - Use tools sequentially in order listed above
    - If quality gate fails (completeness <90%), immediately return gate_passed=False
    - For domain inference, check config first before calling LLM
    - Log all findings clearly
    """,

    agent=data_quality_agent,

    expected_output="""JSON object with data quality report:
    {
        "quality_score": 0.95,
        "incomplete_records": ["cc_123", "cc_456"],
        "schema_violations": [],
        "duplicates": {"duplicate_count": 3, "duplicate_groups": [...]},
        "domain_config": {"domain": "business_operations", "max_age_hours": 48, "confidence": 1.0},
        "freshness_violations": [],
        "gate_passed": true
    }
    """
)
```

---

## Testing Requirements

Create `/tests/test_agents/test_data_quality_agent.py`:

```python
import pytest
from src.agents.data_quality_agent import data_quality_agent, data_quality_task
from src.tools.data_quality_tools import (
    check_data_completeness,
    validate_schema_conformity,
    detect_duplicate_records
)

def test_check_completeness():
    """Test completeness check with mock data"""
    result = check_data_completeness.func("gold.recent_transactions")
    assert 'total_records' in result
    assert 'completeness_score' in result
    assert 0 <= result['completeness_score'] <= 1

def test_schema_validation():
    """Test schema validation"""
    expected_schema = {
        'amount': 'float',
        'vendor': 'str',
        'date': 'datetime'
    }
    result = validate_schema_conformity.func("gold.recent_transactions", expected_schema)
    assert isinstance(result, list)

def test_duplicate_detection():
    """Test duplicate detection"""
    result = detect_duplicate_records.func("gold.recent_transactions", ['txn_id'])
    assert 'duplicate_count' in result
    assert 'duplicate_groups' in result

def test_domain_inference():
    """Test domain inference with manual config"""
    from src.tools.data_quality_tools import infer_domain_freshness
    pattern = {'domain': 'default', 'frequency': 'daily'}
    result = infer_domain_freshness.func(pattern)
    assert 'domain' in result
    assert 'max_age_hours' in result
```

---

## Success Criteria

✅ All 5 tools are properly decorated with `@tool`
✅ Tools return structured dicts (not strings)
✅ Completeness check works with mock data
✅ Schema validation detects type mismatches
✅ Duplicate detection groups correctly
✅ Domain inference tries config first, then LLM
✅ Quality gates halt audit if completeness <90%
✅ Agent and task are properly defined
✅ All imports work
✅ Unit tests pass

---

## Important Notes

- **CrewAI Note**: When `llm=None` for an agent, it relies purely on tool outputs
- **Tool Decorator**: `@tool` decorator tells CrewAI this is a usable tool
- **Return Types**: Tools MUST return dict/list, NOT strings (CrewAI requirement)
- **Logging**: Use structured logger extensively for debugging
- **Error Handling**: All tools should have try/except and return sensible defaults on error
- **Performance**: Target <30 seconds execution for 1000 transactions

---

## Dependencies
- `crewai`
- `crewai-tools`
- `pandas`
- Existing modules: `src.tools.databricks_client`, `src.tools.llm_client`, `src.utils.*`

---

## Estimated Effort
~350 lines of code, 1-2 hours for implementation + testing.
