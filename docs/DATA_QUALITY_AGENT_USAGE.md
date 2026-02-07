# Data Quality Agent - Usage Guide

## Quick Start

### Basic Usage

```python
from crewai import Crew
from src.agents.data_quality_agent import data_quality_agent, data_quality_task

# Create crew with just the Data Quality Agent
crew = Crew(
    agents=[data_quality_agent],
    tasks=[data_quality_task],
    verbose=True
)

# Execute data quality check
result = crew.kickoff(inputs={
    "table_name": "gold.recent_transactions"
})

print(result)
```

### Expected Output

```json
{
    "quality_score": 0.95,
    "incomplete_records": ["cc_123", "cc_456"],
    "schema_violations": [],
    "duplicates": {
        "duplicate_count": 3,
        "duplicate_groups": [
            {
                "key": {"txn_id": "txn_001"},
                "count": 2,
                "ids": ["txn_001", "txn_001"]
            }
        ]
    },
    "domain_config": {
        "domain": "business_operations",
        "max_age_hours": 48,
        "confidence": 1.0,
        "source": "manual_config"
    },
    "freshness_violations": [],
    "gate_passed": true
}
```

---

## Using Individual Tools

### Tool 1: Check Data Completeness

```python
from src.tools.data_quality_tools import check_data_completeness

result = check_data_completeness.func("gold.recent_transactions")

# Returns:
{
    'total_records': 1000,
    'missing_vendor': 5,
    'missing_amount': 0,
    'missing_date': 2,
    'missing_source': 8,
    'completeness_score': 0.996  # 0-1 scale
}
```

**When to use**: Validate that required fields are populated before processing.

---

### Tool 2: Validate Schema Conformity

```python
from src.tools.data_quality_tools import validate_schema_conformity

expected_schema = {
    'amount': 'float',
    'vendor': 'str',
    'date': 'datetime',
    'txn_id': 'str'
}

errors = validate_schema_conformity.func(
    "gold.recent_transactions",
    expected_schema
)

# Returns:
[
    {
        'field': 'amount',
        'error': 'Type mismatch',
        'details': 'Expected float, got object'
    }
]
# Empty list if all valid
```

**When to use**: Ensure data types match expected schema before analytics.

---

### Tool 3: Detect Duplicate Records

```python
from src.tools.data_quality_tools import detect_duplicate_records

result = detect_duplicate_records.func(
    "gold.recent_transactions",
    key_fields=['txn_id']  # or ['vendor', 'amount', 'date']
)

# Returns:
{
    'duplicate_count': 4,
    'duplicate_groups': [
        {
            'key': {'txn_id': 'txn_001'},
            'count': 2,
            'ids': ['row_1', 'row_5']
        },
        {
            'key': {'txn_id': 'txn_042'},
            'count': 2,
            'ids': ['row_42', 'row_87']
        }
    ]
}
```

**When to use**: Identify duplicate transactions before reconciliation.

---

### Tool 4: Infer Domain Freshness

```python
from src.tools.data_quality_tools import infer_domain_freshness

# Option 1: With known domain (uses config)
result = infer_domain_freshness.func({
    'domain': 'inventory_management',
    'frequency': 'daily'
})

# Option 2: Without domain (LLM inference)
result = infer_domain_freshness.func({
    'frequency': 'daily',
    'vendor_type': 'supplier',
    'avg_amount': 5000.0
})

# Returns:
{
    'domain': 'inventory_management',
    'max_age_hours': 24,
    'confidence': 1.0,
    'source': 'manual_config',  # or 'inferred'
    'critical_amount_threshold': 10000
}
```

**When to use**: Determine how fresh data needs to be for a specific domain.

---

### Tool 5: Check Data Quality Gates

```python
from src.tools.data_quality_tools import check_data_quality_gates

quality_metrics = {
    'total_records': 1000,
    'completeness_score': 0.92
}

thresholds = {
    'completeness_threshold': 0.90
}

passed = check_data_quality_gates.func(quality_metrics, thresholds)

# Returns: True (if passed) or False (if failed)
```

**When to use**: Decide if data quality is sufficient to proceed with audit.

---

## Configuration

### Domain Configuration (config/rules.yaml)

```yaml
domains:
  inventory_management:
    max_age_hours: 24
    critical_amount_threshold: 10000

  senior_living:
    max_age_hours: 48
    critical_amount_threshold: 1000

  business_operations:
    max_age_hours: 48
    critical_amount_threshold: 5000

  default:
    max_age_hours: 48
    critical_amount_threshold: 1000

quality_gates:
  completeness_threshold: 0.90  # 90% required
```

### Environment Variables

```bash
# .env file
ENVIRONMENT=development  # or production
DATABRICKS_HOST=your-workspace.databricks.com
DATABRICKS_TOKEN=your-token
OPENROUTER_API_KEY=your-key  # Only needed if domain inference with LLM
```

---

## Integration with Orchestrator

The Data Quality Agent runs FIRST in the audit pipeline:

```python
from crewai import Crew
from src.agents.data_quality_agent import data_quality_agent, data_quality_task
from src.agents.reconciliation_agent import reconciliation_agent, reconciliation_task

# Sequential execution
crew = Crew(
    agents=[data_quality_agent, reconciliation_agent],
    tasks=[data_quality_task, reconciliation_task],
    process="sequential"  # Data quality must pass first
)

result = crew.kickoff(inputs={
    "table_name": "gold.recent_transactions"
})

# If data_quality_task returns gate_passed=False, audit halts
```

---

## Error Handling

### Databricks Connection Failure

```python
# Tool automatically falls back to mock data in development
from src.tools.data_quality_tools import check_data_completeness

# If ENVIRONMENT=development, uses fixtures/mock_transactions.json
# If ENVIRONMENT=production and connection fails, raises DatabricksConnectionError
```

### LLM Failure (Tool 4)

```python
# Graceful fallback to default domain config
result = infer_domain_freshness.func({
    'frequency': 'daily',
    'vendor_type': 'unknown'
})

# If LLM fails, returns:
{
    'domain': 'default',
    'max_age_hours': 48,
    'confidence': 0.0,
    'source': 'error_fallback'
}
```

### Empty Tables

All tools handle empty tables gracefully:

```python
check_data_completeness.func("empty_table")
# Returns: {'total_records': 0, 'completeness_score': 0.0, ...}

detect_duplicate_records.func("empty_table", ['txn_id'])
# Returns: {'duplicate_count': 0, 'duplicate_groups': []}
```

---

## Performance Optimization

### For Large Datasets (>100K records)

1. **Use indexed key fields** for duplicate detection:
   ```python
   # Faster if txn_id is indexed
   detect_duplicate_records.func("large_table", ['txn_id'])
   ```

2. **Limit query scope** if possible:
   ```sql
   # In databricks_client wrapper
   SELECT * FROM gold.recent_transactions
   WHERE date >= CURRENT_DATE - INTERVAL 7 DAYS
   ```

3. **Skip LLM inference** by providing manual domain config:
   ```python
   infer_domain_freshness.func({'domain': 'default', 'frequency': 'daily'})
   # No LLM call, instant response
   ```

### Monitoring

Key metrics to track:

```python
from prometheus_client import Histogram, Counter

data_quality_check_duration = Histogram(
    'data_quality_check_duration_seconds',
    'Time spent in data quality checks'
)

quality_gate_failures = Counter(
    'quality_gate_failures_total',
    'Number of times quality gate failed'
)
```

---

## Testing

### Unit Tests

```bash
# Run all data quality tests
pytest tests/test_agents/test_data_quality_agent.py -v

# Run specific test
pytest tests/test_agents/test_data_quality_agent.py::test_check_completeness -v
```

### Integration Test

```python
from src.agents.data_quality_agent import data_quality_agent, data_quality_task
from crewai import Crew

# Test with mock data
crew = Crew(
    agents=[data_quality_agent],
    tasks=[data_quality_task],
    verbose=True
)

result = crew.kickoff(inputs={
    "table_name": "fixtures/mock_transactions.json"
})

assert result['gate_passed'] == True
```

---

## Troubleshooting

### Issue: "No module named 'crewai_tools'"

**Solution**: Install dependencies
```bash
pip install crewai==0.28.0 crewai-tools==0.2.6
```

### Issue: "DatabricksConnectionError"

**Solution**: Check environment and credentials
```bash
# Verify environment variables
echo $DATABRICKS_HOST
echo $DATABRICKS_TOKEN

# Or use development mode
export ENVIRONMENT=development
```

### Issue: "Quality gate always failing"

**Solution**: Lower threshold or improve data quality
```yaml
# config/rules.yaml
quality_gates:
  completeness_threshold: 0.80  # Lower from 0.90
```

### Issue: "LLM inference too slow"

**Solution**: Add manual domain configs
```yaml
# config/rules.yaml
domains:
  your_domain:
    max_age_hours: 48
    critical_amount_threshold: 1000
```

---

## Best Practices

1. **Always run Data Quality first** before other agents
2. **Use manual domain configs** to avoid LLM costs
3. **Set appropriate thresholds** for your data quality standards
4. **Monitor quality gate failures** to identify data issues
5. **Test with edge cases** (empty tables, all nulls, etc.)
6. **Log extensively** for debugging

---

## API Reference

### Agent

```python
data_quality_agent = Agent(
    role="Data Quality Specialist",
    goal="Validate data completeness, schema conformity, and freshness with 95%+ accuracy",
    backstory="Expert data engineer...",
    tools=[...],
    verbose=True,
    allow_delegation=False,
    llm=None
)
```

### Task

```python
data_quality_task = Task(
    description="Given a table of transactions...",
    agent=data_quality_agent,
    expected_output="JSON object with data quality report..."
)
```

### Tools

All tools follow this pattern:

```python
@tool("tool_name")
def tool_function(param: type) -> return_type:
    """Docstring"""
    # Implementation
    return result
```

---

## Additional Resources

- **Specification**: [docs/TASK_2_DATA_QUALITY_AGENT.md](TASK_2_DATA_QUALITY_AGENT.md)
- **Architecture**: [system_specs.md](../system_specs.md)
- **ADRs**: [development_history.md](../development_history.md)
- **Completion Summary**: [TASK2_COMPLETION_SUMMARY.md](../TASK2_COMPLETION_SUMMARY.md)

---

**Last Updated**: 2024-02-06
