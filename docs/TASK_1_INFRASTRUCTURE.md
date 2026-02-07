# TASK 1: Core Infrastructure Layer

## Objective
Build the foundational infrastructure components: Databricks client with mock data adapter, LLM client with cost tracking, Redis state manager, and database schema definitions.

## Context
You are building the infrastructure layer for an agentic audit system. This is a **tool-first, LLM-light architecture** where 97% of operations are deterministic (SQL/Python) and only 3% use LLM calls. The system will eventually connect to Databricks Gold tables, but for now needs a mock data adapter for local development.

## Files to Create

### 1. `/src/tools/databricks_client.py` (~150 lines)

**Purpose**: Abstract Databricks connection with automatic fallback to mock JSON data for local dev.

**Key Requirements**:
- Singleton connection pool using `@lru_cache`
- Check `DATABRICKS_HOST` and `ENVIRONMENT` env vars to determine real vs mock
- Mock adapter should parse SQL queries and return appropriate JSON fixture data
- Include retry logic with exponential backoff for real Databricks connections
- Export functions: `query_gold_tables()`, `get_last_audit_timestamp()`, `load_mock_data_from_json()`

**Implementation Details**:
```python
from functools import lru_cache
import os
import pandas as pd
from datetime import datetime
from typing import Optional
from src.utils.errors import DatabricksConnectionError
from src.utils.logging import get_logger

logger = get_logger(__name__)

@lru_cache(maxsize=1)
def get_databricks_connection():
    """
    Singleton connection pool with automatic retry
    Returns None if in dev mode (uses mock data instead)
    """
    if os.getenv("DATABRICKS_HOST") and os.getenv("ENVIRONMENT") == "production":
        try:
            from databricks import sql
            conn = sql.connect(
                server_hostname=os.getenv("DATABRICKS_HOST"),
                http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
                auth_type="pat",
                token=os.getenv("DATABRICKS_TOKEN"),
                timeout_seconds=300
            )
            logger.info("Connected to Databricks", host=os.getenv("DATABRICKS_HOST"))
            return conn
        except Exception as e:
            raise DatabricksConnectionError(f"Failed to connect to Databricks: {e}")
    else:
        logger.info("Using mock data adapter (dev mode)")
        return None

def query_gold_tables(sql_query: str) -> pd.DataFrame:
    """
    Execute SQL query with automatic fallback to mock data

    Args:
        sql_query: SQL query string

    Returns:
        DataFrame with query results

    Raises:
        DatabricksConnectionError: If real connection fails
    """
    conn = get_databricks_connection()

    if conn:
        # Real Databricks query
        try:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            result = cursor.fetchall_arrow().to_pandas()
            logger.info(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            raise DatabricksConnectionError(f"Query failed: {e}")
    else:
        # Mock data adapter
        return load_mock_data_from_json(sql_query)

def load_mock_data_from_json(sql_query: str) -> pd.DataFrame:
    """
    Mock data adapter for local development
    Parses SQL query to determine which fixture to load

    Args:
        sql_query: SQL query (used to infer which fixture to load)

    Returns:
        DataFrame from JSON fixture
    """
    import json
    from pathlib import Path

    # Parse table name from query (simple heuristic)
    query_lower = sql_query.lower()

    fixture_map = {
        "gold.recent_transactions": "tests/fixtures/sample_transactions.json",
        "gold.credit_cards": "tests/fixtures/sample_credit_cards.json",
        "gold.bank_accounts": "tests/fixtures/sample_bank_accounts.json",
        "gold.emails": "tests/fixtures/sample_emails.json",
        "workflow_state": "tests/fixtures/sample_workflow_state.json"
    }

    for table, fixture_path in fixture_map.items():
        if table in query_lower:
            try:
                return pd.read_json(fixture_path)
            except FileNotFoundError:
                logger.warning(f"Fixture not found: {fixture_path}, returning empty DataFrame")
                return pd.DataFrame()

    # Default: return empty DataFrame with expected schema
    logger.warning(f"No fixture found for query, returning empty DataFrame")
    return pd.DataFrame(columns=['txn_id', 'source', 'amount', 'vendor', 'date'])

def get_last_audit_timestamp() -> datetime:
    """
    Retrieve timestamp of last completed audit

    Returns:
        Datetime of last audit, or default (2025-01-01) if none found
    """
    try:
        result = query_gold_tables("""
            SELECT MAX(created_at) as last_audit
            FROM workflow_state
            WHERE workflow_status = 'completed'
        """)
        if not result.empty and result['last_audit'][0]:
            return pd.to_datetime(result['last_audit'][0])
    except Exception as e:
        logger.warning(f"Could not get last audit timestamp: {e}")

    return datetime(2025, 1, 1)

def check_databricks_health() -> bool:
    """
    Check if Databricks connection is healthy

    Returns:
        True if connection is healthy, False otherwise
    """
    try:
        conn = get_databricks_connection()
        if conn:
            query_gold_tables("SELECT 1")
            return True
        return False  # Mock mode, always "healthy"
    except Exception:
        return False
```

**Dependencies**:
- `databricks-sql-connector` (optional, only needed in production)
- `pandas`
- `src.utils.errors`
- `src.utils.logging`

---

### 2. `/src/tools/llm_client.py` (~120 lines)

**Purpose**: OpenRouter LLM client with automatic cost tracking and model selection.

**Key Requirements**:
- Support OpenAI API via OpenRouter (compatible API)
- Automatic cost calculation based on token usage
- Prometheus metrics integration
- Model tiering: GPT-4o-mini (cheap, default) vs Claude Haiku (edge cases)
- Error handling with retries

**Implementation Details**:
```python
import openai
import os
import time
from typing import Optional
from src.utils.metrics import llm_tokens_counter, llm_cost_counter, llm_api_latency
from src.utils.errors import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Configure OpenRouter
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# Pricing per 1M tokens (input tokens, simplified)
MODEL_PRICING = {
    "openai/gpt-4o-mini": 0.15 / 1_000_000,
    "anthropic/claude-3-5-haiku": 0.80 / 1_000_000,
    "openai/gpt-4-turbo": 10.0 / 1_000_000  # Expensive, avoid unless necessary
}

def call_llm(
    prompt: str,
    model: Optional[str] = None,
    agent_name: str = "unknown",
    max_retries: int = 3
) -> str:
    """
    Call LLM with automatic cost tracking and retries

    Args:
        prompt: User prompt
        model: Model name (defaults to GPT-4o-mini)
        agent_name: Name of agent calling LLM (for metrics)
        max_retries: Max retry attempts

    Returns:
        LLM response text

    Raises:
        LLMError: If API call fails after retries
    """
    model = model or os.getenv("GPT4O_MINI_MODEL", "openai/gpt-4o-mini")

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )

            # Track metrics
            latency = time.time() - start_time
            tokens = response['usage']['total_tokens']
            cost = calculate_cost(tokens, model)

            llm_tokens_counter.labels(model_name=model, agent_name=agent_name).inc(tokens)
            llm_cost_counter.labels(model_name=model).inc(cost)
            llm_api_latency.labels(model_name=model).observe(latency)

            logger.info(
                f"LLM call successful",
                model=model,
                tokens=tokens,
                cost=cost,
                latency=latency,
                agent=agent_name
            )

            return response['choices'][0]['message']['content']

        except openai.error.RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying... (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise LLMError(f"Rate limit exceeded after {max_retries} attempts: {e}")

        except Exception as e:
            logger.error(f"LLM API error: {e}", attempt=attempt)
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise LLMError(f"LLM API call failed after {max_retries} attempts: {e}")

def calculate_cost(tokens: int, model: str) -> float:
    """
    Calculate cost based on token usage and model pricing

    Args:
        tokens: Number of tokens used
        model: Model name

    Returns:
        Cost in USD
    """
    price_per_token = MODEL_PRICING.get(model, 0.15 / 1_000_000)
    return tokens * price_per_token

def batch_call_llm(prompts: list[str], model: Optional[str] = None) -> list[str]:
    """
    Batch LLM calls for efficiency (processes sequentially but with shared setup)

    Args:
        prompts: List of prompts
        model: Model name

    Returns:
        List of responses
    """
    return [call_llm(prompt, model) for prompt in prompts]
```

**Dependencies**:
- `openai` (v1.10.0 or compatible)
- `src.utils.metrics`
- `src.utils.errors`

---

### 3. `/src/orchestrator/state_manager.py` (~100 lines)

**Purpose**: Redis-backed state management for workflow resumability.

**Key Requirements**:
- Save/restore workflow state to Redis
- 24-hour TTL on state entries
- JSON serialization
- Fallback to empty state if Redis unavailable
- Export functions: `save_workflow_state()`, `restore_workflow_state()`, `mark_audit_complete()`

**Implementation Details**:
```python
import redis
import json
import os
from typing import Dict, Any
from datetime import datetime
from src.utils.errors import StateManagerError
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Redis client setup
try:
    redis_host, redis_port = os.getenv("REDIS_HOST", "localhost:6379").split(':')
    redis_client = redis.Redis(
        host=redis_host,
        port=int(redis_port),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
        socket_keepalive=True,
        socket_connect_timeout=5
    )
    redis_client.ping()  # Test connection
    logger.info("Connected to Redis", host=redis_host, port=redis_port)
except Exception as e:
    logger.warning(f"Redis connection failed, state persistence disabled: {e}")
    redis_client = None

def save_workflow_state(audit_run_id: str, state: Dict[str, Any]) -> None:
    """
    Save workflow state to Redis with 24-hour TTL

    Args:
        audit_run_id: Unique audit run ID
        state: State dictionary to save

    Raises:
        StateManagerError: If save fails
    """
    if not redis_client:
        logger.warning("Redis unavailable, state not saved")
        return

    try:
        key = f"audit:{audit_run_id}:state"
        value = json.dumps(state, default=str)  # default=str handles datetime
        redis_client.setex(key, 86400, value)  # 24 hour TTL
        logger.info(f"Saved workflow state", audit_run_id=audit_run_id)
    except Exception as e:
        raise StateManagerError(f"Failed to save workflow state: {e}")

def restore_workflow_state(audit_run_id: str) -> Dict[str, Any]:
    """
    Restore workflow state from Redis

    Args:
        audit_run_id: Unique audit run ID

    Returns:
        State dictionary, or empty dict if not found
    """
    if not redis_client:
        logger.warning("Redis unavailable, returning empty state")
        return {}

    try:
        key = f"audit:{audit_run_id}:state"
        value = redis_client.get(key)

        if value:
            state = json.loads(value)
            logger.info(f"Restored workflow state", audit_run_id=audit_run_id)
            return state
        else:
            logger.warning(f"No saved state found for {audit_run_id}")
            return {}

    except Exception as e:
        logger.error(f"Failed to restore workflow state: {e}")
        return {}

def mark_audit_complete(audit_run_id: str, summary: Dict[str, Any]) -> None:
    """
    Mark audit as completed and save final summary

    Args:
        audit_run_id: Unique audit run ID
        summary: Final summary data
    """
    state = {
        'status': 'completed',
        'summary': summary,
        'completed_at': datetime.now().isoformat()
    }
    save_workflow_state(audit_run_id, state)

def check_redis_health() -> bool:
    """
    Check if Redis connection is healthy

    Returns:
        True if Redis is reachable, False otherwise
    """
    if not redis_client:
        return False

    try:
        redis_client.ping()
        return True
    except Exception:
        return False
```

**Dependencies**:
- `redis`
- `src.utils.errors`
- `src.utils.logging`

---

### 4. `/src/db/schemas.py` (~200 lines)

**Purpose**: SQL schema definitions for Flag DB, Audit Trail, and State Store.

**Key Requirements**:
- 3 table schemas: `flags`, `audit_trail`, `workflow_state`
- Include CREATE TABLE statements
- Add indexes for common queries
- Support both Postgres and Databricks SQL dialects

**Implementation Details**:
```python
"""SQL schemas for audit database tables"""

# Flags table - stores audit flags for human review
FLAGS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS flags (
    flag_id VARCHAR(36) PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL,
    audit_run_id VARCHAR(36) NOT NULL,
    severity_level VARCHAR(20) CHECK (severity_level IN ('CRITICAL', 'WARNING', 'INFO')),
    confidence_score FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
    explanation TEXT NOT NULL,
    supporting_evidence_links TEXT,  -- JSON stored as text
    reviewed BOOLEAN DEFAULT FALSE,
    human_decision VARCHAR(30),
    reviewer_id VARCHAR(255),
    review_timestamp TIMESTAMP,
    reviewer_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_flags_txn_id ON flags(transaction_id);
CREATE INDEX IF NOT EXISTS idx_flags_audit_run ON flags(audit_run_id);
CREATE INDEX IF NOT EXISTS idx_flags_reviewed ON flags(reviewed, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flags_severity ON flags(severity_level, created_at DESC);
"""

# Audit trail table - immutable append-only log
AUDIT_TRAIL_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_trail (
    audit_run_id VARCHAR(36) NOT NULL,
    log_sequence_number BIGINT NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    tool_called VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_time_ms BIGINT,
    input_data TEXT,  -- JSON stored as text
    output_summary TEXT,  -- JSON stored as text
    llm_model VARCHAR(100),
    llm_tokens_used BIGINT,
    llm_cost_dollars DECIMAL(10, 4),
    error_message TEXT,
    error_stack_trace TEXT,
    decision_chain TEXT,  -- JSON array stored as text
    PRIMARY KEY (audit_run_id, log_sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_audit_trail_agent ON audit_trail(agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
"""

# Workflow state table - tracks audit run state
WORKFLOW_STATE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_state (
    audit_run_id VARCHAR(36) PRIMARY KEY,
    workflow_status VARCHAR(20) CHECK (workflow_status IN ('pending', 'in_progress', 'completed', 'failed', 'paused')),
    current_agent VARCHAR(100),
    completed_agents TEXT,  -- JSON array
    pending_agents TEXT,  -- JSON array
    intermediate_results TEXT,  -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflow_state(workflow_status, created_at DESC);
"""

# Helper function to create all tables
def create_all_tables(cursor):
    """
    Execute all CREATE TABLE statements

    Args:
        cursor: Database cursor object
    """
    cursor.execute(FLAGS_TABLE_SCHEMA)
    cursor.execute(AUDIT_TRAIL_TABLE_SCHEMA)
    cursor.execute(WORKFLOW_STATE_TABLE_SCHEMA)
```

**Note**: These are standard SQL schemas. Adjust for Databricks-specific syntax when deploying to production.

---

### 5. Create Sample Fixture Data

**Files to create in `/tests/fixtures/`**:

`sample_transactions.json`:
```json
[
  {
    "txn_id": "cc_12345",
    "source": "credit_card",
    "amount": 500.00,
    "vendor": "AWS",
    "vendor_id": "aws_infrastructure",
    "date": "2025-02-03T10:00:00Z",
    "domain": "business_operations"
  },
  {
    "txn_id": "cc_12346",
    "source": "credit_card",
    "amount": 45.67,
    "vendor": "Starbucks",
    "vendor_id": "starbucks_corporate",
    "date": "2025-02-03T11:30:00Z",
    "domain": "business_operations"
  },
  {
    "txn_id": "cc_12347",
    "source": "credit_card",
    "amount": 1250.00,
    "vendor": "Unknown Corp",
    "vendor_id": null,
    "date": "2025-02-03T14:00:00Z",
    "domain": "business_operations"
  }
]
```

`sample_credit_cards.json`, `sample_bank_accounts.json`, `sample_emails.json` - similar structure.

---

## Testing Requirements

Create unit tests in `/tests/test_infrastructure.py`:
```python
def test_databricks_mock_adapter():
    """Test that mock data adapter works"""
    from src.tools.databricks_client import query_gold_tables
    result = query_gold_tables("SELECT * FROM gold.recent_transactions")
    assert not result.empty
    assert 'txn_id' in result.columns

def test_llm_client():
    """Test LLM client with simple prompt"""
    from src.tools.llm_client import call_llm
    result = call_llm("Say 'test'")
    assert isinstance(result, str)

def test_state_manager():
    """Test Redis state save/restore"""
    from src.orchestrator.state_manager import save_workflow_state, restore_workflow_state
    state = {'test': 'data'}
    save_workflow_state('test_run_123', state)
    restored = restore_workflow_state('test_run_123')
    assert restored == state
```

---

## Success Criteria

✅ Databricks client can query mock data without real Databricks connection
✅ LLM client successfully makes API calls and tracks costs
✅ State manager saves/restores to Redis (or gracefully fails if Redis down)
✅ Database schemas are valid SQL
✅ All imports work (no circular dependencies)
✅ Unit tests pass

---

## Important Notes

- **Do NOT install packages** - assume requirements.txt already installed
- Use existing modules in `src/utils/` (errors, logging, metrics, config_loader)
- All file paths are relative to project root `/Users/kittenoverlord/projects/ergonosis_auditing/`
- Import convention: `from src.utils.errors import DatabricksConnectionError`
- Log extensively using the structured logger
- Handle all errors gracefully (no silent failures)

---

## Estimated Effort
~400 lines of code total, should take 1-2 hours for an experienced developer.
