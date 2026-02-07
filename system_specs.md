## System Overview

**Core Philosophy**: Tool-heavy, LLM-light architecture that minimizes API costs while maximizing audit quality through intelligent agent orchestration.

**Key Innovations**:

- Automatic domain inference with manual override capability
- Hybrid cost optimization (deterministic tools → cheap LLMs → expensive LLMs)
- Batch processing with memory preservation to avoid catastrophic forgetting
- Knowledge graph integration for cross-source entity resolution
- Automatic rule tuning based on false positive feedback
- Full trace-level logging for transparency and debugging

---

## Agent Specifications

### **🎯 Orchestrator Agent (CrewAI Manager)**

**Role**: Master coordinator that manages workflow, delegates tasks, handles failures

**Responsibilities**:

- Trigger audit cycle (hourly, event-based, or manual)
- Query Databricks Gold tables for transactions to audit
- Dispatch work to sub-agents (parallel where possible)
- Aggregate results from all agents
- Handle graceful degradation if Databricks goes down
- Manage retry logic with exponential backoff (30s, 1min, 2min, 4min, 8min)
- Track state in Redis/Databricks for resumability

**Key Decisions**:

- Which transactions to audit (all new since last run, or specific subset)
- Whether to run agents in parallel (Data Quality + Reconciliation + Anomaly) or sequentially
- When to invoke expensive LLMs vs cheap ones vs no LLM at all
- Whether audit is "complete" or needs human intervention

**Tools**:

```python
# Query tools
query_gold_tables(sql: str) → DataFrame
get_knowledge_graph_entities(entity_type: str) → List[Entity]
get_last_audit_timestamp() → datetime

# State management
save_workflow_state(audit_run_id: str, state: dict) → None
restore_workflow_state(audit_run_id: str) → dict
mark_audit_complete(audit_run_id: str, summary: dict) → None

# Agent coordination
dispatch_parallel_agents(agents: List[Agent], data: dict) → List[Result]
wait_for_completion(timeout: int) → bool
handle_agent_failure(agent_name: str, error: Exception) → RetryDecision

```

**LLM Usage**: **GPT-4o-mini** for high-level orchestration decisions only (e.g., "Should I retry this agent or skip?")

**Cost per Audit**: ~$0.001 (1 orchestration call)

---

### **📊 Data Quality Agent**

**Role**: Validate data completeness, schema conformity, and freshness before auditing

**Responsibilities**:

- Check if required fields are populated (vendor, amount, date, source)
- Validate schema (correct data types, no malformed records)
- Detect duplicate records within same source
- Infer domain-specific freshness requirements (or use manual override)
- Apply data quality gates (e.g., "if >10% records incomplete, halt audit")

**Tools**:

```python
# Completeness checks (pure SQL, no LLM)
check_data_completeness(table_name: str) → dict:
    """
    Returns: {
        'total_records': 10000,
        'missing_vendor': 50,
        'missing_amount': 10,
        'missing_date': 5,
        'completeness_score': 0.993
    }
    """

validate_schema_conformity(table_name: str, expected_schema: dict) → List[ValidationError]

detect_duplicate_records(table_name: str, key_fields: List[str]) → DataFrame:
    """Returns DataFrame of duplicate records"""

# Domain inference (lightweight LLM or rule-based)
infer_domain_freshness(transaction_pattern: dict) → dict:
    """
    Input: {
        'frequency': 'daily',
        'vendor_type': 'inventory_supplier',
        'avg_amount': 500
    }
    Output: {
        'domain': 'inventory_management',
        'max_age_hours': 24,
        'confidence': 0.85
    }

    Uses manual override if exists in config, else infers
    """

check_data_quality_gates(quality_metrics: dict, thresholds: dict) → bool:
    """
    Returns True if quality passes, False if audit should halt
    Example threshold: completeness_score > 0.90
    """

```

**LLM Usage**: **GPT-4o-mini** only for domain inference if no manual config exists (~200 calls/month)

**Cost per Audit**: ~$0.002 (domain inference only)

**Output Schema**:

```python
{
    'quality_score': 0.95,
    'incomplete_records': [record_id_1, record_id_2],
    'schema_violations': [],
    'duplicates': [{'record_1': id_x, 'record_2': id_y}],
    'domain_config': {'max_age_hours': 48, 'source': 'inferred'},
    'freshness_violations': [],
    'gate_passed': True
}

```

---

### **🔗 Reconciliation Agent**

**Role**: Match transactions across credit cards, bank accounts, emails, receipts, and aggregator system

**Responsibilities**:

- Cross-source matching (credit card charge = bank debit?)
- Entity resolution via knowledge graph (vendor name variations)
- Fuzzy matching for vendor names with typos
- Receipt-to-transaction matching (OCR amount + date range)
- Identify orphan transactions (appear in one source but not others)

**Tools**:

```python
# Cross-source matching (deterministic SQL + KG lookup, no LLM)
cross_source_matcher(source_1: str, source_2: str, date_range: tuple) → DataFrame:
    """
    Matches transactions from two sources based on:
    - Amount (exact or ±5% for currency conversion)
    - Date (±3 days window)
    - Vendor (via KG entity resolution)

    Returns DataFrame with columns:
    - source_1_id, source_2_id, match_confidence (0-1), match_reason
    """

entity_resolver_kg(vendor_name: str) → dict:
    """
    Query knowledge graph to resolve vendor entity

    Input: "AMZN MKTP US*1A2B3C4D5"
    Output: {
        'canonical_entity_id': 'amazon_marketplace',
        'canonical_name': 'Amazon Marketplace',
        'aliases': ['AMZN MKTP', 'Amazon.com', 'AMAZON MKTPLACE'],
        'confidence': 0.95
    }
    """

fuzzy_vendor_matcher(vendor_a: str, vendor_b: str) → float:
    """
    Uses Levenshtein distance + semantic embeddings
    Returns similarity score 0-1

    Example: fuzzy_match("Starbucks", "Star bucks") → 0.92
    """

receipt_transaction_matcher(receipt_data: dict, transactions: DataFrame) → dict:
    """
    Matches OCR-extracted receipt to credit card transaction

    Input receipt_data: {
        'vendor': 'Starbucks',
        'amount': 15.47,
        'date': '2025-02-01'
    }

    Returns: {
        'matched_transaction_id': 'txn_12345',
        'confidence': 0.88,
        'amount_delta': 0.00,  # exact match
        'date_delta_days': 0
    }
    """

find_orphan_transactions(all_sources: List[DataFrame]) → DataFrame:
    """
    Returns transactions that appear in only one source (suspicious)
    """

```

**LLM Usage**: **None** - all deterministic matching via SQL + KG + embeddings

**Cost per Audit**: $0 (no LLM calls)

**Output Schema**:

```python
{
    'matched_pairs': [
        {'source_1': 'credit_card', 'source_2': 'bank', 'txn_id': 'x', 'confidence': 0.95},
        ...
    ],
    'unmatched_transactions': [
        {'txn_id': 'y', 'source': 'credit_card', 'amount': 500, 'vendor': 'Unknown Inc'},
        ...  # These are SUSPICIOUS, pass to Escalation Agent
    ],
    'low_confidence_matches': [
        {'txn_id': 'z', 'confidence': 0.65, 'reason': 'vendor name mismatch'},
        ...  # Pass to Context Enrichment Agent for more investigation
    ]
}

```

---

### **🚨 Anomaly Detection Agent**

**Role**: Detect statistical and ML-based anomalies in transaction patterns

**Responsibilities**:

- Run Isolation Forest on transaction features
- Check vendor spending profiles (is this amount typical for this vendor?)
- Detect amount outliers (z-score >3)
- Time-series deviation for recurring transactions
- Batch-score all transactions for efficiency

**Tools**:

```python
# ML-based anomaly detection (pre-trained models, no LLM)
run_isolation_forest(transactions: DataFrame) → DataFrame:
    """
    Features used:
    - amount (log-scaled)
    - vendor_id (encoded)
    - day_of_week, day_of_month
    - time_since_last_transaction_from_vendor
    - amount_deviation_from_vendor_avg

    Returns DataFrame with anomaly_score column (-1 to +1)
    -1 = strong outlier, +1 = normal
    """

check_vendor_spending_profile(vendor_id: str, amount: float) → dict:
    """
    Retrieves vendor stats from cache/DB:
    - mean_amount, std_dev
    - typical_day_of_month (for recurring charges)
    - frequency (txns per month)

    Returns: {
        'is_outlier': True,
        'z_score': 3.2,
        'vendor_mean': 50,
        'vendor_std': 10,
        'explanation': 'Amount $82 is 3.2σ above mean $50'
    }
    """

detect_amount_outliers(transactions: DataFrame) → DataFrame:
    """
    Simple statistical check: flag if amount > mean + 2*std_dev
    """

time_series_deviation_check(recurring_txns: DataFrame) → DataFrame:
    """
    For recurring transactions (rent, subscriptions):
    - Uses Prophet model (pre-trained) to predict expected amount
    - Flags if actual amount outside 80% confidence interval
    - Flags if payment >5 days late
    """

batch_anomaly_scorer(transactions: DataFrame) → DataFrame:
    """
    Combines all anomaly signals into single score 0-100
    Higher score = more suspicious
    """

```

**LLM Usage**: **None** - all statistical/ML models

**Cost per Audit**: $0 (no LLM calls)

**Output Schema**:

```python
{
    'anomaly_scores': [
        {'txn_id': 'x', 'score': 85, 'reasons': ['isolation_forest', 'vendor_outlier']},
        {'txn_id': 'y', 'score': 65, 'reasons': ['time_series_late']},
        ...
    ],
    'flagged_transactions': [
        # Transactions with score >70 go here
    ]
}

```

---

### **🔍 Context Enrichment Agent**

**Role**: Search for supporting documentation (emails, calendar, receipts) to validate transactions

**Responsibilities**:

- Search emails for mentions of vendor/amount/date
- Search calendar for events matching transaction date
- Extract approval chains from email threads
- Find receipt images from cloud storage or email attachments
- Semantic search across documents for context

**Tools**:

```python
# Search tools (batch processing to minimize LLM calls)
search_emails_batch(transactions: List[dict]) → dict:
    """
    Batch search for 50-100 transactions at once
    Uses Databricks SQL to query email table (already ingested)

    Input: [
        {'txn_id': 'x', 'vendor': 'AWS', 'amount': 500, 'date': '2025-02-01'},
        ...
    ]

    Returns: {
        'txn_x': {
            'email_matches': [
                {'email_id': 'e1', 'subject': 'AWS Invoice', 'confidence': 0.9}
            ]
        },
        'txn_y': {'email_matches': []},
        ...
    }
    """

search_calendar_events(transaction_date: date, vendor: str) → List[dict]:
    """
    SQL query against calendar table
    Returns matching events ±3 days from transaction
    """

extract_approval_chains(email_thread_id: str) → dict:
    """
    Parses email thread to find approval keywords:
    "approved", "authorized", "go ahead", etc.

    Returns: {
        'approved': True,
        'approver': 'john@company.com',
        'timestamp': '2025-01-30 14:23:00'
    }
    """

find_receipt_images(vendor: str, amount: float, date_range: tuple) → List[str]:
    """
    Search cloud storage (S3/Databricks) for receipt images
    Uses OCR metadata if already processed
    Returns list of file paths
    """

semantic_search_documents(query: str, top_k: int = 5) → List[dict]:
    """
    Uses embedding-based search (Sentence Transformers)
    Query: "AWS server upgrade approval"
    Returns relevant documents from email/Slack/Notion

    Only called for high-priority unmatched transactions
    """

```

**LLM Usage**: **GPT-4o-mini** for semantic search on <5% of transactions (~500 calls/month)

**Cost per Audit**: ~$0.01 (semantic search for complex cases)

**Output Schema**:

```python
{
    'enriched_transactions': [
        {
            'txn_id': 'x',
            'email_approval': True,
            'calendar_event': {'event_id': 'cal_123', 'title': 'Client dinner'},
            'receipt_found': True,
            'confidence': 0.95
        },
        {
            'txn_id': 'y',
            'email_approval': False,
            'calendar_event': None,
            'receipt_found': False,
            'confidence': 0.10  # SUSPICIOUS - no supporting docs
        },
        ...
    ]
}

```

---

### **⚖️ Escalation & Classification Agent**

**Role**: Classify transaction severity and generate human-readable explanations

**Responsibilities**:

- Calculate severity score based on all previous agent outputs
- Classify into CRITICAL / WARNING / INFO
- Generate root cause analysis (why was this flagged?)
- Batch-classify with LLM for complex cases
- Create flags in database with full context

**Tools**:

```python
# Severity calculation (rule-based + LLM for edge cases)
calculate_severity_score(transaction: dict, agent_results: dict) → dict:
    """
    Inputs from all agents:
    - Data quality: completeness_score
    - Reconciliation: matched (bool), confidence
    - Anomaly: anomaly_score
    - Context: email_approval, receipt_found

    Rule-based scoring:
    score = 0
    if not matched: score += 40
    if anomaly_score > 70: score += 30
    if not email_approval: score += 20
    if not receipt_found: score += 10

    Returns: {
        'severity_score': 85,  # 0-100
        'level': 'CRITICAL',
        'confidence': 0.90
    }
    """

generate_root_cause_analysis(transaction: dict, agent_results: dict) → str:
    """
    Deterministic template for simple cases:
    "Flagged because: (1) No matching bank transaction found,
    (2) Amount $847 is 3.2σ above vendor average,
    (3) No email approval located,
    (4) Receipt not found"

    LLM-based for complex cases (only if confidence <0.7):
    Uses Claude Haiku to synthesize nuanced explanation
    """

batch_classify_with_llm(transactions: List[dict], agent_results: List[dict]) → List[dict]:
    """
    For transactions where rule-based classifier is uncertain (score 40-60),
    batch-call LLM with context from all agents

    Prompt template:
    "Here are 50 transactions with anomaly data, reconciliation status,
    and context. For each, classify as CRITICAL/WARNING/INFO and explain why."

    Uses Claude Haiku ($0.25/1M tokens) for cost efficiency
    Max 100 transactions per batch to fit in context window
    """

create_audit_flag(transaction_id: str, severity: str, explanation: str, evidence: dict) → str:
    """
    Writes to Flag Database with full context
    Returns flag_id (UUID)
    """

check_escalation_rules(severity: str, amount: float, vendor: str) → str:
    """
    Domain-specific rules:
    - If severity=CRITICAL and amount >$1000 → escalate to CFO
    - If vendor in whitelist and severity=WARNING → downgrade to INFO
    - If recurring vendor and <5% deviation → downgrade

    Returns updated severity level
    """

```

**LLM Usage**: **Claude Haiku** for batch classification of edge cases (~1000 calls/month)

**Cost per Audit**: ~$0.03 (batch LLM calls for ~10% of transactions)

**Output Schema**:

```python
{
    'flags': [
        {
            'flag_id': 'uuid-1234',
            'txn_id': 'x',
            'severity': 'CRITICAL',
            'confidence': 0.92,
            'explanation': 'Unauthorized charge: No matching bank transaction...',
            'evidence': {
                'reconciliation_match': False,
                'anomaly_score': 85,
                'email_approval': False,
                'receipt_found': False
            },
            'lineage': ['DataQuality→ Reconciliation→ Anomaly→ Context→ Escalation']
        },
        ...
    ]
}

```

---

### **📝 Audit Logging Agent**

**Role**: Record every decision, tool call, and state change for transparency and compliance

**Responsibilities**:

- Log every agent decision with full context
- Create immutable audit trail entries
- Save workflow state for resumability
- Generate lineage graph (data provenance)
- Archive logs for compliance (7-year retention for financial audits)

**Tools**:

```python
# Logging tools (append-only database writes)
log_agent_decision(agent_name: str, action: str, input_data: dict, output_data: dict, metadata: dict) → None:
    """
    Writes to Audit Trail DB:
    {
        'audit_run_id': 'run_5432',
        'timestamp': '2025-02-03 10:23:45',
        'agent_name': 'Reconciliation',
        'tool_called': 'cross_source_matcher',
        'input_sample': {'source_1': 'credit_card', 'source_2': 'bank'},
        'output_summary': {'matched': 850, 'unmatched': 15},
        'llm_tokens_used': 0,
        'llm_cost': 0.0,
        'execution_time_ms': 1234
    }
    """

create_audit_trail_entry(flag_id: str, decision_chain: List[dict]) → None:
    """
    Links flag to all agent decisions that led to it
    Example decision_chain:
    [
        {'agent': 'Reconciliation', 'finding': 'No match found'},
        {'agent': 'Anomaly', 'finding': 'Outlier score 0.85'},
        {'agent': 'Context', 'finding': 'No email approval'},
        {'agent': 'Escalation', 'decision': 'CRITICAL'}
    ]
    """

save_workflow_state(audit_run_id: str, state: dict) → None:
    """
    Saves to Redis or Databricks table for graceful degradation
    State includes:
    - Completed agents
    - Pending agents
    - Intermediate results
    - Timestamp of last update

    If Databricks crashes, can resume from last saved state
    """

generate_lineage_trace(transaction_id: str) → dict:
    """
    Creates data lineage graph:
    "Transaction txn_123 from Bronze.credit_cards row 456
    → matched to Silver.bank_accounts row 789
    → flagged by Anomaly agent (ISO Forest score -0.8)
    → enriched with email thread email_999
    → classified as CRITICAL by Escalation agent"

    Returns graph structure (nodes + edges)
    """

archive_to_immutable_log(audit_run_id: str) → None:
    """
    After audit completes, copy all logs to immutable storage
    (Delta Lake with time travel, or S3 with object lock)
    For compliance: 7-year retention for financial audits
    """

```

**LLM Usage**: **None** - pure logging

**Cost per Audit**: $0

**Output**: All writes to Audit Trail Database (append-only)

---

## Tool Catalog Summary

### **Cost-Optimized Tool Architecture**

| **Tool Category**           | **Example Tools**                                                   | **LLM Required?**              | **Cost Impact** |
| --------------------------- | ------------------------------------------------------------------- | ------------------------------ | --------------- |
| **Data Access**             | `query_gold_tables`, `get_knowledge_graph_entities`                 | ❌ No                          | $0              |
| **Data Quality**            | `check_completeness`, `validate_schema`, `detect_duplicates`        | ❌ No                          | $0              |
| **Reconciliation**          | `cross_source_matcher`, `entity_resolver_kg`, `fuzzy_matcher`       | ❌ No (uses embeddings)        | $0              |
| **Anomaly Detection**       | `run_isolation_forest`, `check_vendor_profile`, `time_series_check` | ❌ No (pre-trained ML)         | $0              |
| **Context Enrichment**      | `search_emails_batch`, `search_calendar`, `find_receipts`           | ⚠️ Rare (semantic search <5%)  | ~$0.01/audit    |
| **Severity Classification** | `calculate_severity`, `batch_classify_with_llm`                     | ⚠️ Sometimes (~10% edge cases) | ~$0.03/audit    |
| **Explanation Generation**  | `generate_root_cause_analysis`                                      | ⚠️ Sometimes (complex cases)   | ~$0.02/audit    |
| **Logging**                 | `log_decision`, `create_audit_trail`, `save_state`                  | ❌ No                          | $0              |

**Total Cost per Audit**: ~$0.06 - $0.10 (well under $100/month for 1000 audits)

**LLM Token Budget**:

- 1000 audits/month × $0.08/audit = **$80/month** (within budget!)
- Headroom for growth: Can scale to 1200 audits before hitting $100 limit

---

## Workflow Execution Example

### **Sample Audit Cycle (1000 transactions)**

```
10:00:00 - Hourly trigger fires (Databricks Workflows cron)
10:00:01 - Orchestrator queries Gold tables: 1000 new transactions since last run
10:00:02 - Orchestrator creates audit_run_id: "run_5432"
10:00:03 - Orchestrator dispatches parallel agents:

[PARALLEL EXECUTION - 10:00:03 to 10:02:15]
├─ Data Quality Agent (30 seconds)
│  ├─ check_completeness(): 995/1000 complete (99.5%)
│  ├─ validate_schema(): 0 errors
│  ├─ detect_duplicates(): 3 duplicates found
│  ├─ infer_domain_freshness(): "business_operations", max_age=48h
│  └─ Result: Quality score 0.99, gate PASSED
│
├─ Reconciliation Agent (2 minutes)
│  ├─ cross_source_matcher(credit_card, bank): 850 matches, 150 unmatched
│  ├─ entity_resolver_kg() for 150 unmatched: 120 resolved, 30 still unmatched
│  ├─ fuzzy_vendor_matcher(): 10 additional matches
│  └─ Result: 880 matched, 20 orphan transactions (SUSPICIOUS)
│
└─ Anomaly Detection Agent (1.5 minutes)
   ├─ run_isolation_forest(1000 txns): 45 flagged (score <-0.5)
   ├─ check_vendor_profile(): 12 additional outliers
   ├─ time_series_deviation(): 5 late payments
   └─ Result: 62 anomalies detected

10:02:16 - Orchestrator receives parallel results
10:02:17 - Orchestrator identifies 82 suspicious transactions (20 orphans + 62 anomalies)
10:02:18 - Orchestrator dispatches Context Enrichment Agent (sequential)

[SEQUENTIAL EXECUTION - 10:02:18 to 10:05:30]
Context Enrichment Agent (3 minutes)
├─ search_emails_batch(82 txns): 60 have email matches
├─ search_calendar(): 35 have calendar events
├─ extract_approval_chains(): 45 have approvals
├─ find_receipt_images(): 50 have receipts
├─ semantic_search() for 10 high-priority unmatched: 8 found context
└─ Result: 27 transactions still lack supporting docs

10:05:31 - Orchestrator dispatches Escalation Agent

[SEQUENTIAL EXECUTION - 10:05:31 to 10:07:45]
Escalation Agent (2 minutes)
├─ calculate_severity_score(82 txns):
│  ├─ 5 CRITICAL (no match + no approval + high amount)
│  ├─ 22 WARNING (anomaly + missing receipt)
│  └─ 55 INFO (minor discrepancies)
├─ generate_root_cause() for 27 txns: 20 deterministic, 7 need LLM
├─ batch_classify_with_llm(7 edge cases): Claude Haiku call
│  └─ Reclassifies 2 as WARNING, 5 as INFO
├─ create_audit_flag() for 27 txns: writes to Flag DB
└─ Result: 5 CRITICAL, 22 WARNING flags created

10:07:46 - Audit Logging Agent (continuous background logging throughout)
└─ Writes 1000+ log entries to Audit Trail DB

10:07:47 - Orchestrator marks audit complete
10:07:48 - Frontend displays 27 new flags to finance team

[COST BREAKDOWN]
- LLM calls: 8 (1 orchestration + 7 edge case classifications)
- Tokens used: ~15,000 (Claude Haiku)
- Cost: $0.004 (well under budget)

[PERFORMANCE]
- Total time: 7 minutes 48 seconds (well under 30min SLA)
- Transactions processed: 1000
- Flags created: 27 (2.7% flag rate)

```

---

## Cost Optimization Deep Dive

### **How We Stay Under $100/month per Client**

**Assumptions**:

- 10,000 transactions/month per client
- 3% flag rate (300 transactions need deeper analysis)
- 10% of flagged items need LLM (30 LLM calls per run)

**Daily Audit** (assuming 500 txns/day):

```
Tool-based filtering: 500 txns → 15 suspicious (3%)
Context enrichment: 15 txns × $0.0002 (Haiku) = $0.003
Edge case classification: 2 txns × $0.001 (Haiku) = $0.002
Total: $0.005/day

Monthly: $0.005 × 30 days = $0.15/month

```

**BUT we also have:**

- Weekly feedback analysis: $0.50/month (batch job)
- Monthly model retraining: $1.00/month (one-time large job)

**Total Monthly Cost**: ~$2/month per client 🎉

**Headroom**: $98/month to spare! Can support:

- 10x transaction volume (100k txns/month)
- Higher flag rate (up to 30%)
- More LLM-heavy analysis

---

## Production Deployment Recommendations

### **Option 1: Databricks Workflows (Recommended)**

**Pros**:

- Native integration with Delta Lake
- Built-in scheduler (cron jobs)
- Can run Python/SQL directly
- Automatic scaling

**Cons**:

- Debugging is harder (less visibility into agent execution)
- Databricks compute costs (~$100-200/month)

**Setup**:

```python
# databricks_workflow.py
from crewai import Crew, Agent, Task
from databricks import sql

# Define agents (as shown above)
orchestrator = Agent(...)
data_quality = Agent(...)
# ... etc

# Create crew
crew = Crew(
    agents=[orchestrator, data_quality, reconciliation, ...],
    tasks=[...],
    process=Process.hierarchical,  # Orchestrator manages sub-agents
    manager_llm="gpt-4o-mini"
)

# Schedule in Databricks Workflows
# Trigger: Cron("0 * * * *")  # Hourly
# Cluster: Job cluster (auto-terminate after run)

```

**Cost**: ~$150/month (Databricks compute + LLM)

---

### **Option 2: Modal.com (Alternative)**

**Pros**:

- Serverless Python (no infrastructure management)
- No cold starts
- Great local dev experience
- Built-in scheduling

**Cons**:

- Need to connect to Databricks via JDBC (extra latency)
- Separate platform to manage

**Setup**:

```python
# modal_app.py
import modal
from crewai import Crew

stub = modal.Stub("audit-agent")

@stub.function(
    schedule=modal.Cron("0 * * * *"),  # Hourly
    timeout=1800,  # 30min max
    secrets=[modal.Secret.from_name("databricks-token")]
)
def run_audit():
    # Same CrewAI code as above
    crew = Crew(...)
    crew.kickoff()

@stub.local_entrypoint()
def main():
    run_audit.remote()

```

**Cost**: ~$100/month (Modal compute + LLM)

---

### **Option 3: Docker on AWS ECS (Most Control)**

**Pros**:

- Full control over infrastructure
- Easy local development
- Can scale horizontally (multiple audit workers)

**Cons**:

- More DevOps work
- Need to manage scheduling (EventBridge or Airflow)

**Cost**: ~$200/month (ECS Fargate + LLM + RDS for state)

---

## Monitoring & Observability Stack

### **Metrics to Track**

```python
# System health metrics
audit_completion_time_seconds = Histogram("audit_completion_time")
agent_success_rate = Gauge("agent_success_rate", ["agent_name"])
llm_tokens_used = Counter("llm_tokens_used", ["model_name"])
llm_cost_dollars = Counter("llm_cost_dollars")

# Business metrics
transactions_processed = Counter("transactions_processed")
flags_created = Counter("flags_created", ["severity"])
false_positive_rate = Gauge("false_positive_rate")
human_review_time_hours = Gauge("human_review_time_hours")

# Data quality metrics
data_completeness_score = Gauge("data_completeness_score", ["domain"])
reconciliation_match_rate = Gauge("reconciliation_match_rate")
anomaly_detection_recall = Gauge("anomaly_detection_recall")  # from labeled data

```

### **Alerts**

```yaml
alerts:
  - name: AuditTimeout
    condition: audit_completion_time_seconds > 1800 # 30min
    action: Send Slack alert to #eng-alerts

  - name: AgentFailureSpike
    condition: agent_success_rate{agent_name=~".*"} < 0.90
    action: PagerDuty page on-call engineer

  - name: LLMCostSpike
    condition: rate(llm_cost_dollars[1h]) > 5 # $5/hour = $120/day
    action: Email finance team + throttle LLM calls

  - name: DataQualityDegradation
    condition: data_completeness_score < 0.85
    action: Halt audits + alert data eng team

  - name: FalsePositiveRateHigh
    condition: false_positive_rate > 0.20
    action: Trigger automatic rule tuning job
```

### **Dashboard Panels**

```
┌─────────────────────────────────────────────────────────┐
│  Audit System Overview (Last 24 Hours)                 │
├─────────────────────────────────────────────────────────┤
│  Audits Completed: 24 / 24 ✅                          │
│  Avg Completion Time: 6m 32s                            │
│  Transactions Processed: 12,450                         │
│  Flags Created: 287 (2.3%)                              │
│    ├─ CRITICAL: 12                                      │
│    ├─ WARNING: 98                                       │
│    └─ INFO: 177                                         │
│  LLM Cost (24h): $1.87                                  │
│  False Positive Rate: 8.2% (↓ from 12% last week)      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Agent Performance                                      │
├─────────────────────────────────────────────────────────┤
│  Data Quality:      ████████████ 100% (avg 28s)        │
│  Reconciliation:    ███████████░  95% (avg 2m 15s)      │
│  Anomaly Detection: ████████████ 100% (avg 1m 45s)      │
│  Context Enrichment:████████████ 100% (avg 3m 10s)      │
│  Escalation:        ████████████ 100% (avg 2m 5s)       │
│  Logging:           ████████████ 100% (avg 15s)         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Human Review Queue Status                              │
├─────────────────────────────────────────────────────────┤
│  Pending Review: 127 items                              │
│    ├─ CRITICAL: 8 (avg age: 2 hours)                    │
│    └─ WARNING: 119 (avg age: 1 day)                     │
│  Reviewed Today: 45 items                               │
│  Avg Review Time: 3m 12s per item                       │
│  Team Capacity Remaining: 62% (5.2 hrs left today)      │
└─────────────────────────────────────────────────────────┘

```

---

## Graceful Degradation Strategy

### **If Databricks Goes Down**

```python
# In Orchestrator Agent
try:
    transactions = query_gold_tables(sql="SELECT * FROM txns WHERE ...")
except DatabricksConnectionError:
    # Save current state
    save_workflow_state(audit_run_id, {
        'status': 'paused',
        'reason': 'databricks_unavailable',
        'completed_agents': [],
        'pending_agents': ['all'],
        'timestamp': datetime.now()
    })

    # Send alert
    send_alert(
        channel="slack",
        message="🚨 Audit paused: Databricks unavailable. Will auto-resume when connection restored."
    )

    # Retry with exponential backoff
    retry_schedule = [30, 60, 120, 240, 480]  # seconds
    for wait_time in retry_schedule:
        time.sleep(wait_time)
        if check_databricks_health():
            # Resume from saved state
            restore_and_resume(audit_run_id)
            break
    else:
        # After 5 retries (15.5 minutes), escalate
        send_alert(
            channel="pagerduty",
            message="🚨 CRITICAL: Databricks still down after 15min. Manual intervention needed."
        )

```

### **If LLM API Goes Down**

```python
# In Escalation Agent
try:
    explanation = generate_explanation_with_llm(transaction, context)
except OpenAIError:
    # Fallback to deterministic template
    explanation = f"Flagged because: {', '.join(flag_reasons)}"

    # Log degraded mode
    log_agent_decision(
        agent_name="Escalation",
        action="explanation_generation",
        status="degraded",
        fallback="template_based",
        reason="llm_api_unavailable"
    )

    # Continue audit with reduced quality
    # (Still flag transactions, just less detailed explanations)

```

---

## Automatic Rule Tuning Details

### **Weekly Feedback Analysis Job**

```python
# feedback_analyzer.py (runs weekly via scheduled job)

def analyze_false_positives():
    """
    Queries Flag DB for all human-reviewed items from past week
    Identifies patterns in false positives
    """

    # Get all flags marked as "false positive" by humans
    false_positives = db.query("""
        SELECT
            flag_id,
            txn_id,
            severity,
            flag_reason,
            vendor_id,
            amount
        FROM flags
        WHERE reviewed = true
          AND human_decision = 'false_positive'
          AND created_at > NOW() - INTERVAL '7 days'
    """)

    # Analyze patterns
    vendor_fp_rates = false_positives.groupby('vendor_id').size() / \
                      all_flags.groupby('vendor_id').size()

    # Auto-whitelist vendors with >80% false positive rate
    for vendor_id, fp_rate in vendor_fp_rates.items():
        if fp_rate > 0.80:
            whitelist_vendor(vendor_id)
            log_rule_change(
                action="whitelist_vendor",
                vendor_id=vendor_id,
                reason=f"FP rate: {fp_rate:.2%}",
                auto_approved=True
            )

    # Adjust thresholds for rules with high FP rates
    rule_fp_rates = false_positives.groupby('flag_reason').size() / \
                    all_flags.groupby('flag_reason').size()

    for rule_name, fp_rate in rule_fp_rates.items():
        if fp_rate > 0.50:
            # Example: "amount_outlier" rule has 60% FP rate
            # → Increase threshold from 2σ to 3σ
            adjust_rule_threshold(
                rule_name=rule_name,
                adjustment="increase",
                reason=f"FP rate: {fp_rate:.2%}"
            )

def whitelist_vendor(vendor_id: str):
    """Add vendor to whitelist config"""
    config = load_config('rules.yaml')
    config['whitelisted_vendors'].append(vendor_id)
    save_config(config)

    # Also update knowledge graph
    kg_client.update_entity(
        entity_id=vendor_id,
        metadata={'whitelisted': True, 'reason': 'auto_tuned'}
    )

def adjust_rule_threshold(rule_name: str, adjustment: str, reason: str):
    """Modify rule threshold (e.g., 2σ → 3σ)"""
    config = load_config('rules.yaml')

    if rule_name == "amount_outlier":
        current_threshold = config['rules']['amount_outlier']['sigma']
        new_threshold = current_threshold + 0.5 if adjustment == "increase" else current_threshold - 0.5
        config['rules']['amount_outlier']['sigma'] = new_threshold

        log_rule_change(
            action="adjust_threshold",
            rule_name=rule_name,
            old_value=current_threshold,
            new_value=new_threshold,
            reason=reason
        )

    save_config(config)

```

### **Example Rules Config (YAML)**

```yaml
# rules.yaml
version: "2.1.3"
last_updated: "2025-02-03T10:00:00Z"

# Vendor whitelists (auto-updated by feedback analyzer)
whitelisted_vendors:
  - amazon_marketplace
  - starbucks_corporate
  - google_workspace

# Rule thresholds (auto-tuned based on false positive rates)
rules:
  amount_outlier:
    sigma: 3.0 # Originally 2.0, increased due to high FP rate
    min_transactions: 5 # Need 5+ txns from vendor before applying

  missing_receipt:
    grace_period_days: 7 # Increased from 5 due to feedback
    exclude_vendors: [google_workspace, aws] # Digital services don't have receipts

  unauthorized_vendor:
    use_knowledge_graph: true
    fuzzy_match_threshold: 0.85

  time_series_late_payment:
    late_threshold_days: 7 # Increased from 5
    confidence_interval: 0.80

# Domain-specific configs (manual overrides)
domain_configs:
  inventory_management:
    max_age_hours: 24
    critical_amount_threshold: 5000

  senior_living:
    max_age_hours: 168 # 7 days
    critical_amount_threshold: 10000

  business_operations:
    max_age_hours: 48
    critical_amount_threshold: 1000
```

---

## Sample Agent Code (Python)

```python
# reconciliation_agent.py
from crewai import Agent, Task
from tools import cross_source_matcher, entity_resolver_kg, find_orphan_transactions

reconciliation_agent = Agent(
    role="Transaction Reconciliation Specialist",
    goal="Match transactions across credit card statements, bank accounts, emails, and receipts with 95%+ accuracy",
    backstory="""You are an expert auditor with 15 years of experience in financial reconciliation.
    You have a keen eye for detail and can spot discrepancies across multiple data sources.
    You use the knowledge graph to resolve vendor entity variations and fuzzy matching for typos.""",

    tools=[
        cross_source_matcher,
        entity_resolver_kg,
        find_orphan_transactions
    ],

    verbose=True,  # For debugging
    allow_delegation=False,  # This agent doesn't delegate to sub-agents
    llm=None  # No LLM needed - all tools are deterministic
)

# Task definition
reconciliation_task = Task(
    description="""
    Given transactions from multiple sources (credit_card, bank, emails, receipts):
    1. Use cross_source_matcher to find matching transactions across sources
    2. Use entity_resolver_kg to resolve vendor name variations via knowledge graph
    3. Use find_orphan_transactions to identify transactions that appear in only one source
    4. Return a dict with 'matched_pairs', 'unmatched_transactions', and 'low_confidence_matches'
    """,

    agent=reconciliation_agent,

    expected_output="""
    {
        'matched_pairs': [...],
        'unmatched_transactions': [...],  # THESE ARE SUSPICIOUS
        'low_confidence_matches': [...]
    }
    """,

    context=[data_quality_task],  # Wait for data quality check to complete first
)

```

---

## Next Steps for Implementation

### **Phase 1: Foundation (Week 1-2)**

- [ ] Set up Databricks connection (SQL queries working)
- [ ] Implement Orchestrator Agent skeleton
- [ ] Build 3 core tools: `query_gold_tables`, `check_completeness`, `cross_source_matcher`
- [ ] Test end-to-end flow with 100 sample transactions
- [ ] Deploy to dev environment (Databricks Workflows or Modal)

### **Phase 2: Core Agents (Week 3-4)**

- [ ] Implement all 6 agents (Data Quality, Reconciliation, Anomaly, Context, Escalation, Logging)
- [ ] Build tool catalog (15-20 tools total)
- [ ] Set up Flag Database + Audit Trail Database schemas
- [ ] Implement graceful degradation (state saving, retries)
- [ ] Test with 1000 transactions from 25 Capital Partners

### **Phase 3: Human Feedback Loop (Week 5-6)**

- [ ] Build frontend flag review interface (or integrate with existing)
- [ ] Implement feedback collection pipeline
- [ ] Build automatic rule tuning job (weekly batch)
- [ ] Test rule adjustments with simulated feedback data

### **Phase 4: Monitoring & Production (Week 7-8)**

- [ ] Set up monitoring (Datadog, Grafana, or Databricks dashboards)
- [ ] Implement alerting (Slack, PagerDuty)
- [ ] Load test with 10,000 transactions
- [ ] Deploy to production
- [ ] Run parallel with existing manual process for 2 weeks (validation)

### **Phase 5: Optimization (Ongoing)**

- [ ] Analyze LLM cost breakdown, optimize further
- [ ] A/B test different ML models (XGBoost vs AutoML)
- [ ] Expand to additional domains beyond 25 CP
- [ ] Build self-service rule configuration UI for clients
