"""Prometheus metrics definitions"""

from prometheus_client import Counter, Histogram, Gauge


# System health metrics
audit_completion_time = Histogram(
    'audit_completion_time_seconds',
    'Time to complete full audit cycle',
    buckets=[30, 60, 120, 300, 600, 1800]  # up to 30 minutes
)

agent_success_rate = Gauge(
    'agent_success_rate',
    'Agent success rate (0-1)',
    labelnames=['agent_name']
)

agent_execution_time = Histogram(
    'agent_execution_time_seconds',
    'Execution time per agent',
    labelnames=['agent_name'],
    buckets=[5, 15, 30, 60, 120, 300]
)

workflow_state_saves = Counter(
    'workflow_state_saves_total',
    'Number of workflow state saves',
    labelnames=['status']  # success, failure
)

# LLM cost & usage tracking
llm_tokens_counter = Counter(
    'llm_tokens_used_total',
    'Total LLM tokens consumed',
    labelnames=['model_name', 'agent_name']
)

llm_cost_counter = Counter(
    'llm_cost_dollars_total',
    'Total LLM cost in USD',
    labelnames=['model_name']
)

llm_api_latency = Histogram(
    'llm_api_latency_seconds',
    'Latency of LLM API calls',
    labelnames=['model_name'],
    buckets=[0.5, 1, 2, 5, 10, 30]
)

llm_rate_limit_hits = Counter(
    'llm_rate_limit_hits_total',
    'Number of LLM rate limit errors',
    labelnames=['model_name']
)

# Business metrics
transactions_processed = Counter(
    'transactions_processed_total',
    'Total transactions audited',
    labelnames=['domain']
)

flags_created = Counter(
    'flags_created_total',
    'Total flags created',
    labelnames=['severity']
)

flags_pending_review = Gauge(
    'flags_pending_review',
    'Flags awaiting human review',
    labelnames=['severity']
)

flags_reviewed = Counter(
    'flags_reviewed_total',
    'Flags reviewed by humans',
    labelnames=['decision']  # approved, rejected, false_positive
)

human_review_time = Histogram(
    'human_review_time_seconds',
    'Time to review a flag',
    buckets=[30, 60, 180, 300, 600]  # 30s to 10min
)

false_positive_rate = Gauge(
    'false_positive_rate',
    'Ratio of false positives to total reviewed'
)

# Data quality metrics
data_completeness_score = Gauge(
    'data_completeness_score',
    'Percentage of complete records (0-1)',
    labelnames=['source', 'domain']
)

reconciliation_match_rate = Gauge(
    'reconciliation_match_rate',
    'Percentage of transactions matched across sources',
    labelnames=['source_pair']
)

orphan_transaction_count = Gauge(
    'orphan_transaction_count',
    'Transactions appearing in only one source'
)

anomaly_detection_precision = Gauge(
    'anomaly_detection_precision',
    'Precision of anomaly detection (from labeled feedback)'
)

anomaly_detection_recall = Gauge(
    'anomaly_detection_recall',
    'Recall of anomaly detection (from labeled feedback)'
)

# Infrastructure metrics
databricks_connection_healthy = Gauge(
    'databricks_connection_healthy',
    'Whether Databricks connection is alive (0/1)'
)

redis_connection_healthy = Gauge(
    'redis_connection_healthy',
    'Whether Redis (state store) is alive (0/1)'
)

knowledge_graph_query_latency = Histogram(
    'knowledge_graph_query_latency_seconds',
    'Latency of KG entity lookups',
    buckets=[0.1, 0.5, 1, 5, 10]
)

audit_trail_write_latency = Histogram(
    'audit_trail_write_latency_seconds',
    'Latency of appending to audit trail',
    buckets=[0.01, 0.05, 0.1, 0.5, 1]
)
