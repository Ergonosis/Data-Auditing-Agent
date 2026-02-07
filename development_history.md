# Ergonosis Auditing - Architecture Decision Log

## ADR-001: Databricks Workflows Deployment (2025-02-06)
**Status**: Accepted
**Context**: Need to choose deployment platform from 3 options (Databricks, Modal, AWS ECS)
**Decision**: Use Databricks Workflows for native Delta Lake integration
**Consequences**:
- ✅ Lower data latency (direct Gold table access)
- ✅ Native Python + SQL execution
- ❌ Higher compute cost (~$150/month vs $100)
- ⚠️ Vendor lock-in to Databricks ecosystem

## ADR-002: Tool-First Architecture (2025-02-06)
**Status**: Accepted
**Context**: Need to minimize LLM costs (target: <$100/month for 10k transactions)
**Decision**: Filter to suspicious subset (1-3%) using deterministic tools before LLM calls
**Consequences**:
- ✅ ~97% cost reduction ($2/month actual vs $60+ without filtering)
- ✅ Faster execution (no LLM latency for 97% of transactions)
- ❌ Requires robust SQL/ML tools for filtering
- ⚠️ Need to validate tool accuracy to avoid missing fraud

## ADR-003: Weekly Auto-Tuning Without Approval (2025-02-06)
**Status**: Accepted
**Context**: Manual rule tuning is slow (weeks of lag) and error-prone
**Decision**: Automatically adjust thresholds based on FP rates >50% without human approval
**Consequences**:
- ✅ Faster adaptation to false positives (1 week vs 4+ weeks manual)
- ✅ Reduces finance team review burden
- ❌ Risk of over-tuning if FP feedback is biased
- ⚠️ Need audit trail for compliance review

## ADR-004: Delta Lake for Knowledge Graph (2025-02-06)
**Status**: Accepted
**Context**: Need entity resolution (vendor name variations) - 3 options: Delta Lake, Neo4j, Neptune
**Decision**: Use Delta Lake tables for KG (avoid separate graph database)
**Consequences**:
- ✅ Zero additional infrastructure cost
- ✅ Native Databricks integration (SQL queries)
- ❌ Less expressive than Neo4j for complex graph traversals
- 💡 Can migrate to Neo4j later if graph complexity grows

## ADR-005: Abstract Databricks Interface with Mock Data (2025-02-06)
**Status**: Accepted
**Context**: Databricks Gold tables not ready yet, but need to start development
**Decision**: Build abstract Databricks client with JSON mock data adapter for local dev
**Consequences**:
- ✅ Unblocks development (can build agents/tools immediately)
- ✅ Easy to swap in real Databricks connection later
- ⚠️ Need to ensure mock data matches real schema

## ADR-006: Single Broad Domain Strategy (2025-02-06)
**Status**: Accepted
**Context**: System supports 3 domains (inventory, senior_living, business_ops) - build all or focus?
**Decision**: Start with single "default" domain config, add domain-specific rules later
**Consequences**:
- ✅ Faster initial deployment
- ✅ Domain inference still works (LLM can classify transactions)
- ❌ Less optimized thresholds initially
- 💡 Can add specific domains after production validation
