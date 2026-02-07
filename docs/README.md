# Task Specifications - Implementation Guide

This directory contains detailed, self-contained task specifications for implementing the Ergonosis Auditing System. Each task can be assigned to a separate Claude Code agent for parallel execution.

---

## 📋 Quick Navigation

### Start Here
1. **[TASK_ASSIGNMENTS_SUMMARY.md](TASK_ASSIGNMENTS_SUMMARY.md)** - Overview of all tasks, dependencies, and execution strategy

### Task Specifications (Execute in Order)

#### Phase 1: Foundation (Sequential)
- **[TASK_1_INFRASTRUCTURE.md](TASK_1_INFRASTRUCTURE.md)** - Core infrastructure layer
  - Databricks client with mock data adapter
  - LLM client with cost tracking
  - Redis state manager
  - Database schemas
  - **Effort**: 2 hours | **Dependencies**: None | **Status**: 🚀 Ready to start

#### Phase 2: Agents (Parallel Execution Possible)
- **[TASK_2_DATA_QUALITY_AGENT.md](TASK_2_DATA_QUALITY_AGENT.md)** - Data Quality Agent + 5 tools
  - Completeness checks
  - Schema validation
  - Duplicate detection
  - Domain inference
  - Quality gates
  - **Effort**: 2 hours | **Dependencies**: TASK 1 | **Status**: ⏳ After TASK 1

- **[TASK_3_RECONCILIATION_ANOMALY_AGENTS.md](TASK_3_RECONCILIATION_ANOMALY_AGENTS.md)** - Reconciliation + Anomaly Detection Agents
  - Cross-source matching (5 tools)
  - Isolation Forest anomaly detection (5 tools)
  - **Effort**: 3 hours | **Dependencies**: TASK 1 | **Status**: ⏳ After TASK 1

- **[TASK_4_CONTEXT_ESCALATION_AGENTS.md](TASK_4_CONTEXT_ESCALATION_AGENTS.md)** - Context Enrichment + Escalation Agents
  - Email/calendar/receipt search (5 tools)
  - Severity classification (5 tools)
  - **Effort**: 3 hours | **Dependencies**: TASK 1 | **Status**: ⏳ After TASK 1

#### Phase 3: Integration (Sequential)
- **[TASK_5_ORCHESTRATOR_MAIN.md](TASK_5_ORCHESTRATOR_MAIN.md)** - Orchestrator + Main Entry Point
  - Logging agent (5 tools)
  - Orchestrator coordinator
  - Main entry point
  - Feedback analyzer script
  - **Effort**: 4 hours | **Dependencies**: TASKS 2, 3, 4 | **Status**: ⏳ After Phase 2

---

## 🎯 Task Assignment Strategy

### Recommended Approach: Maximum Parallelization

```
┌──────────────────────────────────────────────────┐
│ Phase 1: Foundation (Sequential)                 │
│ TASK 1: Infrastructure                  (2 hrs)  │
│ - Databricks client, LLM client, State Manager  │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Phase 2: Agents (Parallel - 3 agents)            │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐    │
│ │  TASK 2    │ │  TASK 3    │ │  TASK 4    │    │
│ │ Data Qual. │ │ Recon +    │ │ Context +  │    │
│ │  (2 hrs)   │ │ Anomaly    │ │ Escalation │    │
│ │            │ │  (3 hrs)   │ │  (3 hrs)   │    │
│ └────────────┘ └────────────┘ └────────────┘    │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Phase 3: Integration (Sequential)                │
│ TASK 5: Orchestrator + Main             (4 hrs) │
│ - Brings all agents together                     │
└──────────────────────────────────────────────────┘

Total Time: 2 + 3 + 4 = 9 hours (with parallelization)
```

### Alternative: Sequential Execution

If you can't parallelize, execute in order: TASK 1 → 2 → 3 → 4 → 5 (total ~14 hours)

---

## 📊 Task Details at a Glance

| Task | Focus | Files Created | Lines of Code | LLM Usage |
|------|-------|---------------|---------------|-----------|
| **1** | Infrastructure | 5 core files | ~400 | No LLM |
| **2** | Data Quality Agent | 2 files + tests | ~350 | Minimal (~200 calls/month) |
| **3** | Recon + Anomaly | 4 files + tests | ~550 | No LLM (pure ML/SQL) |
| **4** | Context + Escalation | 4 files + tests | ~600 | Selective (<10% of txns) |
| **5** | Orchestrator | 6 files + tests | ~550 | No LLM (coordination only) |
| **Total** | Full System | **21 files** | **~2450** | **~$2/month** |

---

## 🚀 Getting Started

### For Each Task:

1. **Read the task specification thoroughly**
   - All requirements are self-contained
   - Code templates and examples included
   - Success criteria clearly defined

2. **Check dependencies**
   - TASK 1 has no dependencies (start here)
   - TASKS 2, 3, 4 depend on TASK 1 (can run in parallel)
   - TASK 5 depends on TASKS 2, 3, 4 (final integration)

3. **Create the specified files**
   - Follow the provided code structure
   - Use existing utilities from `src/utils/*`
   - Log extensively with structured logger

4. **Write unit tests**
   - Test specifications included in each task
   - Verify all success criteria

5. **Verify completion**
   - Check off all items in success criteria checklist
   - Ensure imports work
   - Run unit tests

---

## 🔍 Key Information in Each Task Spec

Every task specification includes:

- ✅ **Objective** - What this task accomplishes
- ✅ **Context** - Why this task exists and how it fits
- ✅ **Architecture** - Visual diagram of components
- ✅ **Files to Create** - Complete list with line counts
- ✅ **Code Templates** - Detailed implementation guidance
- ✅ **Testing Requirements** - Unit test specifications
- ✅ **Success Criteria** - Checklist for completion
- ✅ **Dependencies** - What must be complete first
- ✅ **Important Notes** - Critical implementation details
- ✅ **Estimated Effort** - Time estimate for completion

---

## 🎓 Learning Path

If you're unfamiliar with the system:

1. Read **[TASK_ASSIGNMENTS_SUMMARY.md](TASK_ASSIGNMENTS_SUMMARY.md)** first
2. Review **[../system_specs.md](../system_specs.md)** for full context
3. Check **[../development_history.md](../development_history.md)** for design decisions
4. Then dive into individual task specifications

---

## 🧪 Testing & Validation

After completing all tasks:

```bash
# Run all unit tests
pytest tests/

# Run integration test
pytest tests/test_integration.py

# Execute full audit cycle
python src/main.py

# Run feedback analyzer
python scripts/feedback_analyzer.py
```

Expected output after full implementation:
- ✅ All 6 agents execute successfully
- ✅ 1000 transactions processed (from mock data)
- ✅ ~27 flags created (5 CRITICAL, 22 WARNING)
- ✅ Audit completes in <10 minutes
- ✅ State saved to Redis
- ✅ Metrics tracked
- ✅ Full audit trail logged

---

## 📞 Support & Troubleshooting

### Common Issues

**Import Errors**
- Check that all paths are relative to project root
- Ensure `src/` modules use dot notation: `from src.utils.errors import ...`

**Mock Data Issues**
- Sample fixture files should be in `tests/fixtures/`
- Databricks client checks `ENVIRONMENT=development` to use mock data

**Test Failures**
- Ensure Redis is running (or tests will skip state management)
- Check that `.env` file has `OPENROUTER_API_KEY` set

### Getting Help

1. Check the specific task specification for detailed guidance
2. Review existing code in `src/models/`, `src/utils/` for patterns
3. See `system_specs.md` for full architectural context
4. Check `development_history.md` for design rationale

---

## ✨ Final Notes

This is a **production-ready system** designed for:
- ✅ Financial audit compliance (7-year retention)
- ✅ Cost optimization (<$150/month including infrastructure)
- ✅ Performance at scale (10,000+ transactions/month)
- ✅ Auto-tuning based on feedback (no manual rule updates)
- ✅ Full transparency (immutable audit trail)

**The task specifications are intentionally detailed and self-contained** to enable parallel development by multiple Claude Code agents.

Good luck with implementation! 🚀
