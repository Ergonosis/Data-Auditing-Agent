# Demo Pipeline Guide

This guide explains how to run the Ergonosis Auditing System in demo mode using the RIA CSV data.

## Overview

The demo mode allows you to test the full audit pipeline without requiring:
- ✅ Databricks connection
- ✅ Redis server
- ✅ Production infrastructure

Instead, it uses:
- 📁 Local CSV files from `ria_data/`
- 💾 In-memory state storage
- 🔧 Demo-optimized configuration

## Quick Start

### 1. Verify Demo Setup

First, verify that the demo data and configuration are working:

```bash
python3 scripts/verify_demo_setup.py
```

This will test:
- ✅ Demo data loading from CSV files
- ✅ Databricks client routing to demo mode
- ✅ Demo configuration loading
- ✅ In-memory state manager

### 2. Preview Demo Data (Dry Run)

View demo data statistics without running the full audit:

```bash
python3 scripts/run_demo.py --dry-run
```

This shows:
- Record counts for each data source
- Date ranges
- Receipt coverage percentage

### 3. Run Demo Audit

Execute the full audit pipeline with demo data:

```bash
python3 scripts/run_demo.py
```

⚠️ **Note**: This requires dependencies installed (`pip install -r requirements.txt`)

### 4. Run with Transaction Limit (Faster)

For quick testing, limit the number of transactions:

```bash
python3 scripts/run_demo.py --limit 100
```

## Demo Data

The demo uses RIA (Registered Investment Adviser) CSV files:

| File | Records | Description |
|------|---------|-------------|
| `ria_clients.csv` | 200 | Client registry |
| `ria_bank_transactions.csv` | 34 | Bank account transactions |
| `ria_credit_card_expenses_with_cardholders.csv` | 2,167 | Credit card expenses (primary audit target) |
| `ria_receipts_travel_and_business_dev.csv` | 1,149 | Supporting receipts (53% coverage) |

### Data Flow

```
Credit Card Expenses (2,167) → Audit Pipeline
    ↓
Bank Transactions (34) → Reconciliation
    ↓
Receipts (1,149) → Matching
    ↓
Audit Flags Created
```

## Expected Results

When running the demo audit, you should see:

### Transaction Counts
- **Total Processed**: 2,167 credit card transactions
- **Suspicious**: ~50-100 (based on anomaly detection and missing receipts)
- **Flags Created**: ~50-100

### Flag Distribution (Estimated)
- **CRITICAL**: 5-10 (high amount + no receipt + no approval)
- **WARNING**: 40-60 (minor discrepancies)
- **INFO**: 900-1,000 (SaaS vendors missing receipts - expected)

### Common Patterns Detected
1. **Missing Receipts** (~47%): Many are SaaS subscriptions (AWS, Slack, etc.) - these get whitelisted
2. **High-Value Entertainment**: Pebble Beach ($1,078), Nobu ($1,007) - flagged as anomalies
3. **Reconciliation**: Most transactions match bank records (clean dataset)

## Architecture

### Demo Mode Activation

Demo mode is activated via environment variables:

```bash
export DEMO_MODE=true
export ENVIRONMENT=demo
export STATE_BACKEND=memory
```

### Data Routing

```
query_gold_tables() [Databricks Client]
    ↓
Detects DEMO_MODE=true
    ↓
Routes to csv_data_loader.load_demo_data()
    ↓
Loads ria_credit_card_expenses.csv
    ↓
Transforms to Transaction schema
    ↓
Returns DataFrame (same interface as production)
```

### Configuration

Demo mode automatically loads `config/rules_demo.yaml` with:
- **Lower thresholds** (smaller dataset, less history)
- **Whitelisted vendors** (AWS, Slack, etc. that lack receipts)
- **Adjusted anomaly detection** (higher contamination rate expected)

## Files Created

### New Files (Demo Infrastructure)
```
src/demo/
├── __init__.py                 # Package marker
├── csv_data_loader.py          # CSV loading & transformation
└── demo_config.py              # Demo configuration helpers

config/
└── rules_demo.yaml             # Demo-specific rule overrides

scripts/
├── run_demo.py                 # Main demo runner
└── verify_demo_setup.py        # Setup verification

tests/
└── test_demo_pipeline.py       # Demo integration tests
```

### Modified Files (Production-Safe)
```
src/tools/databricks_client.py      # Added demo mode routing
src/utils/config_loader.py          # Auto-loads demo config
src/orchestrator/state_manager.py   # Added in-memory backend
```

### No Changes Required
```
src/orchestrator/orchestrator_agent.py   # Works as-is
src/agents/*.py                          # All agents work as-is
src/tools/*.py                           # All tools work as-is
```

## Testing

### Quick Unit Tests

Run demo-specific tests:

```bash
pytest tests/test_demo_pipeline.py -v
```

### Full Integration Test (Slow)

Run the complete audit cycle as a test:

```bash
pytest tests/test_demo_pipeline.py::TestDemoEndToEnd::test_full_demo_audit_cycle -v
```

⚠️ **Note**: This test is marked as `skip` by default because it's slow (2-5 minutes)

## Troubleshooting

### "Demo data directory not found"

Ensure you're in the project root:
```bash
cd /path/to/ergonosis_auditing
ls ria_data/  # Should show CSV files
```

### "Module not found: crewai"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "Redis module not available"

This is expected! Demo mode automatically falls back to in-memory storage.

### Demo vs Production Mode

To switch back to production mode:
```bash
unset DEMO_MODE
export ENVIRONMENT=production
export DATABRICKS_HOST=your-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=your-token
```

## Performance

### Expected Runtime
- **Dry run**: < 1 second
- **Verification**: < 2 seconds
- **Full audit**: 2-5 minutes (depends on LLM API speed)

### Cost Estimate
- **LLM API calls**: $0.05-$0.10 per full demo run
- **Infrastructure**: $0 (all local)

## Safety Guarantees

✅ **Production Isolation**
- Demo mode requires explicit `DEMO_MODE=true`
- Production code paths unchanged
- No risk of demo data corrupting production

✅ **Data Isolation**
- Uses separate `ria_data/` directory
- Separate `rules_demo.yaml` configuration
- In-memory state (not shared Redis)

✅ **Reversibility**
- Can delete `src/demo/` without breaking anything
- Disable by removing environment variable
- All changes are additive

## Next Steps

After successfully running the demo:

1. **Customize Configuration**: Edit `config/rules_demo.yaml` to adjust thresholds
2. **Add Your Own Data**: Replace RIA CSV files with your own data (same schema)
3. **Extend Agents**: Agents work the same in demo and production
4. **Production Deployment**: Set production environment variables

## Support

For issues or questions about demo mode:
1. Run `scripts/verify_demo_setup.py` to diagnose issues
2. Check logs for detailed error messages
3. Ensure all CSV files are present in `ria_data/`
