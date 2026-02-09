# Demo Pipeline Quick Start

## ✅ Setup Complete!

The demo pipeline is now ready to use. All verification tests have passed.

## Quick Run (3 Steps)

### 1. Load Demo Environment

```bash
export $(cat .env.demo | grep -v '^#' | xargs)
```

### 2. Preview Demo Data (Dry Run)

```bash
python3 scripts/run_demo.py --dry-run
```

**Expected Output:**
- ✅ 2,167 credit card transactions
- ✅ 34 bank transactions
- ✅ 1,149 receipts (53% coverage)
- ✅ Date range: Feb 2024 - Feb 2026

### 3. Run Full Audit (Optional - Requires API Key)

```bash
# Edit .env.demo and add your real OPENAI_API_KEY
# Then run:
python3 scripts/run_demo.py
```

## Demo Data Overview

| File | Records | Description |
|------|---------|-------------|
| `ria_clients.csv` | 200 | RIA client registry |
| `ria_bank_transactions.csv` | 34 | Bank account transactions |
| `ria_credit_card_expenses_with_cardholders.csv` | 2,167 | **Primary audit target** |
| `ria_receipts_travel_and_business_dev.csv` | 1,149 | Supporting documentation |

## Key Features

### ✅ What Works in Demo Mode

- **Data Loading**: Loads all CSV files successfully
- **Databricks Client**: Routes to CSV instead of cloud
- **Configuration**: Uses `config/rules_demo.yaml`
- **State Management**: In-memory (no Redis needed)
- **Verification**: All tests pass

### ⚠️ What Requires API Keys

- **Full Audit Run**: Requires `OPENAI_API_KEY` for LLM calls
- **Agent Execution**: CrewAI agents need LLM access
- **Anomaly Detection**: Some features use ML models (work offline)

## Expected Results (When Running Full Audit)

Based on the RIA data patterns:

**Transaction Processing:**
- Total: 2,167 transactions
- Suspicious: ~50-100 (2-5%)
- Flags Created: ~50-100

**Flag Distribution:**
- CRITICAL: 5-10 (high amount + no receipt + no approval)
- WARNING: 40-60 (minor discrepancies)
- INFO: 900-1,000 (SaaS vendors missing receipts - expected)

**Common Patterns:**
1. **Missing Receipts (47%)**: Many are SaaS (AWS, Slack) → whitelisted
2. **High Entertainment**: Pebble Beach ($1,078), Nobu ($1,007) → flagged
3. **Clean Reconciliation**: Most match bank records

## Architecture

```
CSV Files (ria_data/)
    ↓
DemoDataLoader
    ↓
Transaction DataFrame (2,167 rows)
    ↓
Orchestrator → Parallel Agents
    ├─ Data Quality
    ├─ Reconciliation
    └─ Anomaly Detection
    ↓
Sequential Agents
    ├─ Context Enrichment
    └─ Escalation
    ↓
Audit Flags
```

## Files Created

### New Demo Infrastructure
```
src/demo/
├── __init__.py
├── csv_data_loader.py       # CSV → DataFrame transformation
└── demo_config.py            # Demo configuration helpers

config/
└── rules_demo.yaml           # Demo-optimized thresholds

scripts/
├── run_demo.py              # Main demo runner ⭐
└── verify_demo_setup.py     # Quick verification

.env.demo                     # Demo environment variables
```

### Modified Files (Production-Safe)
```
src/tools/databricks_client.py    # Added demo mode routing
src/utils/config_loader.py        # Auto-loads demo config
src/orchestrator/state_manager.py # In-memory backend
```

## Troubleshooting

### "OPENAI_API_KEY is required"

**For dry-run mode**: This shouldn't happen - you found a bug!

**For full audit**:
1. Get an API key from OpenAI
2. Edit `.env.demo` and replace the placeholder
3. Run `export $(cat .env.demo | grep -v '^#' | xargs)`

### "Demo data directory not found"

Ensure you're in the project root:
```bash
cd /path/to/ergonosis_auditing
ls ria_data/  # Should show 4 CSV files
```

### "Module not found" errors

Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Next Steps

1. ✅ **Verify Setup**: `python3 scripts/verify_demo_setup.py`
2. ✅ **Dry Run**: `python3 scripts/run_demo.py --dry-run`
3. 🔑 **Add API Key**: Edit `.env.demo` with real key
4. 🚀 **Run Audit**: `python3 scripts/run_demo.py`
5. 📊 **Review Flags**: Check audit output

## Production Deployment

When ready for production:

```bash
# Create .env with production credentials
cp .env.example .env

# Edit .env with real values:
# - DATABRICKS_HOST
# - DATABRICKS_TOKEN
# - OPENAI_API_KEY

# Switch to production mode
export ENVIRONMENT=production
export DEMO_MODE=false

# Run production audit
python3 src/main.py
```

## Support

**Issues?**
1. Run verification: `python3 scripts/verify_demo_setup.py`
2. Check logs for errors
3. Ensure all CSV files present
4. Verify Python 3.13 compatibility

**Questions?**
- See `docs/DEMO_GUIDE.md` for detailed documentation
- Check `system_specs.md` for architecture details
