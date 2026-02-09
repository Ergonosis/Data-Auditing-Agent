#!/usr/bin/env python3
"""
Generate corrupted test datasets for Ergonosis Auditing demo pipeline.

This script creates augmented datasets with various types of data corruption:
- Missing fields (completeness issues)
- Duplicate transactions
- Orphan transactions (no reconciliation match)

Each dataset includes ground truth metadata for automated testing.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
RANDOM_SEED = 42
RIA_DATA_DIR = Path(__file__).parent.parent / "ria_data"
CLEAN_DATA_DIR = RIA_DATA_DIR / "clean_data"

# CSV file names
CSV_FILES = [
    "ria_clients.csv",
    "ria_bank_transactions.csv",
    "ria_credit_card_expenses_with_cardholders.csv",
    "ria_receipts_travel_and_business_dev.csv"
]

CREDIT_CARD_CSV = "ria_credit_card_expenses_with_cardholders.csv"
BANK_CSV = "ria_bank_transactions.csv"


def migrate_to_clean_data():
    """Copy original CSV files to clean_data/ directory."""
    print("\n=== Step 1: Migrating to clean_data/ ===")

    # Create clean_data directory
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all CSV files
    for csv_file in CSV_FILES:
        source = RIA_DATA_DIR / csv_file
        dest = CLEAN_DATA_DIR / csv_file

        if not source.exists():
            print(f"⚠️  Warning: {csv_file} not found at {source}")
            continue

        shutil.copy2(source, dest)
        print(f"✅ Copied {csv_file} to clean_data/")

    # Generate baseline metadata
    credit_df = pd.read_csv(CLEAN_DATA_DIR / CREDIT_CARD_CSV)

    metadata = {
        "dataset_name": "clean_data",
        "corruption_type": "none",
        "description": "Original pristine data with no corruption",
        "created_at": datetime.now().isoformat(),
        "ground_truth": {
            "total_rows": len(credit_df),
            "corrupted_rows": 0,
            "corrupted_row_ids": [],
            "corruption_details": {}
        },
        "expected_results": {
            "completeness_score": 1.0,
            "flags_created_min": 0,
            "flags_created_max": 100,
            "notes": "Clean data may still have flags due to sparse bank reconciliation data"
        }
    }

    metadata_path = CLEAN_DATA_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Created metadata.json for clean_data/")
    print(f"   Total rows: {len(credit_df)}")


def create_missing_fields_corruption(
    df: pd.DataFrame,
    corruption_rate: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, Dict]:
    """
    Null out required fields (vendor, amount, expense_date).

    Args:
        df: Original credit card DataFrame
        corruption_rate: Percentage of rows to corrupt (0.15 = 15%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (corrupted_df, ground_truth_dict)
    """
    print(f"\n=== Creating Missing Fields Corruption ({corruption_rate*100:.0f}%) ===")

    random.seed(seed)
    np.random.seed(seed)

    df_corrupted = df.copy()
    total_rows = len(df_corrupted)
    num_to_corrupt = int(total_rows * corruption_rate)

    # Randomly select rows to corrupt (stratified by category for variety)
    categories = df_corrupted['category'].unique()
    rows_per_category = num_to_corrupt // len(categories)

    corrupted_indices = []
    for category in categories:
        category_rows = df_corrupted[df_corrupted['category'] == category].index.tolist()
        num_from_category = min(rows_per_category, len(category_rows))
        selected = random.sample(category_rows, num_from_category)
        corrupted_indices.extend(selected)

    # Fill remaining quota randomly if needed
    remaining = num_to_corrupt - len(corrupted_indices)
    if remaining > 0:
        available = [i for i in df_corrupted.index if i not in corrupted_indices]
        corrupted_indices.extend(random.sample(available, remaining))

    # Apply corruption to each selected row
    corruption_details = {}

    for idx in corrupted_indices:
        expense_id = df_corrupted.loc[idx, 'expense_id']

        # Decide which fields to null out
        corruption_type = random.random()

        if corruption_type < 0.5:  # 50% - vendor only
            df_corrupted.loc[idx, 'merchant'] = np.nan
            nulled_fields = ['merchant']
        elif corruption_type < 0.8:  # 30% - amount only
            df_corrupted.loc[idx, 'amount'] = np.nan
            nulled_fields = ['amount']
        else:  # 20% - both vendor AND date
            df_corrupted.loc[idx, 'merchant'] = np.nan
            df_corrupted.loc[idx, 'expense_date'] = np.nan
            nulled_fields = ['merchant', 'expense_date']

        corruption_details[expense_id] = {"nulled_fields": nulled_fields}

    # Generate ground truth
    corrupted_row_ids = [df_corrupted.loc[idx, 'expense_id'] for idx in corrupted_indices]

    ground_truth = {
        "total_rows": total_rows,
        "corrupted_rows": len(corrupted_indices),
        "corrupted_row_ids": corrupted_row_ids,
        "corruption_details": corruption_details
    }

    print(f"✅ Corrupted {len(corrupted_indices)} rows ({len(corrupted_indices)/total_rows*100:.1f}%)")
    print(f"   Fields nulled: vendor={sum(1 for d in corruption_details.values() if 'merchant' in d['nulled_fields'])}, "
          f"amount={sum(1 for d in corruption_details.values() if 'amount' in d['nulled_fields'])}, "
          f"date={sum(1 for d in corruption_details.values() if 'expense_date' in d['nulled_fields'])}")

    return df_corrupted, ground_truth


def create_duplicate_corruption(
    df: pd.DataFrame,
    corruption_rate: float = 0.10,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, Dict]:
    """
    Create duplicate expense_id entries.

    Args:
        df: Original credit card DataFrame
        corruption_rate: Percentage to duplicate (0.10 = 10%)
        seed: Random seed

    Returns:
        Tuple of (corrupted_df, ground_truth_dict)
    """
    print(f"\n=== Creating Duplicate Corruption ({corruption_rate*100:.0f}%) ===")

    random.seed(seed)
    np.random.seed(seed)

    df_corrupted = df.copy()
    total_rows = len(df_corrupted)
    num_to_duplicate = int(total_rows * corruption_rate)

    # Randomly select rows to duplicate
    rows_to_duplicate = random.sample(range(total_rows), num_to_duplicate)

    duplicates = []
    corruption_details = {}

    for idx in rows_to_duplicate:
        original_row = df_corrupted.iloc[idx].copy()
        expense_id = original_row['expense_id']

        # Create 2-3 duplicates per row
        num_duplicates = random.choice([2, 3])

        duplicate_group = []
        for dup_num in range(num_duplicates):
            dup_row = original_row.copy()

            # 50% exact duplicates, 50% near-duplicates (±$5 amount)
            if random.random() > 0.5:
                variation = random.uniform(-5, 5)
                dup_row['amount'] = max(0, dup_row['amount'] + variation)
                duplicate_type = "near-duplicate"
            else:
                duplicate_type = "exact"

            duplicates.append(dup_row)
            duplicate_group.append({
                "duplicate_number": dup_num + 1,
                "type": duplicate_type
            })

        corruption_details[expense_id] = {
            "num_duplicates": num_duplicates,
            "duplicate_info": duplicate_group
        }

    # Add duplicates to dataframe
    df_duplicates = pd.DataFrame(duplicates)
    df_corrupted = pd.concat([df_corrupted, df_duplicates], ignore_index=True)

    # Shuffle to distribute duplicates throughout dataset
    df_corrupted = df_corrupted.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Generate ground truth (list of expense_ids that have duplicates)
    corrupted_row_ids = list(corruption_details.keys())

    ground_truth = {
        "total_rows": len(df_corrupted),  # Includes duplicates
        "original_rows": total_rows,
        "corrupted_rows": num_to_duplicate,  # Number of unique expense_ids that were duplicated
        "duplicate_count": len(df_duplicates),  # Total number of duplicate rows added
        "corrupted_row_ids": corrupted_row_ids,
        "corruption_details": corruption_details
    }

    print(f"✅ Duplicated {num_to_duplicate} unique rows")
    print(f"   Added {len(df_duplicates)} duplicate rows (total dataset size: {len(df_corrupted)})")
    print(f"   Exact duplicates: {sum(1 for d in corruption_details.values() for info in d['duplicate_info'] if info['type'] == 'exact')}")
    print(f"   Near-duplicates: {sum(1 for d in corruption_details.values() for info in d['duplicate_info'] if info['type'] == 'near-duplicate')}")

    return df_corrupted, ground_truth


def create_orphan_corruption(
    credit_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    new_orphans: int = 50,
    delete_bank_count: int = 10,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Create orphan transactions by adding phantom credit card expenses and deleting bank txns.

    Args:
        credit_df: Original credit card DataFrame
        bank_df: Original bank transactions DataFrame
        new_orphans: Number of new phantom expenses to add
        delete_bank_count: Number of bank transactions to delete
        seed: Random seed

    Returns:
        Tuple of (corrupted_credit_df, corrupted_bank_df, ground_truth_dict)
    """
    print(f"\n=== Creating Orphan Corruption ({new_orphans} new + {delete_bank_count} deleted bank) ===")

    random.seed(seed)
    np.random.seed(seed)

    credit_df_corrupted = credit_df.copy()
    bank_df_corrupted = bank_df.copy()

    # Part 1: Add new phantom credit card expenses (no matching bank transaction)
    max_expense_id = int(credit_df['expense_id'].str.replace('EXP_', '').max())

    phantom_expenses = []
    phantom_ids = []

    vendors = ["AWS Cloud Services", "Slack Technologies", "Zoom Video", "Cash Withdrawal - ATM",
               "Unknown Merchant", "Suspicious Vendor LLC", "Crypto Exchange"]
    categories = ["Software & Subscriptions", "Travel & Entertainment", "Miscellaneous"]

    # Get sample cardholder info
    sample_cardholders = credit_df[['cardholder_id', 'cardholder_name', 'card_program', 'card_last4']].drop_duplicates()

    for i in range(new_orphans):
        expense_id = f"EXP_{max_expense_id + i + 1:07d}"
        phantom_ids.append(expense_id)

        # Random cardholder
        ch_info = sample_cardholders.sample(1, random_state=seed+i).iloc[0]

        # High amount to trigger severity scoring ($5k-$15k)
        amount = random.uniform(5000, 15000)

        # Recent date (within last 30 days from latest date in dataset)
        max_date = pd.to_datetime(credit_df['expense_date']).max()
        days_ago = random.randint(1, 30)
        expense_date = (max_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        phantom_expense = {
            'expense_id': expense_id,
            'expense_date': expense_date,
            'cardholder_id': ch_info['cardholder_id'],
            'cardholder_name': ch_info['cardholder_name'],
            'card_program': ch_info['card_program'],
            'card_last4': ch_info['card_last4'],
            'category': random.choice(categories),
            'merchant': random.choice(vendors),
            'amount': round(amount, 2),
            'client_count_at_time': random.randint(150, 250)
        }

        phantom_expenses.append(phantom_expense)

    # Add phantom expenses to credit card dataframe
    df_phantoms = pd.DataFrame(phantom_expenses)
    credit_df_corrupted = pd.concat([credit_df_corrupted, df_phantoms], ignore_index=True)

    # Part 2: Delete bank transactions to orphan existing credit card expenses
    deleted_bank_indices = random.sample(range(len(bank_df_corrupted)), delete_bank_count)
    deleted_bank_txns = bank_df_corrupted.iloc[deleted_bank_indices].copy()

    # Remove deleted transactions
    bank_df_corrupted = bank_df_corrupted.drop(deleted_bank_indices).reset_index(drop=True)

    # Generate ground truth
    corruption_details = {}

    # Phantom expenses
    for phantom_id in phantom_ids:
        corruption_details[phantom_id] = {
            "corruption_type": "phantom_expense",
            "description": "New credit card expense with no matching bank transaction"
        }

    # Orphaned existing expenses (hard to track which exact expenses become orphaned)
    # Just note that bank transactions were deleted

    ground_truth = {
        "total_rows": len(credit_df_corrupted),
        "corrupted_rows": new_orphans,  # Number of phantom expenses added
        "corrupted_row_ids": phantom_ids,
        "corruption_details": corruption_details,
        "bank_deletions": {
            "deleted_count": delete_bank_count,
            "deleted_descriptions": deleted_bank_txns['description'].tolist()
        },
        "notes": f"Added {new_orphans} phantom credit card expenses with high amounts ($5k-$15k). "
                 f"Deleted {delete_bank_count} bank transactions to orphan existing expenses."
    }

    print(f"✅ Added {new_orphans} phantom credit card expenses")
    print(f"   Amount range: ${df_phantoms['amount'].min():.2f} - ${df_phantoms['amount'].max():.2f}")
    print(f"✅ Deleted {delete_bank_count} bank transactions")
    print(f"   Total orphans expected: {new_orphans} (phantom) + unknown (from deleted bank txns)")

    return credit_df_corrupted, bank_df_corrupted, ground_truth


def generate_dataset(
    dataset_name: str,
    corruption_type: str,
    corruption_func,
    corruption_params: Dict,
    expected_flags: Dict
):
    """
    Generate a corrupted dataset and save to directory.

    Args:
        dataset_name: Directory name for dataset
        corruption_type: Type of corruption applied
        corruption_func: Function to apply corruption
        corruption_params: Parameters for corruption function
        expected_flags: Expected results metadata
    """
    print(f"\n{'='*70}")
    print(f"Generating Dataset: {dataset_name}")
    print(f"{'='*70}")

    dataset_dir = RIA_DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load clean data
    clean_credit = pd.read_csv(CLEAN_DATA_DIR / CREDIT_CARD_CSV)
    clean_bank = pd.read_csv(CLEAN_DATA_DIR / BANK_CSV)

    # Apply corruption
    if corruption_type == "orphan_transactions":
        # Special case: corrupts both credit card and bank CSVs
        corrupted_credit, corrupted_bank, ground_truth = corruption_func(
            clean_credit, clean_bank, **corruption_params
        )

        # Save corrupted files
        corrupted_credit.to_csv(dataset_dir / CREDIT_CARD_CSV, index=False)
        corrupted_bank.to_csv(dataset_dir / BANK_CSV, index=False)

        # Copy other clean files
        for csv_file in CSV_FILES:
            if csv_file not in [CREDIT_CARD_CSV, BANK_CSV]:
                shutil.copy2(CLEAN_DATA_DIR / csv_file, dataset_dir / csv_file)

    else:
        # Corrupts only credit card CSV
        corrupted_credit, ground_truth = corruption_func(clean_credit, **corruption_params)

        # Save corrupted credit card file
        corrupted_credit.to_csv(dataset_dir / CREDIT_CARD_CSV, index=False)

        # Copy all other clean files
        for csv_file in CSV_FILES:
            if csv_file != CREDIT_CARD_CSV:
                shutil.copy2(CLEAN_DATA_DIR / csv_file, dataset_dir / csv_file)

    # Generate metadata
    metadata = {
        "dataset_name": dataset_name,
        "corruption_type": corruption_type,
        "corruption_params": corruption_params,
        "created_at": datetime.now().isoformat(),
        "ground_truth": ground_truth,
        "expected_results": expected_flags
    }

    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Dataset generated successfully at {dataset_dir}")
    print(f"   Metadata saved to {metadata_path}")


def generate_all_datasets():
    """Generate all test datasets."""
    print("\n" + "="*70)
    print("ERGONOSIS AUDITING - TEST DATASET GENERATOR")
    print("="*70)

    # Step 1: Migrate clean data
    migrate_to_clean_data()

    # Step 2: Generate missing fields dataset
    generate_dataset(
        dataset_name="missing_fields_15pct",
        corruption_type="missing_fields",
        corruption_func=create_missing_fields_corruption,
        corruption_params={"corruption_rate": 0.15, "seed": RANDOM_SEED},
        expected_flags={
            "completeness_score": 0.85,
            "flags_created_min": 300,
            "flags_created_max": 350,
            "severity_distribution": {
                "CRITICAL": 0,
                "WARNING": 325,
                "INFO": 0
            },
            "notes": "Most corrupted rows should trigger WARNING (+30 points for incomplete data)"
        }
    )

    # Step 3: Generate duplicates dataset
    generate_dataset(
        dataset_name="duplicates_10pct",
        corruption_type="duplicates",
        corruption_func=create_duplicate_corruption,
        corruption_params={"corruption_rate": 0.10, "seed": RANDOM_SEED},
        expected_flags={
            "duplicate_groups_detected": 217,
            "flags_created_min": 200,
            "flags_created_max": 250,
            "severity_distribution": {
                "CRITICAL": 0,
                "WARNING": 217,
                "INFO": 0
            },
            "notes": "Pipeline should detect duplicate expense_id groups. May flag some or all duplicates."
        }
    )

    # Step 4: Generate orphan transactions dataset
    generate_dataset(
        dataset_name="orphan_transactions_60",
        corruption_type="orphan_transactions",
        corruption_func=create_orphan_corruption,
        corruption_params={"new_orphans": 50, "delete_bank_count": 10, "seed": RANDOM_SEED},
        expected_flags={
            "flags_created_min": 50,
            "flags_created_max": 70,
            "severity_distribution": {
                "CRITICAL": 50,
                "WARNING": 10,
                "INFO": 0
            },
            "notes": "Phantom expenses with high amounts ($5k-$15k) should trigger CRITICAL "
                     "(+50 no_match + 20 high_amount = 70+ severity score)"
        }
    )

    print("\n" + "="*70)
    print("✅ ALL DATASETS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nDatasets created in: {RIA_DATA_DIR}")
    print("\nGenerated datasets:")
    print("  - clean_data/")
    print("  - missing_fields_15pct/")
    print("  - duplicates_10pct/")
    print("  - orphan_transactions_60/")
    print("\nNext steps:")
    print("  1. Verify datasets: ls -la ria_data/")
    print("  2. Check metadata: cat ria_data/missing_fields_15pct/metadata.json")
    print("  3. Run tests: python tests/demo_testing.py")


if __name__ == "__main__":
    generate_all_datasets()
