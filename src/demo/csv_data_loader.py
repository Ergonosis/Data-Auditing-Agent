"""Demo data loader for RIA CSV files - simulates Databricks Gold table queries"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import os
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DemoDataLoader:
    """
    Loads demo data from CSV files to simulate production Databricks queries.

    This class mimics the interface of the production DatabricksClient,
    allowing the same agent code to work in both demo and production environments.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize demo data loader

        Args:
            data_dir: Path to directory containing CSV files (defaults to ria_data/)
        """
        if data_dir is None:
            data_dir = os.getenv("DEMO_DATA_DIR", "ria_data")

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Demo data directory not found: {data_dir}")

        self._clients = None
        self._bank_transactions = None
        self._credit_card_expenses = None
        self._receipts = None

        logger.info(f"Demo data loader initialized with data from: {self.data_dir}")

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file with error handling"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()

    @property
    def clients(self) -> pd.DataFrame:
        """Load and cache clients data"""
        if self._clients is None:
            self._clients = self._load_csv("ria_clients.csv")
        return self._clients

    @property
    def bank_transactions(self) -> pd.DataFrame:
        """Load and cache bank transactions"""
        if self._bank_transactions is None:
            df = self._load_csv("ria_bank_transactions.csv")
            # Convert date column
            if not df.empty and 'transaction_date' in df.columns:
                df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            self._bank_transactions = df
        return self._bank_transactions

    @property
    def credit_card_expenses(self) -> pd.DataFrame:
        """Load and cache credit card expenses"""
        if self._credit_card_expenses is None:
            df = self._load_csv("ria_credit_card_expenses_with_cardholders.csv")
            # Convert date column
            if not df.empty and 'expense_date' in df.columns:
                df['expense_date'] = pd.to_datetime(df['expense_date'])
            self._credit_card_expenses = df
        return self._credit_card_expenses

    @property
    def receipts(self) -> pd.DataFrame:
        """Load and cache receipts"""
        if self._receipts is None:
            df = self._load_csv("ria_receipts_travel_and_business_dev.csv")
            # Convert date column
            if not df.empty and 'receipt_date' in df.columns:
                df['receipt_date'] = pd.to_datetime(df['receipt_date'])
            self._receipts = df
        return self._receipts

    def get_transactions_for_audit(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get transactions for audit with optional filtering

        This is the main method called by the orchestrator to get transactions to audit.
        It transforms credit card expenses into the expected Transaction schema.

        Args:
            since: Only return transactions after this date
            limit: Maximum number of transactions to return

        Returns:
            DataFrame of transactions in expected schema
        """
        df = self.credit_card_expenses.copy()

        if df.empty:
            logger.warning("No credit card expenses found in demo data")
            return pd.DataFrame(columns=['txn_id', 'source', 'amount', 'vendor', 'date'])

        # Apply date filter
        if since and 'expense_date' in df.columns:
            df = df[df['expense_date'] >= since]

        # Apply limit
        if limit:
            df = df.head(limit)

        # Transform to expected schema
        df = df.rename(columns={
            'expense_id': 'txn_id',
            'expense_date': 'date',
            'merchant': 'vendor'
        })

        # Add source column
        df['source'] = 'credit_card'

        # Select and reorder columns to match expected schema
        expected_cols = ['txn_id', 'source', 'amount', 'vendor', 'date']

        # Add optional columns if they exist
        optional_cols = ['category', 'cardholder_name', 'card_last4']
        for col in optional_cols:
            if col in df.columns:
                expected_cols.append(col)

        df = df[expected_cols]

        logger.info(f"Retrieved {len(df)} transactions for audit")
        return df

    def get_bank_transactions_for_matching(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get bank transactions for reconciliation matching

        Args:
            start_date: Filter transactions after this date
            end_date: Filter transactions before this date

        Returns:
            DataFrame of bank transactions
        """
        df = self.bank_transactions.copy()

        if not df.empty:
            if start_date and 'transaction_date' in df.columns:
                df = df[df['transaction_date'] >= start_date]

            if end_date and 'transaction_date' in df.columns:
                df = df[df['transaction_date'] <= end_date]

        logger.info(f"Retrieved {len(df)} bank transactions for matching")
        return df

    def get_receipts_for_matching(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get receipts for matching

        Args:
            start_date: Filter receipts after this date
            end_date: Filter receipts before this date

        Returns:
            DataFrame of receipts
        """
        df = self.receipts.copy()

        if not df.empty:
            if start_date and 'receipt_date' in df.columns:
                df = df[df['receipt_date'] >= start_date]

            if end_date and 'receipt_date' in df.columns:
                df = df[df['receipt_date'] <= end_date]

        logger.info(f"Retrieved {len(df)} receipts for matching")
        return df

    def get_summary_stats(self) -> dict:
        """Get summary statistics of demo data"""
        return {
            'clients': len(self.clients),
            'bank_transactions': len(self.bank_transactions),
            'credit_card_expenses': len(self.credit_card_expenses),
            'receipts': len(self.receipts),
            'data_dir': str(self.data_dir.absolute())
        }


# Global instance for caching
_demo_loader = None


def _transform_to_gold_schema(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """
    Transform raw CSV data to match Databricks gold.transactions schema.

    Args:
        df: Raw DataFrame from CSV
        source_type: One of 'credit_card', 'bank', 'receipts'

    Returns:
        Transformed DataFrame with gold schema columns
    """
    if df.empty:
        return pd.DataFrame(columns=['txn_id', 'source', 'amount', 'vendor', 'date', 'vendor_id'])

    df = df.copy()

    # Transform based on source type
    if source_type == 'credit_card':
        df = df.rename(columns={
            'expense_id': 'txn_id',
            'expense_date': 'date',
            'merchant': 'vendor'
        })
        df['source'] = 'credit_card'

    elif source_type == 'bank':
        # Bank transactions don't have IDs, so generate them
        df['txn_id'] = ['BANK_' + str(i).zfill(6) for i in range(1, len(df) + 1)]
        df = df.rename(columns={
            'transaction_date': 'date',
            'description': 'vendor'  # Bank descriptions become vendor names
        })
        df['source'] = 'bank'

    elif source_type == 'receipts':
        df = df.rename(columns={
            'receipt_id': 'txn_id',
            'receipt_date': 'date',
            'vendor_name': 'vendor'
        })
        df['source'] = 'receipts'

    # Add vendor_id placeholder (for future KG entity resolution)
    if 'vendor_id' not in df.columns:
        df['vendor_id'] = None

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df


def _parse_sql_filters(sql_query: str) -> dict:
    """
    Parse SQL query to extract filters.

    Args:
        sql_query: SQL query string

    Returns:
        Dictionary with 'source', 'start_date', 'end_date' filters
    """
    import re

    filters = {}
    sql_lower = sql_query.lower()

    # Extract source filter: WHERE source = 'credit_card'
    source_match = re.search(r"source\s*=\s*['\"](\w+)['\"]", sql_lower)
    if source_match:
        filters['source'] = source_match.group(1)

    # Extract date range: WHERE date BETWEEN 'X' AND 'Y'
    date_range_match = re.search(r"date\s+between\s+['\"]([^'\"]+)['\"]\s+and\s+['\"]([^'\"]+)['\"]", sql_lower)
    if date_range_match:
        filters['start_date'] = date_range_match.group(1)
        filters['end_date'] = date_range_match.group(2)

    return filters


def load_demo_data(sql_query: str) -> pd.DataFrame:
    """
    Load demo data based on SQL query and transform to gold.transactions schema.

    This function simulates Databricks query_gold_tables() for demo mode.
    It routes queries to appropriate CSV files, transforms to gold schema,
    and applies SQL filters.

    Args:
        sql_query: SQL query string (used to infer which data to return)

    Returns:
        DataFrame with query results in gold.transactions schema
    """
    global _demo_loader

    # Initialize loader if needed
    if _demo_loader is None:
        _demo_loader = DemoDataLoader()

    sql_lower = sql_query.lower()

    # Parse filters from SQL
    filters = _parse_sql_filters(sql_query)

    # Route to appropriate data source and transform
    if 'gold.recent_transactions' in sql_lower or 'recent_transactions' in sql_lower:
        # Main transactions table - use the already-transformed method
        df = _demo_loader.get_transactions_for_audit()

    elif 'gold.transactions' in sql_lower:
        # Generic gold.transactions query - route based on source filter
        source = filters.get('source', 'credit_card')

        if source == 'credit_card':
            df = _demo_loader.credit_card_expenses
            df = _transform_to_gold_schema(df, 'credit_card')
        elif source == 'bank':
            df = _demo_loader.bank_transactions
            df = _transform_to_gold_schema(df, 'bank')
        elif source == 'receipts':
            df = _demo_loader.receipts
            df = _transform_to_gold_schema(df, 'receipts')
        else:
            logger.warning(f"Unknown source '{source}', defaulting to credit_card")
            df = _demo_loader.credit_card_expenses
            df = _transform_to_gold_schema(df, 'credit_card')

    elif 'credit_card' in sql_lower or 'expense' in sql_lower:
        df = _demo_loader.credit_card_expenses
        df = _transform_to_gold_schema(df, 'credit_card')

    elif 'bank' in sql_lower:
        df = _demo_loader.bank_transactions
        df = _transform_to_gold_schema(df, 'bank')

    elif 'receipt' in sql_lower:
        df = _demo_loader.receipts
        df = _transform_to_gold_schema(df, 'receipts')

    elif 'client' in sql_lower:
        # Clients table doesn't need transformation (not transactions)
        return _demo_loader.clients

    elif 'kg_entities' in sql_lower:
        # Knowledge Graph entities - return empty DataFrame for now (KG not implemented in demo)
        logger.warning(f"Knowledge Graph not implemented in demo mode, returning empty entity results")
        return pd.DataFrame(columns=['entity_id', 'canonical_name', 'aliases'])

    else:
        # Default to credit card expenses (main transaction table)
        logger.warning(f"Could not route query to specific data source, defaulting to credit_card transactions")
        df = _demo_loader.credit_card_expenses
        df = _transform_to_gold_schema(df, 'credit_card')

    # Apply filters
    if not df.empty:
        # Filter by source (if specified and not already filtered)
        if 'source' in filters and 'source' in df.columns:
            df = df[df['source'] == filters['source']]

        # Filter by date range
        if 'start_date' in filters and 'date' in df.columns:
            df = df[df['date'] >= pd.to_datetime(filters['start_date'])]
        if 'end_date' in filters and 'date' in df.columns:
            df = df[df['date'] <= pd.to_datetime(filters['end_date'])]

    logger.info(f"Loaded {len(df)} records from demo data (transformed to gold schema)")
    return df
