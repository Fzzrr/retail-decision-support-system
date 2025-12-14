# database.py
"""
SQLite Database Module for Retail Decision Support System
Simplified version for reduced dataset (197 customers, 89K products, 399K transactions)
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
import streamlit as st

# Database configuration
DB_PATH = Path(__file__).parent / "datasets" / "retail.db"
ARCHIVED_PATH = Path(__file__).parent / "datasets" 

# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCHEMA = """
-- Campaigns Master Table
CREATE TABLE IF NOT EXISTS campaigns (
    CAMPAIGN INTEGER PRIMARY KEY,
    DESCRIPTION TEXT,
    START_DAY INTEGER,
    END_DAY INTEGER
);

-- Campaign Targeting (which households received which campaigns)
CREATE TABLE IF NOT EXISTS campaign_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    DESCRIPTION TEXT,
    household_key INTEGER,
    CAMPAIGN INTEGER,
    FOREIGN KEY (CAMPAIGN) REFERENCES campaigns(CAMPAIGN),
    FOREIGN KEY (household_key) REFERENCES customers(household_key)
);

-- Customers (Household Demographics)
CREATE TABLE IF NOT EXISTS customers (
    household_key INTEGER PRIMARY KEY,
    AGE_DESC TEXT,
    MARITAL_STATUS_CODE TEXT,
    INCOME_DESC TEXT,
    HOMEOWNER_DESC TEXT,
    HH_COMP_DESC TEXT,
    HOUSEHOLD_SIZE_DESC TEXT,
    KID_CATEGORY_DESC TEXT,
    phone_number TEXT
);

-- Products Master Table
CREATE TABLE IF NOT EXISTS products (
    PRODUCT_ID INTEGER PRIMARY KEY,
    MANUFACTURER INTEGER,
    DEPARTMENT TEXT,
    BRAND TEXT,
    COMMODITY_DESC TEXT,
    SUB_COMMODITY_DESC TEXT,
    CURR_SIZE_OF_PRODUCT TEXT
);

-- Coupon Redemptions
CREATE TABLE IF NOT EXISTS coupon_redemptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    household_key INTEGER,
    DAY INTEGER,
    COUPON_UPC INTEGER,
    CAMPAIGN INTEGER,
    FOREIGN KEY (household_key) REFERENCES customers(household_key),
    FOREIGN KEY (CAMPAIGN) REFERENCES campaigns(CAMPAIGN)
);

-- Transactions (main sales data)
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    household_key INTEGER,
    BASKET_ID INTEGER,
    DAY INTEGER,
    PRODUCT_ID INTEGER,
    QUANTITY INTEGER,
    SALES_VALUE REAL,
    STORE_ID INTEGER,
    RETAIL_DISC REAL,
    TRANS_TIME INTEGER,
    WEEK_NO INTEGER,
    COUPON_DISC REAL,
    COUPON_MATCH_DISC REAL,
    FOREIGN KEY (household_key) REFERENCES customers(household_key),
    FOREIGN KEY (PRODUCT_ID) REFERENCES products(PRODUCT_ID)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_transactions_household ON transactions(household_key);
CREATE INDEX IF NOT EXISTS idx_transactions_product ON transactions(PRODUCT_ID);
CREATE INDEX IF NOT EXISTS idx_transactions_basket ON transactions(BASKET_ID);
CREATE INDEX IF NOT EXISTS idx_transactions_day ON transactions(DAY);
CREATE INDEX IF NOT EXISTS idx_products_dept ON products(DEPARTMENT);
CREATE INDEX IF NOT EXISTS idx_products_commodity ON products(COMMODITY_DESC);
"""

# =============================================================================
# DATABASE CONNECTION & INITIALIZATION
# =============================================================================

def get_connection():
    """Get SQLite database connection."""
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

def initialize_database():
    """Create database schema (tables and indexes)."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    return True

def database_exists():
    """Check if database file exists."""
    return DB_PATH.exists()

def delete_database():
    """Delete the database file."""
    if DB_PATH.exists():
        os.remove(DB_PATH)
        return True
    return False

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_customers(conn):
    """Load hh_demographics.csv into customers table."""
    file_path = ARCHIVED_PATH / "hh_demographics.csv"
    if not file_path.exists():
        return 0, "File not found: hh_demographics.csv"
    
    df = pd.read_csv(file_path)
    df.to_sql('customers', conn, if_exists='replace', index=False)
    return len(df), None

def load_campaigns(conn):
    """Load campaign_desc.csv into campaigns table."""
    file_path = ARCHIVED_PATH / "campaign_desc.csv"
    if not file_path.exists():
        return 0, "File not found: campaign_desc.csv"
    
    df = pd.read_csv(file_path)
    df.to_sql('campaigns', conn, if_exists='replace', index=False)
    return len(df), None

def load_products(conn):
    """Load product.csv into products table."""
    file_path = ARCHIVED_PATH / "product.csv"
    if not file_path.exists():
        return 0, "File not found: product.csv"
    
    df = pd.read_csv(file_path)
    df.to_sql('products', conn, if_exists='replace', index=False)
    return len(df), None

def load_campaign_targets(conn, valid_households=None):
    """Load campaign_table.csv into campaign_targets table."""
    file_path = ARCHIVED_PATH / "campaign_table.csv"
    if not file_path.exists():
        return 0, "File not found: campaign_table.csv"
    
    df = pd.read_csv(file_path)
    if valid_households:
        df = df[df['household_key'].isin(valid_households)]
    df.to_sql('campaign_targets', conn, if_exists='replace', index=False)
    return len(df), None

def load_coupon_redemptions(conn, valid_households=None):
    """Load coupon_redempt.csv into coupon_redemptions table."""
    file_path = ARCHIVED_PATH / "coupon_redempt.csv"
    if not file_path.exists():
        return 0, "File not found: coupon_redempt.csv"
    
    df = pd.read_csv(file_path)
    if valid_households:
        df = df[df['household_key'].isin(valid_households)]
    df.to_sql('coupon_redemptions', conn, if_exists='replace', index=False)
    return len(df), None

def load_transactions(conn, valid_households=None, progress_callback=None):
    """Load transaction_data.csv into transactions table."""
    file_path = ARCHIVED_PATH / "transaction_data.csv"
    if not file_path.exists():
        return 0, "File not found: transaction_data.csv"
    
    df = pd.read_csv(file_path)
    if valid_households:
        df = df[df['household_key'].isin(valid_households)]
    
    if progress_callback:
        progress_callback(0.5, f"Loading {len(df):,} transactions...")
    
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    return len(df), None

def load_all_data(progress_callback=None):
    """
    Load all CSV files into the database.
    Returns a summary of loaded data.
    """
    initialize_database()
    conn = get_connection()
    summary = {}
    
    try:
        # 1. Load customers first (for valid household keys)
        if progress_callback:
            progress_callback(0.1, "Loading customers...")
        count, err = load_customers(conn)
        summary['customers'] = {'count': count, 'error': err}
        
        # Get valid household keys for filtering
        valid_households = None
        if err is None:
            cursor = conn.cursor()
            cursor.execute("SELECT household_key FROM customers")
            valid_households = set(row[0] for row in cursor.fetchall())
        
        # 2. Load campaigns
        if progress_callback:
            progress_callback(0.2, "Loading campaigns...")
        count, err = load_campaigns(conn)
        summary['campaigns'] = {'count': count, 'error': err}
        
        # 3. Load products
        if progress_callback:
            progress_callback(0.3, "Loading products...")
        count, err = load_products(conn)
        summary['products'] = {'count': count, 'error': err}
        
        # 4. Load campaign targets
        if progress_callback:
            progress_callback(0.4, "Loading campaign targets...")
        count, err = load_campaign_targets(conn, valid_households)
        summary['campaign_targets'] = {'count': count, 'error': err}
        
        # 5. Load coupon redemptions
        if progress_callback:
            progress_callback(0.5, "Loading coupon redemptions...")
        count, err = load_coupon_redemptions(conn, valid_households)
        summary['coupon_redemptions'] = {'count': count, 'error': err}
        
        # 6. Load transactions
        def tx_progress(pct, msg):
            if progress_callback:
                progress_callback(0.6 + pct * 0.3, msg)
        
        count, err = load_transactions(conn, valid_households, tx_progress)
        summary['transactions'] = {'count': count, 'error': err}
        
        # 7. Create indexes
        if progress_callback:
            progress_callback(0.95, "Creating indexes...")
        conn.executescript(SCHEMA)
        conn.commit()
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
    except Exception as e:
        summary['error'] = str(e)
    finally:
        conn.close()
    
    return summary

# =============================================================================
# QUERY UTILITIES
# =============================================================================

def execute_query(query, params=None):
    """Execute a SQL query and return results as DataFrame."""
    conn = get_connection()
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

def get_table_info():
    """Get list of all tables with row counts."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    table_info = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        table_info[table] = {
            'row_count': count,
            'columns': [(col[1], col[2]) for col in columns]
        }
    
    conn.close()
    return table_info

def get_table_sample(table_name, limit=100):
    """Get sample data from a table."""
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    return execute_query(query)

# =============================================================================
# BUSINESS QUERY FUNCTIONS
# =============================================================================

def get_customer_summary(household_key):
    """Get comprehensive summary for a single customer."""
    query = """
    SELECT 
        c.*,
        COUNT(DISTINCT t.BASKET_ID) as total_transactions,
        SUM(t.QUANTITY) as total_items,
        SUM(t.SALES_VALUE) as total_spend,
        MIN(t.DAY) as first_purchase_day,
        MAX(t.DAY) as last_purchase_day
    FROM customers c
    LEFT JOIN transactions t ON c.household_key = t.household_key
    WHERE c.household_key = ?
    GROUP BY c.household_key
    """
    return execute_query(query, (household_key,))

def get_campaign_performance():
    """Get campaign performance metrics."""
    query = """
    SELECT 
        c.CAMPAIGN,
        c.DESCRIPTION as campaign_type,
        c.START_DAY,
        c.END_DAY,
        COUNT(DISTINCT ct.household_key) as targeted_customers,
        COUNT(DISTINCT cr.household_key) as redeeming_customers,
        COUNT(cr.COUPON_UPC) as total_redemptions,
        ROUND(COUNT(DISTINCT cr.household_key) * 100.0 / NULLIF(COUNT(DISTINCT ct.household_key), 0), 2) as redemption_rate
    FROM campaigns c
    LEFT JOIN campaign_targets ct ON c.CAMPAIGN = ct.CAMPAIGN
    LEFT JOIN coupon_redemptions cr ON c.CAMPAIGN = cr.CAMPAIGN
    GROUP BY c.CAMPAIGN
    ORDER BY redemption_rate DESC
    """
    return execute_query(query)

def get_product_performance(limit=50):
    """Get top performing products by revenue."""
    query = f"""
    SELECT 
        p.PRODUCT_ID,
        p.DEPARTMENT,
        p.COMMODITY_DESC,
        p.BRAND,
        COUNT(DISTINCT t.BASKET_ID) as transactions,
        SUM(t.QUANTITY) as total_quantity,
        ROUND(SUM(t.SALES_VALUE), 2) as total_revenue,
        ROUND(AVG(t.SALES_VALUE), 2) as avg_transaction_value
    FROM products p
    JOIN transactions t ON p.PRODUCT_ID = t.PRODUCT_ID
    GROUP BY p.PRODUCT_ID
    ORDER BY total_revenue DESC
    LIMIT {limit}
    """
    return execute_query(query)

def get_customer_segments():
    """Get customer segments based on spending."""
    query = """
    SELECT 
        c.AGE_DESC,
        c.INCOME_DESC,
        COUNT(DISTINCT c.household_key) as customer_count,
        ROUND(AVG(customer_spend.total_spend), 2) as avg_spend,
        ROUND(AVG(customer_spend.total_transactions), 1) as avg_transactions
    FROM customers c
    JOIN (
        SELECT 
            household_key,
            SUM(SALES_VALUE) as total_spend,
            COUNT(DISTINCT BASKET_ID) as total_transactions
        FROM transactions
        GROUP BY household_key
    ) customer_spend ON c.household_key = customer_spend.household_key
    GROUP BY c.AGE_DESC, c.INCOME_DESC
    ORDER BY avg_spend DESC
    """
    return execute_query(query)

def get_department_sales():
    """Get sales by department."""
    query = """
    SELECT 
        p.DEPARTMENT,
        COUNT(DISTINCT t.BASKET_ID) as transactions,
        SUM(t.QUANTITY) as total_quantity,
        ROUND(SUM(t.SALES_VALUE), 2) as total_revenue,
        COUNT(DISTINCT p.PRODUCT_ID) as product_count
    FROM products p
    JOIN transactions t ON p.PRODUCT_ID = t.PRODUCT_ID
    GROUP BY p.DEPARTMENT
    ORDER BY total_revenue DESC
    """
    return execute_query(query)

def get_coupon_effectiveness():
    """Analyze coupon effectiveness by campaign type."""
    query = """
    SELECT 
        camp.DESCRIPTION as campaign_type,
        COUNT(DISTINCT cr.household_key) as unique_redeemers,
        COUNT(cr.id) as total_redemptions,
        ROUND(COUNT(cr.id) * 1.0 / COUNT(DISTINCT cr.household_key), 2) as redemptions_per_customer
    FROM coupon_redemptions cr
    JOIN campaigns camp ON cr.CAMPAIGN = camp.CAMPAIGN
    GROUP BY camp.DESCRIPTION
    """
    return execute_query(query)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_transaction_count():
    """Get total transaction count from database."""
    query = "SELECT COUNT(DISTINCT BASKET_ID) as count FROM transactions"
    df, err = execute_query(query)
    if err:
        return 0
    return df['count'].iloc[0] if not df.empty else 0

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_customer_count():
    """Get total customer count from database."""
    query = "SELECT COUNT(*) as count FROM customers"
    df, err = execute_query(query)
    if err:
        return 0
    return df['count'].iloc[0] if not df.empty else 0

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_product_count():
    """Get total unique product count from database."""
    query = "SELECT COUNT(DISTINCT PRODUCT_ID) as count FROM products"
    df, err = execute_query(query)
    if err:
        return 0
    return df['count'].iloc[0] if not df.empty else 0

# =============================================================================
# ANALYSIS DATA FUNCTIONS
# =============================================================================

def get_basket_options():
    """Get available options for basket configuration."""
    return {
        'group_by': {
            'BASKET_ID': 'Per Transaction (setiap keranjang belanja)',
            'household_key': 'Per Customer (semua transaksi per Customer digabung)'
        },
        'product_level': {
            'DEPARTMENT': 'Department - Fastest',
            'COMMODITY_DESC': 'Commodity - Recommended',
            'SUB_COMMODITY_DESC': 'Sub-Commodity - Slowest',
        }
    }

@st.cache_data(ttl=600, show_spinner="📊 Memuat data analisis...")  # Cache for 10 minutes
def get_analysis_data(group_by='BASKET_ID', product_level='COMMODITY_DESC'):
    """
    Get consolidated data for Association Rules, RFM, and ANN analysis.
    
    Parameters:
    -----------
    group_by : str - 'BASKET_ID' for per-transaction, 'household_key' for per-customer
    product_level : str - 'DEPARTMENT', 'COMMODITY_DESC', 'SUB_COMMODITY_DESC', 'BRAND'
    
    Returns:
    --------
    DataFrame with basket data and customer demographics
    """
    query = f"""
    SELECT 
        t.household_key,
        t.BASKET_ID,
        MIN(t.DAY) as DAY,
        GROUP_CONCAT(DISTINCT p.{product_level}) as product_list,
        c.AGE_DESC,
        c.MARITAL_STATUS_CODE,
        c.INCOME_DESC,
        c.HOMEOWNER_DESC,
        c.HH_COMP_DESC,
        c.HOUSEHOLD_SIZE_DESC,
        c.KID_CATEGORY_DESC,
        c.phone_number,
        SUM(t.QUANTITY) as total_quantity,
        ROUND(SUM(t.SALES_VALUE), 2) as total_sales
    FROM transactions t
    JOIN products p ON t.PRODUCT_ID = p.PRODUCT_ID
    JOIN customers c ON t.household_key = c.household_key
    WHERE p.{product_level} IS NOT NULL
    GROUP BY t.{group_by}
    ORDER BY t.household_key, t.DAY
    """
    return execute_query(query)

def get_product_level_sample(level='COMMODITY_DESC', limit=10):
    """Get sample values for a product level."""
    query = f"SELECT DISTINCT {level} FROM products WHERE {level} IS NOT NULL LIMIT {limit}"
    return execute_query(query)

# =============================================================================
# DEMOGRAPHIC-BASED PRODUCT AFFINITY QUERIES
# =============================================================================

def get_demographic_options():
    """Get available demographic dimensions for analysis."""
    return {
        'AGE_DESC': 'Age Group',
        'INCOME_DESC': 'Income Level',
        'MARITAL_STATUS_CODE': 'Marital Status',
        'HOMEOWNER_DESC': 'Homeowner Status',
        'HH_COMP_DESC': 'Household Composition',
        'HOUSEHOLD_SIZE_DESC': 'Household Size',
        'KID_CATEGORY_DESC': 'Kids Category'
    }

@st.cache_data(ttl=600, show_spinner="📊 Menghitung product affinity...")
def get_product_affinity_by_demographic(demo_column, product_level='DEPARTMENT', top_n=10):
    """
    Get product affinity scores for each demographic segment.
    Uses lift calculation: (% segment buying) / (% all customers buying)
    """
    query = f"""
    WITH demographic_totals AS (
        SELECT 
            c.{demo_column} as segment,
            COUNT(DISTINCT c.household_key) as segment_customers
        FROM customers c
        WHERE c.{demo_column} IS NOT NULL AND c.{demo_column} != ''
        GROUP BY c.{demo_column}
    ),
    overall_product_stats AS (
        SELECT 
            p.{product_level} as product,
            COUNT(DISTINCT t.household_key) as total_buyers,
            (SELECT COUNT(DISTINCT household_key) FROM customers) as total_customers
        FROM transactions t
        JOIN products p ON t.PRODUCT_ID = p.PRODUCT_ID
        WHERE p.{product_level} IS NOT NULL
        GROUP BY p.{product_level}
    ),
    segment_product_stats AS (
        SELECT 
            c.{demo_column} as segment,
            p.{product_level} as product,
            COUNT(DISTINCT t.household_key) as segment_buyers,
            SUM(t.QUANTITY) as total_quantity,
            ROUND(SUM(t.SALES_VALUE), 2) as total_sales
        FROM transactions t
        JOIN customers c ON t.household_key = c.household_key
        JOIN products p ON t.PRODUCT_ID = p.PRODUCT_ID
        WHERE c.{demo_column} IS NOT NULL AND c.{demo_column} != ''
            AND p.{product_level} IS NOT NULL
        GROUP BY c.{demo_column}, p.{product_level}
    )
    SELECT 
        sps.segment,
        sps.product,
        sps.segment_buyers,
        dt.segment_customers,
        ops.total_buyers,
        ops.total_customers,
        sps.total_quantity,
        sps.total_sales,
        ROUND(sps.segment_buyers * 100.0 / dt.segment_customers, 2) as segment_penetration,
        ROUND(ops.total_buyers * 100.0 / ops.total_customers, 2) as overall_penetration,
        ROUND(
            (sps.segment_buyers * 1.0 / dt.segment_customers) / 
            (ops.total_buyers * 1.0 / ops.total_customers), 
            2
        ) as affinity_index
    FROM segment_product_stats sps
    JOIN demographic_totals dt ON sps.segment = dt.segment
    JOIN overall_product_stats ops ON sps.product = ops.product
    WHERE sps.segment_buyers >= 3
    ORDER BY sps.segment, affinity_index DESC
    """
    return execute_query(query)

def get_top_products_by_segment(demo_column, segment_value, product_level='DEPARTMENT', top_n=10):
    """Get top products for a specific demographic segment."""
    query = f"""
    SELECT 
        p.{product_level} as product,
        COUNT(DISTINCT t.household_key) as unique_buyers,
        COUNT(DISTINCT t.BASKET_ID) as transactions,
        SUM(t.QUANTITY) as total_quantity,
        ROUND(SUM(t.SALES_VALUE), 2) as total_sales,
        ROUND(AVG(t.SALES_VALUE), 2) as avg_transaction
    FROM transactions t
    JOIN customers c ON t.household_key = c.household_key
    JOIN products p ON t.PRODUCT_ID = p.PRODUCT_ID
    WHERE c.{demo_column} = ?
        AND p.{product_level} IS NOT NULL
    GROUP BY p.{product_level}
    ORDER BY total_sales DESC
    LIMIT {top_n}
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(segment_value,))
        return df, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

@st.cache_data(ttl=600)
def get_demographic_distribution(demo_column):
    """Get distribution of a demographic dimension."""
    query = f"""
    SELECT 
        c.{demo_column} as segment,
        COUNT(DISTINCT c.household_key) as customers,
        COUNT(DISTINCT t.BASKET_ID) as transactions,
        ROUND(SUM(t.SALES_VALUE), 2) as total_sales,
        ROUND(AVG(customer_totals.customer_spend), 2) as avg_spend_per_customer
    FROM customers c
    LEFT JOIN transactions t ON c.household_key = t.household_key
    LEFT JOIN (
        SELECT household_key, SUM(SALES_VALUE) as customer_spend
        FROM transactions
        GROUP BY household_key
    ) customer_totals ON c.household_key = customer_totals.household_key
    WHERE c.{demo_column} IS NOT NULL AND c.{demo_column} != ''
    GROUP BY c.{demo_column}
    ORDER BY total_sales DESC
    """
    return execute_query(query)

@st.cache_data(ttl=600)
def get_segment_comparison(demo_column, product_level='DEPARTMENT'):
    """Get pivot-style comparison of product preferences across segments."""
    query = f"""
    WITH segment_product AS (
        SELECT 
            c.{demo_column} as segment,
            p.{product_level} as product,
            ROUND(SUM(t.SALES_VALUE), 2) as sales
        FROM transactions t
        JOIN customers c ON t.household_key = c.household_key
        JOIN products p ON t.PRODUCT_ID = p.PRODUCT_ID
        WHERE c.{demo_column} IS NOT NULL AND c.{demo_column} != ''
            AND p.{product_level} IS NOT NULL
        GROUP BY c.{demo_column}, p.{product_level}
    ),
    segment_totals AS (
        SELECT segment, SUM(sales) as total_sales
        FROM segment_product
        GROUP BY segment
    )
    SELECT 
        sp.segment,
        sp.product,
        sp.sales,
        st.total_sales,
        ROUND(sp.sales * 100.0 / st.total_sales, 2) as pct_of_segment_sales
    FROM segment_product sp
    JOIN segment_totals st ON sp.segment = st.segment
    ORDER BY sp.segment, sp.sales DESC
    """
    return execute_query(query)


def clear_cached_queries():
    """Clear cache for database query helpers (call after DB refresh/reload)."""
    cached_funcs = [
        get_transaction_count,
        get_customer_count,
        get_product_count,
        get_analysis_data,
        get_product_affinity_by_demographic,
        get_demographic_distribution,
        get_segment_comparison,
    ]
    for func in cached_funcs:
        try:
            func.clear()
        except AttributeError:
            pass
