import sys
import os

ROOT_DIR = r'C:\Users\HP\Documents\assessment'
sys.path.append(ROOT_DIR)

import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# ── Configuration ──────────────────────────────────────────────
DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 5432,
    'database': 'finance_db',
    'user'    : 'postgres',
    'password': 'Pixie@1510'   # ← replace with your password
}

TABLE_NAME   = 'finance_economic_data'
PARQUET_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'processed.parquet')


# ── Load Data ──────────────────────────────────────────────────
def load_data():
    print(f"[1/4] Loading processed data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"      Rows    : {df.shape[0]}")
    print(f"      Columns : {df.shape[1]}")
    return df


# ── Connect to PostgreSQL ──────────────────────────────────────
from urllib.parse import quote_plus

def get_engine():
    print(f"[2/4] Connecting to PostgreSQL...")
    password = quote_plus(DB_CONFIG['password'])  # encodes special characters
    connection_string = (
        f"postgresql+psycopg2://"
        f"{DB_CONFIG['user']}:{password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
        f"/{DB_CONFIG['database']}"
    )
    engine = create_engine(connection_string)
    print(f"      Connected to {DB_CONFIG['database']} on {DB_CONFIG['host']}")
    return engine

# ── Write to PostgreSQL ────────────────────────────────────────
def write_to_postgres(df, engine):
    print(f"[3/4] Writing to PostgreSQL table '{TABLE_NAME}'...")
    df.to_sql(
        name      = TABLE_NAME,
        con       = engine,
        if_exists = 'replace',
        index     = False,
        chunksize = 500
    )
    print(f"      ✅ {len(df)} rows written to '{TABLE_NAME}'")


# ── Verify Load ────────────────────────────────────────────────
def verify_load(engine):
    print(f"[4/4] Verifying load...")
    with engine.connect() as conn:
        # Row count
        result = conn.execute(text(f'SELECT COUNT(*) FROM {TABLE_NAME}'))
        count = result.scalar()
        print(f"      Rows in PostgreSQL : {count}")

        # Date range
        result = conn.execute(text(f'SELECT MIN("Date"), MAX("Date") FROM {TABLE_NAME}'))
        min_date, max_date = result.fetchone()
        print(f"      Date range         : {min_date} → {max_date}")

        # Null check
        result = conn.execute(text(f'SELECT COUNT(*) FROM {TABLE_NAME} WHERE "Close Price" IS NULL'))
        nulls = result.scalar()
        print(f"      Nulls in Close Price: {nulls}")


# ── Master Function ────────────────────────────────────────────
def run_load():
    print("=" * 50)
    print("   LOAD TO POSTGRESQL")
    print("=" * 50)

    df     = load_data()
    engine = get_engine()
    write_to_postgres(df, engine)
    verify_load(engine)

    print("\n" + "=" * 50)
    print("   ✅ LOAD COMPLETE")
    print("=" * 50)


if __name__ == '__main__':
    run_load()