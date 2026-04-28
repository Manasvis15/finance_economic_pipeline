from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

ROOT_DIR = '/opt/airflow/assessment'
sys.path.append(ROOT_DIR)

# ── Default Arguments ──────────────────────────────────────────
default_args = {
    'owner'           : 'airflow',
    'retries'         : 1,
    'retry_delay'     : timedelta(minutes=5),
    'start_date'      : datetime(2024, 1, 1),
}

# ── Task Functions ─────────────────────────────────────────────
def task_ingestion():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'raw', 'finance_economic_dataset.csv'))
    print(f"✅ Ingestion complete: {df.shape[0]} rows × {df.shape[1]} columns")

def task_cleaning_transformation():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'raw', 'data_80.csv'))

    # Nulls
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude='number').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Text
    df['Stock Index'] = df['Stock Index'].astype(str).str.strip().str.upper()

    # Outliers
    for col in df.select_dtypes(include='number').columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # Feature engineering
    df['Daily_Return (%)']       = ((df['Close Price'] - df['Open Price']) / df['Open Price']) * 100
    df['Misery_Index']           = df['Inflation Rate (%)'] + df['Unemployment Rate (%)']
    df['Real_Interest_Rate (%)'] = df['Interest Rate (%)'] - df['Inflation Rate (%)']
    df['Volatility_Band (%)']    = ((df['Daily High'] - df['Daily Low']) / df['Close Price']) * 100
    df['Economic_Health_Score']  = (
        df['GDP Growth (%)'] * 0.4 +
        df['Consumer Confidence Index'] * 0.3 +
        (df['Corporate Profits (Billion USD)'] / df['Corporate Profits (Billion USD)'].max()) * 100 * 0.3
    ).round(4)

    # Date features
    df['Year']       = df['Date'].dt.year
    df['Month']      = df['Date'].dt.month
    df['Quarter']    = df['Date'].dt.quarter
    df['Month_Name'] = df['Date'].dt.strftime('%b')

    # Encode
    le = LabelEncoder()
    df['Stock_Index_encoded'] = le.fit_transform(df['Stock Index'].astype(str))

    # Save
    os.makedirs(os.path.join(ROOT_DIR, 'data', 'processed'), exist_ok=True)
    df.to_parquet(os.path.join(ROOT_DIR, 'data', 'processed', 'processed.parquet'), index=False)
    print(f"✅ Cleaning & transformation complete: {df.shape[0]} rows × {df.shape[1]} columns")

def task_load_to_postgres():
    import pandas as pd
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus

    df = pd.read_parquet(os.path.join(ROOT_DIR, 'data', 'processed', 'processed.parquet'))

    password = quote_plus('your_password')  # ← replace with your password
    engine = create_engine(
        f'postgresql+psycopg2://postgres:{password}@host.docker.internal:5432/finance_db'
    )

    df.to_sql(
        name      = 'finance_economic_data',
        con       = engine,
        if_exists = 'replace',
        index     = False,
        chunksize = 500
    )
    print(f"✅ Load complete: {len(df)} rows written to PostgreSQL")

# ── DAG Definition ─────────────────────────────────────────────
with DAG(
    dag_id            = 'finance_etl_pipeline',
    default_args      = default_args,
    description       = 'ETL pipeline for finance economic dataset',
    schedule_interval = '@daily',
    catchup           = False,
    tags              = ['finance', 'etl']
) as dag:

    ingestion = PythonOperator(
        task_id         = 'ingestion',
        python_callable = task_ingestion
    )

    cleaning_transformation = PythonOperator(
        task_id         = 'cleaning_transformation',
        python_callable = task_cleaning_transformation
    )

    load_to_postgres = PythonOperator(
        task_id         = 'load_to_postgres',
        python_callable = task_load_to_postgres
    )

    # ── Pipeline Order ─────────────────────────────────────────
    ingestion >> cleaning_transformation >> load_to_postgres