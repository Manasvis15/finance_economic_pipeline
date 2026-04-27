import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# ── Column Groups ──────────────────────────────────────────────
DATE_COL        = 'Date'
STOCK_COLS      = ['Open Price', 'Close Price', 'Daily High', 'Daily Low', 'Trading Volume']
MACRO_COLS      = ['GDP Growth (%)', 'Inflation Rate (%)', 'Unemployment Rate (%)', 'Interest Rate (%)']
MARKET_COLS     = ['Consumer Confidence Index', 'Crude Oil Price (USD per Barrel)',
                   'Gold Price (USD per Ounce)', 'Real Estate Index']
FINANCE_COLS    = ['Government Debt (Billion USD)', 'Corporate Profits (Billion USD)',
                   'Retail Sales (Billion USD)', 'Consumer Spending (Billion USD)']
FOREX_COLS      = ['Forex USD/EUR', 'Forex USD/JPY']
BUSINESS_COLS   = ['Bankruptcy Rate (%)', 'Mergers & Acquisitions Deals',
                   'Venture Capital Funding (Billion USD)']
CATEGORICAL_COLS = ['Stock Index']

ALL_NUMERIC     = STOCK_COLS + MACRO_COLS + MARKET_COLS + FINANCE_COLS + FOREX_COLS + BUSINESS_COLS


# ── 1. Date Features ───────────────────────────────────────────
def extract_date_features(df):
    """
    Extracts year, month, quarter, day of week from Date column.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df['Year']        = df[DATE_COL].dt.year
    df['Month']       = df[DATE_COL].dt.month
    df['Quarter']     = df[DATE_COL].dt.quarter
    df['Day_of_Week'] = df[DATE_COL].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Is_Quarter_End'] = df[DATE_COL].dt.is_quarter_end.astype(int)

    print("✅ Date features extracted: Year, Month, Quarter, Day_of_Week, Is_Quarter_End")
    return df


# ── 2. Feature Engineering ─────────────────────────────────────
def feature_engineering(df):
    """
    Creates new meaningful columns from existing ones.
    """
    df = df.copy()

    # Stock features
    df['Daily_Return (%)']     = ((df['Close Price'] - df['Open Price']) / df['Open Price']) * 100
    df['Price_Range']          = df['Daily High'] - df['Daily Low']
    df['Price_Momentum']       = df['Close Price'] - df['Close Price'].shift(1)
    df['Volatility_7D']        = df['Close Price'].rolling(window=7).std()
    df['Volatility_30D']       = df['Close Price'].rolling(window=30).std()
    df['Moving_Avg_7D']        = df['Close Price'].rolling(window=7).mean()
    df['Moving_Avg_30D']       = df['Close Price'].rolling(window=30).mean()

    # Macro features
    df['Real_Interest_Rate']   = df['Interest Rate (%)'] - df['Inflation Rate (%)']
    df['Misery_Index']         = df['Inflation Rate (%)'] + df['Unemployment Rate (%)']
    df['GDP_Inflation_Ratio']  = df['GDP Growth (%)'] / (df['Inflation Rate (%)'].replace(0, np.nan))

    # Financial health features
    df['Profit_Margin']        = df['Corporate Profits (Billion USD)'] / df['Retail Sales (Billion USD)'].replace(0, np.nan)
    df['Spending_to_Debt']     = df['Consumer Spending (Billion USD)'] / df['Government Debt (Billion USD)'].replace(0, np.nan)
    df['VC_to_Profits']        = df['Venture Capital Funding (Billion USD)'] / df['Corporate Profits (Billion USD)'].replace(0, np.nan)

    # Forex features
    df['Forex_Spread']         = df['Forex USD/JPY'] - df['Forex USD/EUR']

    # Commodity features
    df['Oil_Gold_Ratio']       = df['Crude Oil Price (USD per Barrel)'] / df['Gold Price (USD per Ounce)'].replace(0, np.nan)

    # Market sentiment
    df['Market_Sentiment']     = (
        df['Consumer Confidence Index'] +
        df['Daily_Return (%)'] -
        df['Bankruptcy Rate (%)']
    )

    print("✅ Feature engineering complete — new columns added:")
    new_cols = ['Daily_Return (%)', 'Price_Range', 'Price_Momentum',
                'Volatility_7D', 'Volatility_30D', 'Moving_Avg_7D', 'Moving_Avg_30D',
                'Real_Interest_Rate', 'Misery_Index', 'GDP_Inflation_Ratio',
                'Profit_Margin', 'Spending_to_Debt', 'VC_to_Profits',
                'Forex_Spread', 'Oil_Gold_Ratio', 'Market_Sentiment']
    for col in new_cols:
        print(f"   + {col}")

    return df


# ── 3. Encode Categorical Columns ─────────────────────────────
def encode_categoricals(df):
    """
    Encodes categorical columns using Label Encoding and One Hot Encoding.
    """
    df = df.copy()

    # Label Encoding — for ML models
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            print(f"✅ Label encoded: {col} → {col}_encoded")

    # One Hot Encoding — for PostgreSQL and analysis
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
            df = pd.concat([df, dummies], axis=1)
            print(f"✅ One-hot encoded: {col} → {dummies.columns.tolist()}")

    return df


# ── 4. Scaling & Normalisation ─────────────────────────────────
def scale_features(df, method='minmax'):
    """
    Scales numeric columns.

    Parameters:
        method : 'minmax'   → scales to 0-1 range (good for ML)
                 'standard' → zero mean, unit variance (good for regression)
    """
    df = df.copy()

    # Only scale original numeric columns — not engineered ratios
    cols_to_scale = [col for col in ALL_NUMERIC if col in df.columns]

    if method == 'minmax':
        scaler = MinMaxScaler()
        scaled_label = 'MinMax'
    elif method == 'standard':
        scaler = StandardScaler()
        scaled_label = 'Standard'
    else:
        print("❌ Unknown method. Use 'minmax' or 'standard'")
        return df

    scaled_values = scaler.fit_transform(df[cols_to_scale])
    scaled_df = pd.DataFrame(scaled_values,
                             columns=[f'{col}_scaled' for col in cols_to_scale],
                             index=df.index)

    df = pd.concat([df, scaled_df], axis=1)
    print(f"✅ {scaled_label} scaling applied to {len(cols_to_scale)} columns")

    return df, scaler


# ── 5. Handle Skewed Columns ───────────────────────────────────
def fix_skewness(df, threshold=1.0):
    """
    Applies log transformation to highly skewed numeric columns.
    """
    df = df.copy()
    skewed_cols = []

    for col in ALL_NUMERIC:
        if col in df.columns:
            skewness = df[col].skew()
            if abs(skewness) > threshold:
                # Shift to positive before log if needed
                min_val = df[col].min()
                if min_val <= 0:
                    df[col] = df[col] + abs(min_val) + 1
                df[f'{col}_log'] = np.log1p(df[col])
                skewed_cols.append(col)

    if skewed_cols:
        print(f"✅ Log transformation applied to {len(skewed_cols)} skewed columns:")
        for col in skewed_cols:
            print(f"   + {col}_log")
    else:
        print("✅ No highly skewed columns found")

    return df


# ── 6. Save Transformed Data ───────────────────────────────────
def save_transformed(df, path='data/processed/data_transformed.csv'):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Transformed data saved to {path}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ── 7. Run Full Transformation ─────────────────────────────────
def run_transformation(df):
    """
    Runs the full transformation pipeline in sequence.
    Returns fully transformed DataFrame.
    """
    print("=" * 50)
    print("   TRANSFORMATION PIPELINE")
    print("=" * 50)

    print("\n[1/5] Extracting date features...")
    df = extract_date_features(df)

    print("\n[2/5] Feature engineering...")
    df = feature_engineering(df)

    print("\n[3/5] Encoding categoricals...")
    df = encode_categoricals(df)

    print("\n[4/5] Fixing skewness...")
    df = fix_skewness(df)

    print("\n[5/5] Scaling features...")
    df, scaler = scale_features(df, method='minmax')

    print("\n[Saving] Saving transformed data...")
    save_transformed(df)

    print("\n✅ Transformation complete!")
    print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df