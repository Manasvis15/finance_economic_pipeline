import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
ALL_NUMERIC     = STOCK_COLS + MACRO_COLS + MARKET_COLS + FINANCE_COLS + FOREX_COLS + BUSINESS_COLS


# ── 1. Descriptive Statistics ──────────────────────────────────
def descriptive_statistics(df):
    print("=== Descriptive Statistics ===\n")
    stats_df = df[ALL_NUMERIC].describe().T
    stats_df['skewness'] = df[ALL_NUMERIC].skew()
    stats_df['kurtosis'] = df[ALL_NUMERIC].kurt()
    stats_df['cv%'] = (df[ALL_NUMERIC].std() / df[ALL_NUMERIC].mean()) * 100
    print(stats_df.round(3))
    return stats_df


# ── 2. Correlation Analysis ────────────────────────────────────
def correlation_analysis(df):
    corr = df[ALL_NUMERIC].corr()

    # Full heatmap
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, square=True)
    plt.title('Full Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Top correlations
    corr_pairs = corr.unstack().reset_index()
    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]
    corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
    corr_pairs = corr_pairs.sort_values('Abs Correlation', ascending=False).drop_duplicates(subset='Abs Correlation')

    print("\n=== Top 15 Strongest Correlations ===")
    print(corr_pairs.head(15).to_string(index=False))


# ── 3. Growth Rates & Percentage Change ───────────────────────
def growth_rates(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    growth_cols = ['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)',
                   'Corporate Profits (Billion USD)', 'Consumer Spending (Billion USD)']

    for col in growth_cols:
        df[f'{col}_pct_change'] = df[col].pct_change() * 100

    print("=== Growth Rates (% Change) ===")
    pct_cols = [f'{col}_pct_change' for col in growth_cols]
    print(df[pct_cols].describe().round(3))

    # Plot
    fig, axes = plt.subplots(len(growth_cols), 1, figsize=(14, len(growth_cols) * 3))
    for i, col in enumerate(growth_cols):
        axes[i].plot(df[DATE_COL], df[f'{col}_pct_change'], color='steelblue', linewidth=1)
        axes[i].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axes[i].set_title(f'{col} — % Change Over Time')
        axes[i].set_xlabel('Date')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    return df


# ── 4. Rolling Statistics ──────────────────────────────────────
def rolling_statistics(df, window=12):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    roll_cols = ['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)', 'Interest Rate (%)']

    fig, axes = plt.subplots(len(roll_cols), 1, figsize=(14, len(roll_cols) * 4))

    for i, col in enumerate(roll_cols):
        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std  = df[col].rolling(window=window).std()

        axes[i].plot(df[DATE_COL], df[col], label='Actual', alpha=0.4, color='steelblue')
        axes[i].plot(df[DATE_COL], rolling_mean, label=f'{window}M Rolling Mean', color='orange', linewidth=2)
        axes[i].fill_between(df[DATE_COL],
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.2, color='orange', label='±1 Std Dev')
        axes[i].set_title(f'{col} — Rolling Statistics')
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle(f'Rolling Statistics (Window={window} months)', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()


# ── 5. Volatility Analysis ─────────────────────────────────────
def volatility_analysis(df, window=12):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    vol_cols = ['Close Price', 'Crude Oil Price (USD per Barrel)',
                'Gold Price (USD per Ounce)', 'Forex USD/EUR']

    fig, axes = plt.subplots(len(vol_cols), 1, figsize=(14, len(vol_cols) * 3))

    for i, col in enumerate(vol_cols):
        volatility = df[col].rolling(window=window).std()
        axes[i].plot(df[DATE_COL], volatility, color='crimson', linewidth=1.5)
        axes[i].set_title(f'{col} — Rolling Volatility ({window}M)')
        axes[i].set_xlabel('Date')
        axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle('Volatility Analysis', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()


# ── 6. Z-Score Anomaly Detection ──────────────────────────────
def zscore_anomaly_detection(df, threshold=3):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    anomaly_summary = {}

    fig, axes = plt.subplots(len(MACRO_COLS), 1, figsize=(14, len(MACRO_COLS) * 3))

    for i, col in enumerate(MACRO_COLS):
        z = np.abs(stats.zscore(df[col].dropna()))
        anomalies = df[col][z > threshold]
        anomaly_summary[col] = len(anomalies)

        axes[i].plot(df[DATE_COL], df[col], color='steelblue', linewidth=1, label=col)
        axes[i].scatter(df[DATE_COL][z > threshold], anomalies,
                        color='red', zorder=5, label='Anomaly', s=40)
        axes[i].set_title(f'{col} — Anomaly Detection (Z > {threshold})')
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle('Z-Score Anomaly Detection — Macro Indicators', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

    print("\n=== Anomaly Summary ===")
    for col, count in anomaly_summary.items():
        print(f"{col}: {count} anomalies detected")


# ── 7. Trend Decomposition ─────────────────────────────────────
def trend_decomposition(df, col='Close Price', period=12):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)

    series = df[col].dropna()

    result = seasonal_decompose(series, model='additive', period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    result.observed.plot(ax=axes[0], title='Observed', color='steelblue')
    result.trend.plot(ax=axes[1], title='Trend', color='orange')
    result.seasonal.plot(ax=axes[2], title='Seasonality', color='green')
    result.resid.plot(ax=axes[3], title='Residual', color='crimson')

    plt.suptitle(f'Trend Decomposition — {col}', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()


# ── 8. Hypothesis Testing ──────────────────────────────────────
def hypothesis_testing(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print("=== Hypothesis Testing (T-Test: Pre vs Post 2008) ===\n")

    test_cols = ['GDP Growth (%)', 'Inflation Rate (%)',
                 'Unemployment Rate (%)', 'Close Price']

    for col in test_cols:
        pre  = df[df[DATE_COL].dt.year < 2008][col].dropna()
        post = df[df[DATE_COL].dt.year >= 2008][col].dropna()
        t_stat, p_value = stats.ttest_ind(pre, post)

        significance = "✅ Significant difference" if p_value < 0.05 else "❌ No significant difference"
        print(f"{col}:")
        print(f"   Pre-2008 Mean  : {pre.mean():.3f}")
        print(f"   Post-2008 Mean : {post.mean():.3f}")
        print(f"   T-Statistic    : {t_stat:.3f}")
        print(f"   P-Value        : {p_value:.5f}  → {significance}\n")


# ── 9. Stationarity Test ───────────────────────────────────────
def stationarity_test(df):
    print("=== Augmented Dickey-Fuller Stationarity Test ===\n")

    test_cols = ['Close Price', 'GDP Growth (%)', 'Inflation Rate (%)',
                 'Interest Rate (%)', 'Gold Price (USD per Ounce)',
                 'Crude Oil Price (USD per Barrel)']

    results = []
    for col in test_cols:
        series = df[col].dropna()
        adf_result = adfuller(series)
        stationary = "✅ Stationary" if adf_result[1] < 0.05 else "⚠️  Non-Stationary"
        results.append({
            'Column'        : col,
            'ADF Statistic' : round(adf_result[0], 4),
            'p-value'       : round(adf_result[1], 5),
            'Result'        : stationary
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


# ── 10. Regression Analysis ────────────────────────────────────
def regression_analysis(df):
    df = df.copy().dropna()

    # Predict Close Price from macro indicators
    features = ['GDP Growth (%)', 'Inflation Rate (%)',
                 'Unemployment Rate (%)', 'Interest Rate (%)',
                 'Consumer Confidence Index', 'Crude Oil Price (USD per Barrel)']
    target = 'Close Price'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    r2 = model.score(X_scaled, y)

    coef_df = pd.DataFrame({
        'Feature'     : features,
        'Coefficient' : model.coef_
    }).sort_values('Coefficient', ascending=False)

    print(f"=== Regression Analysis ===")
    print(f"Target  : {target}")
    print(f"R² Score: {r2:.4f}\n")
    print(coef_df.to_string(index=False))

    # Plot coefficients
    plt.figure(figsize=(10, 5))
    colors = ['steelblue' if c > 0 else 'crimson' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f'Regression Coefficients — Predicting {target}')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()


# ── 11. Stock Market Analysis ──────────────────────────────────
def stock_market_analysis(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    # Daily return
    df['Daily Return (%)'] = ((df['Close Price'] - df['Open Price']) / df['Open Price']) * 100
    # Price range
    df['Price Range']      = df['Daily High'] - df['Daily Low']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    axes[0].plot(df[DATE_COL], df['Close Price'], color='steelblue', linewidth=1.5)
    axes[0].set_title('Close Price Over Time')

    axes[1].bar(df[DATE_COL], df['Daily Return (%)'],
                color=df['Daily Return (%)'].apply(lambda x: 'green' if x >= 0 else 'red'),
                width=10)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Daily Returns (%)')

    axes[2].plot(df[DATE_COL], df['Trading Volume'], color='purple', linewidth=1)
    axes[2].set_title('Trading Volume Over Time')

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Stock Market Analysis', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

    print(f"\n=== Stock Summary ===")
    print(f"Avg Daily Return  : {df['Daily Return (%)'].mean():.3f}%")
    print(f"Max Daily Return  : {df['Daily Return (%)'].max():.3f}%")
    print(f"Min Daily Return  : {df['Daily Return (%)'].min():.3f}%")
    print(f"Avg Price Range   : {df['Price Range'].mean():.3f}")


# ── 12. Forex Analysis ─────────────────────────────────────────
def forex_analysis(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for i, col in enumerate(FOREX_COLS):
        axes[i].plot(df[DATE_COL], df[col], linewidth=1.5, color='steelblue')
        axes[i].set_title(f'{col} Over Time')
        axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle('Forex Exchange Rate Trends', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

    # Correlation with macro indicators
    forex_corr = df[FOREX_COLS + MACRO_COLS].corr()[FOREX_COLS].drop(FOREX_COLS)
    print("\n=== Forex Correlation with Macro Indicators ===")
    print(forex_corr.round(3))


# ── 13. Commodity Analysis ─────────────────────────────────────
def commodity_analysis(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    commodity_cols = ['Crude Oil Price (USD per Barrel)', 'Gold Price (USD per Ounce)']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for i, col in enumerate(commodity_cols):
        axes[i].plot(df[DATE_COL], df[col], linewidth=1.5, color='goldenrod')
        axes[i].set_title(f'{col} Over Time')
        axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle('Commodity Price Trends', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

    # Correlation with stock and forex
    corr_cols = commodity_cols + ['Close Price'] + FOREX_COLS
    print("\n=== Commodity Correlations ===")
    print(df[corr_cols].corr().round(3))