import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore(df):
    print("=== Shape ===")
    print(df.shape)
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== Duplicates ===")
    print(df.duplicated().sum())
    print("\n=== Basic Stats ===")
    print(df.describe(include='all'))

def missing_value_analysis(df):
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    missing = pd.DataFrame({'Missing Count': total, 'Missing %': percent})
    missing = missing[missing['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if missing.empty:
        print("✅ No missing values found!")
        return

    print(missing)

    plt.figure(figsize=(12, 5))
    sns.barplot(x=missing.index, y=missing['Missing %'], palette='Reds_r')
    plt.title('Missing Values by Column (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Missing %')
    plt.tight_layout()
    plt.show()

def distribution_plots(df):
    numeric_cols = df.select_dtypes(include='number').columns
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution of Economic Indicators', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        square=True
    )
    plt.title('Correlation Heatmap — Economic Indicators')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def outlier_detection(df):
    numeric_cols = df.select_dtypes(include='number').columns
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()

    outlier_summary = {}

    for i, col in enumerate(numeric_cols):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_summary[col] = len(outliers)

        sns.boxplot(y=df[col].dropna(), ax=axes[i], color='lightcoral')
        axes[i].set_title(f'{col}\n({len(outliers)} outliers)')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Outlier Detection — Economic Indicators', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n=== Outlier Summary (IQR method) ===")
    for col, count in outlier_summary.items():
        print(f"{col}: {count} outliers")

def skewness_kurtosis(df):
    numeric_df = df.select_dtypes(include='number')
    stats = pd.DataFrame({
        'Skewness': numeric_df.skew(),
        'Kurtosis': numeric_df.kurt()
    }).sort_values('Skewness', ascending=False)

    print("=== Skewness & Kurtosis ===")
    print(stats)

    # Flag highly skewed columns
    skewed = stats[abs(stats['Skewness']) > 1]
    if not skewed.empty:
        print(f"\n⚠️  Highly skewed columns (may need transformation):")
        print(skewed.index.tolist())
def count_plot(df, col, title=None, color='steelblue'):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df, color=color)
    plt.title(title if title else f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def scatter_plot(df, x_col, y_col, title=None, color='steelblue'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, color=color, marker='o')
    plt.title(title if title else f'{x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def pair_plot(df, title=None):
    numeric_df = df.select_dtypes(include='number')
    sns.pairplot(numeric_df, diag_kind='kde', plot_kws={'alpha': 0.5, 'color': 'steelblue'})
    plt.suptitle(title if title else 'Pairplot of Numeric Columns', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def line_plot(df, x_col, y_col, title=None, color='steelblue'):
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=x_col, y=y_col, data=df, color=color, linewidth=1.5)
    plt.title(title if title else f'{y_col} Over {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def box_plot(df, x_col=None, y_col=None, title=None, color='lightcoral'):
    plt.figure(figsize=(12, 6))
    if x_col and y_col:
        sns.boxplot(x=x_col, y=y_col, data=df, color=color)
    elif y_col:
        sns.boxplot(y=df[y_col], color=color)
    else:
        # Plot all numeric columns
        numeric_df = df.select_dtypes(include='number')
        numeric_df.plot(kind='box', figsize=(14, 6), color=color)
    plt.title(title if title else 'Box Plot')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def violin_plot(df, x_col=None, y_col=None, title=None):
    plt.figure(figsize=(12, 6))
    if x_col and y_col:
        sns.violinplot(x=x_col, y=y_col, data=df, palette='coolwarm')
    elif y_col:
        sns.violinplot(y=df[y_col], color='steelblue')
    else:
        # Plot all numeric columns stacked
        numeric_df = df.select_dtypes(include='number')
        melted = numeric_df.melt(var_name='Indicator', value_name='Value')
        sns.violinplot(x='Indicator', y='Value', data=melted, palette='coolwarm')
    plt.title(title if title else 'Violin Plot')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def stock_index_plot(df, date_col, cols, title='Stock Index Trends'):
    """
    Plots multiple stock index columns over time on the same chart.
    
    Parameters:
        df       : DataFrame
        date_col : Name of the date column
        cols     : List of stock index column names to plot
                   e.g. ['SP500', 'NASDAQ', 'DowJones']
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    plt.figure(figsize=(14, 6))
    for col in cols:
        if col in df.columns:
            plt.plot(df[date_col], df[col], linewidth=1.5, label=col)
        else:
            print(f"⚠️  Column '{col}' not found in DataFrame")

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def forex_plot(df, date_col, cols, title='Forex Exchange Rates'):
    """
    Plots multiple forex/currency pair columns over time.

    Parameters:
        df       : DataFrame
        date_col : Name of the date column
        cols     : List of forex column names to plot
                   e.g. ['USD_INR', 'EUR_USD', 'GBP_USD']
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, n * 4))

    if n == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        if col in df.columns:
            axes[i].plot(df[date_col], df[col], linewidth=1.5, color='steelblue')
            axes[i].set_title(f'{col} Over Time')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(col)
            axes[i].tick_params(axis='x', rotation=45)
        else:
            print(f"⚠️  Column '{col}' not found in DataFrame")

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
def time_series_trends(df, date_col='date'):
    if date_col not in df.columns:
        print(f"⚠️  No '{date_col}' column found")
        return

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    numeric_cols = df.select_dtypes(include='number').columns
    n = len(numeric_cols)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].plot(df[date_col], df[col], color='steelblue', linewidth=1.5)
        axes[i].set_title(f'{col} Over Time')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(col)
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Economic Indicators Over Time', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def eda_summary(df):
    numeric_df = df.select_dtypes(include='number')
    categorical_df = df.select_dtypes(exclude='number')

    print("=" * 50)
    print("           EDA SUMMARY REPORT")
    print("=" * 50)
    print(f"Total Rows              : {len(df)}")
    print(f"Total Columns           : {df.shape[1]}")
    print(f"Numeric Columns         : {len(numeric_df.columns)}")
    print(f"Categorical Columns     : {len(categorical_df.columns)}")
    print(f"Duplicate Rows          : {df.duplicated().sum()}")
    print(f"Columns with Nulls      : {df.isnull().any().sum()}")
    print(f"Total Missing Values    : {df.isnull().sum().sum()}")

    skewed = numeric_df.skew()
    skewed = skewed[abs(skewed) > 1]
    print(f"Highly Skewed Columns   : {len(skewed)}")

    outlier_count = 0
    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count += len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
    print(f"Total Outliers Detected : {outlier_count}")
    print("=" * 50)