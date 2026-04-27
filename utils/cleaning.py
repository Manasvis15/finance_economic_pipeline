# def drop_duplicates(df):
#     before = df.shape[0]
#     df = df.drop_duplicates()
#     print(f"Removed {before - df.shape[0]} duplicates")
#     return df

# def fill_nulls(df, strategy='mean'):
#     for col in df.select_dtypes(include='number').columns:
#         if strategy == 'mean':
#             df[col] = df[col].fillna(df[col].mean())
#         elif strategy == 'median':
#             df[col] = df[col].fillna(df[col].median())
#         elif strategy == 'zero':
#             df[col] = df[col].fillna(0)
    
#     # Fill non-numeric columns with mode
#     for col in df.select_dtypes(exclude='number').columns:
#         df[col] = df[col].fillna(df[col].mode()[0])
    
#     print("Nulls filled successfully")
#     print(df.isnull().sum())
#     return df

import pandas as pd

def drop_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - df.shape[0]} duplicates")
    return df

def fill_nulls(df, strategy='mean'):
    for col in df.select_dtypes(include='number').columns:
        if strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'zero':
            df[col] = df[col].fillna(0)

    for col in df.select_dtypes(exclude='number').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Nulls filled successfully")
    print(df.isnull().sum())
    return df

def drop_columns(df, columns: list):
    df = df.drop(columns=columns, errors='ignore')
    print(f"Dropped columns: {columns}")
    return df

def rename_columns(df, mapping: dict):
    df = df.rename(columns=mapping)
    print(f"Renamed columns: {mapping}")
    return df

def fix_dtypes(df, dtype_map: dict):
    # Example dtype_map: {'age': 'int', 'date': 'datetime64'}
    for col, dtype in dtype_map.items():
        if dtype == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            df[col] = df[col].astype(dtype, errors='ignore')
    print(f"Fixed dtypes: {dtype_map}")
    return df