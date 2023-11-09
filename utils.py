import pandas as pd
from sklearn.preprocessing import LabelEncoder

def sanitize_column_names(df): # remove typical unwanted characters from column names
    df.columns = [col.translate(str.maketrans('[]<>{}', '____')) for col in df.columns]
    return df


def view_data(df):
    for column in df.columns:
        nan_count = df[column].isna().sum()
        nan_percentage = round(nan_count / len(df) * 100, 1)
        print(f'The column {column} has {nan_percentage}% NaN values.')


def get_data(df, columns_to_drop, labels, thresh=0.8):
    y = df[labels]
    df_clean = df.drop(columns=columns_to_drop + labels)
    threshold = thresh * len(df_clean)
    df_clean = df_clean.dropna(axis=1, thresh=threshold)
    combined = pd.concat([df_clean, y], axis=1)
    combined_clean = combined.dropna()
    df_clean = combined_clean[df_clean.columns]
    y = combined_clean[labels]
    le = LabelEncoder()
    columns_to_encode = df_clean.select_dtypes(include=['object', 'string']).columns
    for column in columns_to_encode:
        df_clean[column] = le.fit_transform(df_clean[column])
    X = df_clean
    return X, y