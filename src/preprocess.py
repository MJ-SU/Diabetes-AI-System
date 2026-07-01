import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

FEATURE_COLUMNS = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
]


def get_model_dir():
    return Path(__file__).resolve().parent.parent / 'models'


def get_scaler_path():
    return get_model_dir() / 'scaler.pkl'

#讀取資料函數
def load_data():
    data_path = Path(__file__).resolve().parent.parent / 'data' / 'diabetes.csv'
    df = pd.read_csv(data_path)
    return df

#清理缺失值函數
def clean_data(df):
    cols=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    return df

#填補缺失值函數(使用中位數填補避免離群值影響)
def fill_missing_values(df):
    cols=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    return df

#特徵值/標籤分離函數
def split_features_labels(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

#分割訓練集和測試集函數
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def prepare_raw_data():
    df = load_data()
    df = clean_data(df)
    df = fill_missing_values(df)
    X, y = split_features_labels(df)
    return split_data(X, y)

#標準化特徵值函數
def scale_features(X_train, X_test, scaler=None):
    if scaler is None:
        scaler = fit_scaler(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def transform_features(X, scaler):
    return scaler.transform(X)


def save_scaler(scaler, scaler_path=None):
    if scaler_path is None:
        scaler_path = get_scaler_path()

    scaler_path.parent.mkdir(exist_ok=True)
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)


def load_scaler(scaler_path=None):
    if scaler_path is None:
        scaler_path = get_scaler_path()

    with open(scaler_path, 'rb') as file:
        return pickle.load(file)


def transform_input_features(X, scaler=None):
    if scaler is None:
        scaler = load_scaler()

    return scaler.transform(X)

#Tensor轉換函數
def to_tensor(X_train_scaled, X_test_scaled, y_train, y_test):
    import torch

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)  # Reshape to (n_samples, 1)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)  # Reshape to (n_samples, 1)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

#整合預處理後的數據供其他檔案使用
def prepare_data(use_saved_scaler=False, save_scaler_to_disk=False):
    X_train, X_test, y_train, y_test = prepare_raw_data()
    scaler = None

    if use_saved_scaler:
        scaler_path = get_scaler_path()
        if scaler_path.exists():
            scaler = load_scaler(scaler_path)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler=scaler)

    if save_scaler_to_disk:
        save_scaler(scaler)

    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = to_tensor(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler

if __name__ == "__main__":
    print("Starting preprocessing...", flush=True)

#資料庫訊息
    df = load_data()
    print("=== head ===")
    print(df.head())

    print("\n=== info ===")
    df.info()

    print("\n=== describe ===")
    print(df.describe())

    print("\n=== shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns)

#缺失值處理
    df = clean_data(df)
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

#填補缺失值
    df = fill_missing_values(df)
    print("\n=== Missing Values After Filling ===")
    print(df.isnull().sum())

#特徵值/標籤分離
    X, y = split_features_labels(df)
    print("\n=== Features ===")
    print(X.head())
    print("\n=== Labels ===")
    print(y.head())
    print("\n=== Features Shape ===")
    print(X.shape)
    print("\n=== Labels Shape ===")
    print(y.shape)

#分割訓練集和測試集
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\n=== Training Set Shape ===")
    print(X_train.shape, y_train.shape)
    print("\n=== Test Set Shape ===")
    print(X_test.shape, y_test.shape)

#標準化特徵值
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("\n=== Before Scaling ===")
    print(X_train[:1])
    print("\n=== After Scaling ===")
    print(X_train_scaled[:1])

#Tensor轉換
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = to_tensor(X_train_scaled, X_test_scaled, y_train, y_test)
    print("\n=== Tensor Check ===")
    print(type(X_train_tensor), X_train_tensor.shape)
    print(X_train_tensor.dtype)