import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os

DATA_DIR = r".\data"
RANDOM_STATE = 42

#  MNIST
def load_mnist():
    mnist_train = pd.read_csv(os.path.join(DATA_DIR,"mnist_train.csv"))
    mnist_test = pd.read_csv(os.path.join(DATA_DIR,"mnist_test.csv"))

    # split to training and testing set
    X_train, y_train = mnist_train.drop("label", axis=1).values , mnist_train["label"].values
    X_test, y_test = mnist_test.drop("label", axis=1).values , mnist_test["label"].values

    # min max scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# SPLICE
def load_splice():
    df = pd.read_csv(os.path.join(DATA_DIR,"splice_data.csv"))
    X, y = df.drop(["id","label"], axis=1), df["label"]

    # split to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1000, random_state = RANDOM_STATE)

    # encode categorical variables
    enc = OrdinalEncoder()
    X_train = enc.fit_transform(X_train)
    X_test = enc.transform(X_test)

    # min max scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train.values, X_test, y_test.values


# SUN
def load_sun():
    df = pd.read_csv(os.path.join(DATA_DIR,"sun_data.csv"))
    X, y = df.drop(["label"], axis=1).values, df["label"].values

    # split into training and testing set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=500, random_state=RANDOM_STATE)

    for train_index, test_index in sss.split(X,y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

    # min max scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# load custom data
def load_custom(file_name):
    df = pd.read_csv(os.path.join(DATA_DIR,f"{file_name}.csv"))
    X, y = df.drop(["label"], axis=1), df["label"].values

    # convert categorical variables
    cat_cols = X.select_dtypes("object").columns
    X[cat_cols] = X[cat_cols].astype("category")
    for c in cat_cols:
        X[c] = X[c].cat.codes
    
    # convert to nump
    X = X.values
    # split into training and testing set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)

    for train_index, test_index in sss.split(X, y):
        print(f"Training data: {len(train_index)}, Test data: {len(test_index)}")
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]


    # min max scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test