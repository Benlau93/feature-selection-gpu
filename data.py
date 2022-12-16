import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import os

DATA_DIR = r"C:\Users\ben_l\Desktop\DA Toolkit\Feature Selection\Python Implementation\data"
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

    # encode categorical features
    enc = OrdinalEncoder()
    X_train = enc.fit_transform(X_train)
    X_test = enc.transform(X_test)

    return X_train, y_train.values, X_test, y_test.values