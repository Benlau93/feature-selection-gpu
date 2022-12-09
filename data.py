import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#  MNIST
def load_mnist():
    mnist_train = pd.read_csv("./data/mnist_train.csv")
    mnist_test = pd.read_csv("./data/mnist_test.csv")

    # split to training and testing set
    X_train, y_train = mnist_train.drop("label", axis=1).values , mnist_train["label"].values
    X_test, y_test = mnist_test.drop("label", axis=1).values , mnist_test["label"].values

    # min max scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test