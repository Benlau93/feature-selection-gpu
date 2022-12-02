import pandas as pd


#  MNIST
def load_mnist():
    mnist_train = pd.read_csv("./data/mnist_train.csv")
    mnist_test = pd.read_csv("./data/mnist_test.csv")

    return mnist_train, mnist_test