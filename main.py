import numpy as np
from data import load_mnist
from GENIE import GENIE
from feature_selection import algo1, algo2
from svm import svm
import time
import sys


def main(method, data, algo, idx=-1):
    # format arg
    idx = int(idx)
    algo = algo1 if algo == "1" else algo2

    # get data
    X_train, y_train,X_test, y_test  = load_mnist()

    # track time
    start_time = time.time()

    # train GENIE to get local NN classifer
    nn_model = GENIE(X_train)

    # get overall test dataset acc
    if method == "test":
        print(f"--- Predicting on Index {idx} ---")
        # get nearest neighbour of a single query
        query = X_test[idx]
        nn = nn_model.get_NN([query])[0]
        
        # feature selection on query + nn
        X_nn, y_nn = X_train[nn], y_train[nn]
        if len(np.unique(y_nn)) == 1: # if all neighbour has the same class, no training needed
            pred = [y_nn[0]]
        else:
            best_features = algo(X_nn, y_nn)
            
            # train classifier on best_features
            X_nn_fs, query_fs = X_nn[:, best_features], query[best_features]
            model = svm(X_nn_fs, y_nn)
            pred = model.predict(query_fs.reshape(1,-1))
        print(f"True Label: {y_test[idx]}, Predicted: {pred[0]}")

    elif method == "evaluate":
        
        print(f"--- Evaluating on {idx} Testing Data ---")
        ACC = np.zeros(0)
        for i in range(idx):
            # get query
            query = X_test[i]

            # get nearest neighbour
            nn = nn_model.get_NN([query])[0]

            # feature selection on query + nn
            X_nn, y_nn = X_train[nn], y_train[nn]
            if len(np.unique(y_nn)) == 1: # if all neighbour has the same class, no training needed
                pred = y_nn[0]
            else:
                best_features = algo(X_nn, y_nn)

                # train classifier on best_features
                X_nn_fs, query_fs = X_nn[:, best_features], query[best_features]
                model = svm(X_nn_fs, y_nn)
                pred = model.predict(query_fs.reshape(1,-1))[0]

            # add to acc list
            ACC = np.append(ACC, pred == y_test[i])
        print(f"Accuracy: {ACC.mean():.04f}")
        

    end_time = time.time()
    print(f"Time Taken: {end_time - start_time:.04f}s")
    return 1


if __name__ == "__main__":

    method, data, algo, idx = sys.argv[1:5]  

    main(method, data, algo, idx)