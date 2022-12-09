import numpy as np
from data import load_mnist
from GENIE import GENIE
from feature_selection import algo1, algo2
from svm import svm
import time
import sys

def get_prediction(X_train, y_train, nn_model, query):
    # get nearest neighbour of a single query
    nn = nn_model.get_NN([query])[0]
    
    # feature selection on query + nn
    X_nn, y_nn = X_train[nn], y_train[nn]
    if len(np.unique(y_nn)) == 1: # if all neighbour has the same class, no training needed
        pred = [y_nn[0]]
    else:
        best_features_arg, best_feature = algo(X_nn, y_nn)
        
        # train classifier on best_features
        X_nn_fs, query_fs = X_nn[:, best_features_arg], query[best_features_arg]
        model = svm(X_nn_fs, y_nn)
        pred = model.predict(query_fs.reshape(1,-1))
    
    return pred, best_features_arg, best_feature

def verbose_feature_impt(AVG_FEATURE_IMPT_ARG, AVG_FEATURE_IMPT):
    print()
    print("--- Feature Importance ---")
    print("Feature      |       Importance")
    for i in range(len(AVG_FEATURE_IMPT_ARG)):
        print(f"x{AVG_FEATURE_IMPT_ARG[i]}       |       {AVG_FEATURE_IMPT[i]:.04f}")
    print(f"Average number of Features: {AVG_FEATURES:.02f}")
    print()


def main(method, data, algo, idx, top):

    # get data
    X_train, y_train,X_test, y_test  = load_mnist()

    # initialize feature impt
    FEATURE_IMPT = np.zeros(X_train.shape[1])
    NUM_FEATURES = 0
    NUM_TEST = 0

    # track time
    start_time = time.time()

    # train GENIE to get local NN classifer
    nn_model = GENIE(X_train)

    # get overall test dataset acc
    if method == "test":
        print(f"--- Predicting on Index {idx} ---")
        # get query
        query = X_test[idx]

        # get prediction
        pred, best_features_arg, best_feature = get_prediction(X_train, y_train, query, nn_model)

        # add to feature impt
        FEATURE_IMPT[best_features_arg] += best_feature
        NUM_FEATURES += len(best_feature)
        NUM_TEST +=1

        print(f"True Label: {y_test[idx]}, Predicted: {pred[0]}")

    elif method == "evaluate":
        
        print(f"--- Evaluating on {idx} Testing Data ---")
        ACC = np.zeros(0)
        for i in range(idx):
            # get query
            query = X_test[i]

            # get prediction
            pred, best_features_arg, best_feature = get_prediction(X_train, y_train, query, nn_model)

            # add to feature impt
            FEATURE_IMPT[best_features_arg] += best_feature
            NUM_FEATURES += len(best_feature)
            NUM_TEST += 1

            # add to acc list
            ACC = np.append(ACC, pred == y_test[i])
            
        print(f"Accuracy: {ACC.mean():.04f}")
        

    end_time = time.time()

    # verbose for top feature importance
    AVG_FEATURE_IMPT = FEATURE_IMPT / NUM_TEST
    AVG_FEATURES = NUM_FEATURES / NUM_TEST
    AVG_FEATURE_IMPT_ARG = np.argsort(AVG_FEATURE_IMPT)[::-1][:top]
    AVG_FEATURE_IMPT = AVG_FEATURE_IMPT[AVG_FEATURE_IMPT_ARG]
    
    verbose_feature_impt(AVG_FEATURE_IMPT_ARG, AVG_FEATURE_IMPT)

    print(f"Time Taken: {end_time - start_time:.04f}s")
    return 1


if __name__ == "__main__":

    if len(sys.argv) == 5:
        method, data, algo, idx = sys.argv[1:5]
        top = 10
    else:
        method, data, algo, idx, top = sys.argv[1:6]

    # format arg
    try:
        idx = int(idx)
        algo = int(algo)
        top = int(top)
    except ValueError:
        raise ValueError("Wrong format entered")

    algo = algo1 if algo == 1 else algo2

    main(method, data, algo, idx, top)