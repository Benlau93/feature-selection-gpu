import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import sys

from data import load_mnist
from GENIE import GENIE
from feature_selection import algo1, algo2
from svm import svm


def get_prediction(X_train, y_train, nn_model, algo ,query):

    # if knn
    if "KNN" in algo:
        # get nearest neighbour of a single query
        nn = nn_model.get_NN([query])[0]
        X_, y_ = X_train[nn], y_train[nn]
    else:
        X_, y_ = X_train, y_train

    # if all neighbour has the same class, no training needed
    if len(np.unique(y_)) == 1:
        pred = [y_[0]]
        best_features_arg, best_feature = [], []
    
    else:
        # if feature selection algo is selected
        if "FS" in algo:
            # run feature selection
            algo_f = algo1 if "FS1" in algo else algo2
            best_features_arg, best_feature = algo_f(X_, y_)
            
            # filter to best features
            X_, query = X_[:, best_features_arg], query[:, best_features_arg]

        else:
            best_features_arg, best_feature = [], []


        # train model on selected features
        model = svm(X_, y_)
        # make prediction
        pred = model.predict(query.reshape(-1,X_.shape[1]))
    
    
    return pred, best_features_arg, best_feature

def verbose_feature_impt(AVG_FEATURE_IMPT_ARG, AVG_FEATURE_IMPT, AVG_FEATURES):
    print()
    print("-----    Feature Importance    -----")
    print(f"{'Feature'.ljust(18)}|{'Importance'.rjust(18)}")
    for i in range(len(AVG_FEATURE_IMPT_ARG)):
        print(f"x{str(AVG_FEATURE_IMPT_ARG[i]).ljust(17)}|{format(AVG_FEATURE_IMPT[i],'.4f').rjust(18)}")
    print(f"Average number of Features: {AVG_FEATURES:.02f}")
    print()


def main(method, data, algo, idx, top):

    # get data
    X_train, y_train,X_test, y_test  = load_mnist()

    # initialize feature impt
    num_feature = X_train.shape[1]
    FEATURE_IMPT = np.zeros(num_feature)
    NUM_FEATURES = 0
    NUM_TEST = 0

    # track time
    start_time = time.time()

    if "KNN" in algo:
        # train GENIE to get local NN classifer
        nn_model = GENIE(X_train)
    else:
        nn_model = None

    # run method
    print()
    if method == "test":
        print(f"--- Predicting on Index {idx} [Algorithm {algo}]---")
        # get query
        query = X_test[idx].reshape(-1,num_feature)

        # get prediction
        pred, best_features_arg, best_feature = get_prediction(X_train, y_train, nn_model, algo, query)

        # add to feature impt
        FEATURE_IMPT[best_features_arg] += best_feature
        NUM_FEATURES += len(best_feature)
        NUM_TEST +=1

        print(f"True Label: {y_test[idx]}, Predicted: {pred[0]}")

    elif method == "evaluate":
        
        print(f"--- Evaluating on {idx:,} Testing Data [Algorithm {algo}]---")
        if "KNN" in algo:
            # initialize y_true and pred
            y_true = np.zeros(0)
            y_pred = np.zeros(0)

            for i in range(idx):
                # get query
                query = X_test[i].reshape(-1,num_feature)

                    # get prediction
                    pred, best_features_arg, best_feature = get_prediction(X_train, y_train, nn_model, algo, query)

                # add to feature impt
                if len(best_features_arg) >0:
                    FEATURE_IMPT[best_features_arg] += best_feature
                    NUM_FEATURES += len(best_feature)
                    NUM_TEST += 1

                # add to y list
                y_true = np.append(y_true, y_test[i])
                y_pred = np.append(y_pred, pred)
        
        else:
            # initialize y_true
            y_true = y_test[:idx]

            # get query
            query = x_test[:idx].reshape(-1, num_feature)
            # get prediction
            y_pred, best_features_arg, best_feature = get_prediction(X_train, y_train, nn_model, algo, query)
        
        # get classification metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average = "weighted")

        print(f"Accuracy: {acc:.04f}, Precision: {precision:.04f}, Recall: {recall:.04f}, F1: {f1:.04f}")
        

    end_time = time.time()

    if "FS" in algo:
        # verbose for top feature importance
        AVG_FEATURE_IMPT = FEATURE_IMPT / NUM_TEST
        AVG_FEATURES = NUM_FEATURES / NUM_TEST
        AVG_FEATURE_IMPT_ARG = np.argsort(AVG_FEATURE_IMPT)[::-1][:top]
        AVG_FEATURE_IMPT = AVG_FEATURE_IMPT[AVG_FEATURE_IMPT_ARG]
        
        verbose_feature_impt(AVG_FEATURE_IMPT_ARG, AVG_FEATURE_IMPT, AVG_FEATURES)

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
        top = int(top)
        algo = algo.upper()
    except ValueError:
        raise ValueError("Wrong format entered")

    main(method, data, algo, idx, top)