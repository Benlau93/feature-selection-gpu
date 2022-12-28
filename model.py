from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def cv(X,y, model_name):
    if model_name == "SVM":
        clf = SVC(kernel="linear", random_state = RANDOM_STATE)
    elif model_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    else:
        raise Exception("Unknown Model")
    score = cross_val_score(clf, X, y, cv=5, scoring = "accuracy")

    return np.nanmean(score) # handle CV with nan acc


def model(X,y, model_name):

    if model_name == "SVM":
        clf = SVC(kernel="linear", random_state = RANDOM_STATE)
    elif model_name == "RF":
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    else:
        raise Exception("Unknown Model")
    
    clf.fit(X,y)

    return clf