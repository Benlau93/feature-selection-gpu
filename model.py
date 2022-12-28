from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifer

import numpy as np
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def cv(X,y, model_name):
    if model_name == "SVM":
        clf = SVC(kernel="linear", random_state = RANDOM_STATE)
    elif model_name == "RF":
        clf = RandomForestClassifer(n_estimator=100, max_depth=10, n_jobs=-1)
    score = cross_val_score(clf, X, y, cv=5, scoring = "accuracy")

    return np.nanmean(score) # handle CV with nan acc


def model(X,y, model_name):
    if model_name == "SVM":
        clf = SVC(kernel="linear", random_state = RANDOM_STATE)
    elif model_name == "RF":
        clf = RandomForestClassifer(n_estimator=100, max_depth=10, n_jobs=-1)
    
    clf.fit(X,y)

    return clf