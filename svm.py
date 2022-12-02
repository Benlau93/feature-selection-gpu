from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def svm_cv(X,y):
    clf = SVC(kernel="linear", random_state = RANDOM_STATE)
    score = cross_val_score(clf, X, y, cv=5, scoring = "accuracy")

    return score.mean()


def svm(X,y):
    clf = SVC(kernel="linear", random_state = RANDOM_STATE)
    clf.fit(X, y)

    return clf