import numpy as np
from svm import svm_cv


# helper function
def get_best_features(FEATURE_IMPT, X, y, reduce_ratio = 0.1):

    # sort cv in descending order
    FEATURE_IMPT_ARG = np.argsort(FEATURE_IMPT)[::-1]
    FEATURE_IMPT = FEATURE_IMPT[FEATURE_IMPT_ARG]

    # max_feature
    num_feature = len(FEATURE_IMPT)
    max_num_feature = int(reduce_ratio * num_feature)

    CV_RESULTS = np.zeros(0)
    x_fs = np.ones((X.shape[0],1))
    for i in range(max_num_feature):
        idx = FEATURE_IMPT[i]
        x_fs = np.concatenate((x_fs, X[:,idx].reshape(-1,1)), axis=1)
        acc = svm_cv(x_fs[:,1:], y)
        
        # store cv result
        CV_RESULTS = np.append(CV_RESULTS, acc)
    
    # get number of feature that resulted in highest cv
    num_feature = np.argsort(CV_RESULTS)[-1] + 1
    best_features_arg = FEATURE_IMPT_ARG[:num_feature]
    best_feature = FEATURE_IMPT[:num_feature]

    return best_features_arg, best_feature

# Algoirthm 1
def algo1(X, y):

    baseline = svm_cv(X, y)
    
    FEATURE_IMPT = np.zeros(0)

    num_feature = X.shape[1]
    for i in range(num_feature):
        Xi = np.delete(X,i, axis=1)
        acc = svm_cv(Xi, y)
        imp = acc - baseline

        # store acc
        FEATURE_IMPT = np.append(FEATURE_IMPT, imp)
    
    # get best features
    best_features_arg, best_feature = get_best_features(FEATURE_IMPT, X, y)

    return best_features_arg, best_feature


# Algorithm 2
def algo2(X, y, N = 100):
    
    # define num feature and midpoint
    num_features = X.shape[1]
    midpoint = num_features // 2
    feature_list = np.arange(num_features)

    CONTAINS_F, NO_F = np.zeros(num_features), np.zeros(num_features)
    
    for i in range(N):
        # randomly divide into 2 equal halves
        permu = np.random.permutation(feature_list)
        first_idx, sec_idx = permu[:midpoint], permu[midpoint:]
        first_half, sec_half = X[:, first_idx], X[:, sec_idx]

        # SVM cv on both halves
        acc1 = svm_cv(first_half, y)
        acc2 = svm_cv(sec_half, y)

        # save acc
        CONTAINS_F[first_idx] += acc1
        CONTAINS_F[sec_idx] += acc2
        NO_F[first_idx] += acc2
        NO_F[sec_idx] += acc1
    
    FEATURE_IMPT = CONTAINS_F / NO_F

    # get best features
    best_features_arg, best_feature = get_best_features(FEATURE_IMPT, X, y)

    return best_features_arg, best_feature




