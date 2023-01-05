import numpy as np
from model import Model
from sklearn.feature_selection import RFE


# helper function
def get_best_features(FEATURE_IMPT, X, y, model_name, reduce_ratio = 0.15):

    # sort cv in descending order
	FEATURE_IMPT_ARG = np.argsort(FEATURE_IMPT)[::-1]
	FEATURE_IMPT = FEATURE_IMPT[FEATURE_IMPT_ARG]

    # max number of features
	num_feature = len(FEATURE_IMPT)
	max_num_feature = int(reduce_ratio * num_feature)

	CV_RESULTS = np.zeros(0)
	x_fs = np.ones((X.shape[0],1))
	for i in range(max_num_feature):
        
        # get best feature
		idx = FEATURE_IMPT_ARG[i]
		x_fs = np.concatenate((x_fs, X[:,idx].reshape(-1,1)), axis=1)
        
        # train model
		_model = Model(x_fs[:,1:],y,model_name)
		acc = _model.cv()
        
        # store cv result
		CV_RESULTS = np.append(CV_RESULTS, acc)
    
    # get number of feature that resulted in highest cv
	num_feature = np.argmax(CV_RESULTS) + 1
	best_features_arg = FEATURE_IMPT_ARG[:num_feature]
	best_features = FEATURE_IMPT[:num_feature]

	return best_features_arg, best_features

# Algoirthm 1
def algo1(X, y, model_name):
	
	# get baseline accuracy
	baseline_model = Model(X,y, model_name)
	baseline = baseline_model.cv()
    
	FEATURE_IMPT = np.zeros(0)

	num_feature = X.shape[1]
	for i in range(num_feature):
        
        # remove 1 feature from X
		Xi = np.delete(X,i, axis=1)
        
        # train model
		_model = Model(Xi, y, model_name)
        # get CV accuracy
		acc = _model.cv()
        # get feature importance
		imp = acc - baseline

        # store acc
		FEATURE_IMPT = np.append(FEATURE_IMPT, imp)
    
    # get best features
	best_features_arg, best_feature = get_best_features(FEATURE_IMPT, X, y, model_name)

	return best_features_arg, best_feature


# Algorithm 2
def algo2(X, y, model_name, N = 100):
    
    # define num feature and midpoint
	num_features = X.shape[1]
	midpoint = num_features // 2
	feature_list = np.arange(num_features)

	CONTAINS_F, NO_F = np.zeros(num_features), np.zeros(num_features)
    # set seed
	np.random.seed(42)
	for i in range(N):
        # randomly divide into 2 equal halves
		permu = np.random.permutation(feature_list)
		first_idx, sec_idx = permu[:midpoint], permu[midpoint:]
		first_half, sec_half = X[:, first_idx], X[:, sec_idx]

        # cv on both halves
		_model1 = Model(first_half, y, model_name)
		acc1 = _model1.cv()
		_model2 = Model(sec_half, y, model_name)
		acc2 = _model2.cv()

        # save acc
		CONTAINS_F[first_idx] += acc1
		NO_F[sec_idx] += acc1
		CONTAINS_F[sec_idx] += acc2
		NO_F[first_idx] += acc2
    
	FEATURE_IMPT = CONTAINS_F / NO_F

    # get best features
	best_features_arg, best_feature = get_best_features(FEATURE_IMPT, X, y, model_name)

	return best_features_arg, best_feature


# sklearn RFE
def rfe(X, y, model_name):

    # get estimator
    estimator = Model(X,y, model_name).model

    # feature selection
    selector = RFE(estimator, n_features_to_select=0.15)
    selector = selector.fit(X,y)

    # get features importance
    FEATURE_IMPT = 1 / selector.ranking_ # best feature with largest importance

    # get best features
    best_features_arg, best_feature = get_best_features(FEATURE_IMPT, X, y, model_name)

    return best_features_arg, best_feature
    
    
def feature_selection(fs_name, X, y, model_name):
	            
	# get feature selection algo
	if fs_name == "FS1":
		fs_fn = algo1
	elif fs_name == "FS2":
		fs_fn = algo2
	elif fs_name == "RFE":
		fs_fn = rfe
	else:
		print("[ERROR] Unknown Feature Selection Algorithm")
		raise Exception("Unknown Feature Selection Algorithm")
		
	return fs_fn(X, y, model_name)