import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import sys

from data import load_data
from GENIE import GENIE
from feature_selection import feature_selection
from model import Model

def verbose_feature_impt(FEATURE_IMPT, NUM_FEATURES, NUM_TEST):
	
	# compute average feature importance
	AVG_FEATURE_IMPT = FEATURE_IMPT / NUM_TEST
	AVG_FEATURES = NUM_FEATURES / NUM_TEST
	AVG_FEATURE_IMPT_ARG = np.argsort(AVG_FEATURE_IMPT)[::-1][:10]
	AVG_FEATURE_IMPT = AVG_FEATURE_IMPT[AVG_FEATURE_IMPT_ARG]
        
	print()
	print("-----    Feature Importance    -----")
	print(f"{'Feature'.ljust(18)}|{'Importance'.rjust(18)}")
	for i in range(len(AVG_FEATURE_IMPT_ARG)):
		print(f"x{str(AVG_FEATURE_IMPT_ARG[i]).ljust(17)}|{format(AVG_FEATURE_IMPT[i],'.4f').rjust(18)}")
	print(f"Average number of Features: {AVG_FEATURES:.02f}")
	print()

def verbose_time(total_time, knn_time, model_time, fs_time):
	print()
	print("-------      Time Taken      -------")
	print(f"{'Algorithm'.ljust(18)}|{'Time (s)'.rjust(12)}|{'Time (%)'.rjust(12)}")
	if knn_time >0:
		knn_per = knn_time / total_time
		print(f"{'KNN'.ljust(18)}|{format(knn_time,'.4f').rjust(12)}|{format(knn_per,'.2%').rjust(12)}")
	
	if fs_time > 0:
		fs_per = fs_time / total_time
		print(f"{'Feature Selection'.ljust(18)}|{format(fs_time,'.4f').rjust(12)}|{format(fs_per,'.2%').rjust(12)}")

	model_per = model_time / total_time
	print(f"{'Model'.ljust(18)}|{format(model_time,'.4f').rjust(12)}|{format(model_per,'.2%').rjust(12)}")

def log_result(results):

	# read current log file
	log_file = pd.read_csv("result_log.csv")

	# add current result to file
	log = pd.DataFrame({"data":[results["data"]],"algorithm":[results["algo"]],"sample":[results["sample"]],"accuracy":[results["metrics"]["acc"]],
						"precision":[results["metrics"]["precision"]], "recall":[results["metrics"]["recall"]], "f1":[results["metrics"]["f1"]],"total_time":[results["time"]["total"]],
						"knn_time":[results["time"]["knn"]], "model_time":[results["time"]["model"]],"fs_time":[results["time"]["fs"]]})

	log_file = pd.concat([log_file, log], sort=True, ignore_index=True)

	# save
	log_file.to_csv("result_log.csv", index=False)
	

def main(data, algo, idx):
	
	# load data
	X_train, y_train, X_test, y_test = load_data(data)
	
	# initialize global variables
	TIMEOUT = False
	num_feature = X_train.shape[1]
	FEATURE_IMPT = np.zeros(num_feature)
	NUM_FEATURES = 0
	NUM_TEST = 0

    # track time
	start_time = time.time()
	model_time, fs_time, knn_time = 0,0,0

	# get sample queries
	queries = X_test[:idx].reshape(-1,num_feature)
	y_true = y_test[:idx]

	# determine algorithm
	algo_list = algo.split("+")
	KNN = False
	if "KNN" in algo:
		KNN = True
		algo_list = algo_list[1:]
	try:
		model_name, fs_name = algo_list
	except ValueError:
		model_name = algo_list[0]
		fs_name = None
	
	# run algorithm
	print(f"[INFO] Evaluating on {idx} Testing Sample(s)[{algo}]...")
	
    # train nearest neighbour model
	if KNN:
		print("[INFO] Training Nearest Neighbour Model ...")
		nn_model = GENIE(X_train)
		
		# get nearest neighbour of each query
		nn = nn_model.get_NN(queries)

		# record time
		knn_time += nn_model.train_time + nn_model.nn_time
		print(f"[INFO] Nearest Neighbour Computed for all Queries")
		
		# initialize y pred
		y_pred = np.zeros(0)
			
		# loop through each query
		for i in range(idx):

			if TIMEOUT:
				idx = i
				y_true = y_test[:idx]
				break
			
			# get query
			query = queries[i]
		
			# get training data
			_nn = nn[i]
			X_nn, y_nn = X_train[_nn], y_train[_nn]
			
			if len(np.unique(y_nn)) == 1:
				_pred = [y_nn[0]]
			
			else:
			# feature selection
				if fs_name:
					if NUM_TEST == 0:
						print(f"[INFO] Running Feature Selection Algorithm ({fs_name})...")
					
					# track time
					fs_start = time.time()

					# feature selection algo
					best_features_arg, best_features = feature_selection(fs_name, X_nn, y_nn, model_name)
					
					# filter to best features only
					X_nn, query = X_nn[:, best_features_arg], query[best_features_arg]
					
					# add to feature importance matrix
					FEATURE_IMPT[best_features_arg] += best_features
					NUM_FEATURES += len(best_features)
					NUM_TEST +=1

					# record time
					fs_time += time.time() - fs_start

					if fs_time > 1800:
						TIMEOUT = True
					
					# update number of features
					num_feature = X_nn.shape[1]
				
				
				# get prediction
				# track time
				model_start = time.time()
				model = Model(X_nn, y_nn, model_name)
				_pred = model.predict(query.reshape(-1,num_feature))
				# record time
				model_time += time.time() - model_start

			# add to y_pred
			y_pred = np.append(y_pred, _pred)
		
	else: # no KNN

		# feature selection
		if fs_name:
			print(f"[INFO] Running Feature Selection Algorithm ({fs_name})...")

								
			# track time
			fs_start = time.time()
			
			# feature selection algo
			best_features_arg, best_features = feature_selection(fs_name, X_train, y_train, model_name)
			
			# filter to best features only
			X_train, queries = X_train[:, best_features_arg], queries[:, best_features_arg]
			
			# add to feature importance matrix
			FEATURE_IMPT[best_features_arg] += best_features
			NUM_FEATURES += len(best_features)
			NUM_TEST +=1

			# record time
			fs_time += time.time() - fs_start
			
			# update number of features
			num_feature = X_train.shape[1]

		# model prediction
		# track time
		model_start = time.time()
		model = Model(X_train, y_train, model_name)
		y_pred = model.predict(queries.reshape(-1,num_feature))
		# record time
		model_time += time.time() - model_start
	
	print("[INFO] Algorithm completed. Evaluating Results ...")
	# get classification metrics
	acc = accuracy_score(y_true, y_pred)
	precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average = "weighted")
	print(f"[RESULT] Accuracy: {acc:.04f}, Precision: {precision:.04f}, Recall: {recall:.04f}, F1: {f1:.04f}")
        
		
	# verbose time taken
	end_time = time.time()
	total_time = end_time - start_time
	print(f"[INFO] Evaluation done. Total Time taken: {total_time:.04f}s")

			
	# verbose most important features
	if fs_name and NUM_TEST > 0:
		verbose_feature_impt(FEATURE_IMPT, NUM_FEATURES, NUM_TEST)

	# verbose time taken
	verbose_time(total_time, knn_time, model_time, fs_time)

	# log result
	results = {"algo":algo,
				"data":data,
				"sample":idx,
				"metrics":{"acc":acc, "precision":precision,"recall":recall,"f1":f1},
				"time":{"knn":knn_time,"model":model_time,"fs":fs_time,"total":total_time}}
	
	log_result(results)

	

if __name__ == "__main__":
	data, algo, idx = sys.argv[1:4]

	# format arg
	try:
		idx = int(idx)
		data = data.lower()
		algo = algo.upper()
	except ValueError:
		print("[ERROR] Wrong format entered")
		raise Exception("Wrong format entered")

	main(data, algo, idx)