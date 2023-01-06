from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
    
class Model:
	def __init__(self, X, y, model_name, model_type = "classifier", model_params = {}):
		
		def get_model(self):
			if self.model_name == "SVM":
				
				# set up model parameters
				C =  self.model_params["C"] if self.model_params.get("C") else 1.0
				kernel = self.model_params["kernel"] if self.model_params.get("kernel") else "linear"
				class_weight = self.model_params.get("class_weight")
				
				# initialize model
				if self.model_type == "classifier":
					model = SVC(C=C,  kernel=kernel, class_weight=class_weight, random_state=RANDOM_STATE)
				elif self.model_type == "regressor":
					model = SVR(C=C,  kernel=kernel, random_state = RANDOM_STATE)
			
			elif self.model_name in ["RF","TREE"]:
				
				# set up model parameters
				n_estimators = self.model_params["n_estimators"] if self.model_params.get("n_estimators") else 100
				criterion = self.model_params["criterion"] if self.model_params.get("criterion") else "gini"
				max_depth = self.model_params["max_depth"] if self.model_params.get("max_depth") else 10
				max_features = self.model_params["max_features"] if self.model_params.get("max_features") else "sqrt"
				bootstrap = self.model_params["bootstrap"] if self.model_params.get("bootstrap") else True
				class_weight = self.model_params.get("class_weight")
				
				# initialize model
				if self.model_type == "classifier":
					if self.model_name == "RF":
						model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap,  class_weight=class_weight, n_jobs=-1, random_state = RANDOM_STATE)
					else:
						model = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_features=max_features,class_weight=class_weight, random_state=RANDOM_STATE)
				elif self.model_type == "regressor":
					if self.model_name == "RF":
						model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap, n_jobs=-1, random_state = RANDOM_STATE)	
					else:
						model = DecisionTreeRegressor(criterion=criterion,max_depth=max_depth,max_features=max_features, random_state=RANDOM_STATE)
			elif self.model_name == "LINEAR":

				# initialize model
				if self.model_type == "classifier":
					model = LogisticRegression(random_state = RANDOM_STATE, n_jobs=-1)
				elif self.model_type == "regressor":
					model = LinearRegression(n_jobs=-1)
			

			return model
			
		self.X = X
		self.y = y
		self.model_name = model_name.upper()
		self.model_type = model_type
		self.model_params = model_params
		self.model = get_model(self)
		
		
	def cv(self):
			
		# compute time taken
		start_time = time.time()
		
		# compute cv scores
		cv_score = cross_val_score(self.model, self.X, self.y, cv=5, scoring="accuracy")
		
		# store time taken
		self.cv_time = time.time() - start_time
		
		return np.nanmean(cv_score) # handle CV with nan acc
		
		
	def predict(self, sample):
		
		# train model
		model = self.model.fit(self.X, self.y)
		
		# predict
		return model.predict(sample) 
		

		