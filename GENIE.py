from sklearn.neighbors import NearestNeighbors
import time

# get k nearest neighbour

class GENIE:
	def __init__(self, data, k=101):
    	
		def knn(self):
    		# compute training time
			start_time = time.time()
			
            # training knn
			knn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
			knn.fit(data)
            
			train_time = time.time() - start_time
    
			return knn, train_time

		self.data = data
		self.k = k
		self.knn, self.train_time = knn(self)
		self.nn_time = 0

	def get_NN(self, sample):
        
        # compute time taken
		start_time = time.time()

		nn = self.knn.kneighbors(sample, return_distance=False)
        
        # store time taken
		self.nn_time += time.time() - start_time

		return nn