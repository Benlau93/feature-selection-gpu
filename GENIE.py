from sklearn.neighbors import NearestNeighbors
import time

# get k nearest neighbour to build local classifier

class GENIE:
    def __init__(self, data, k=100):
        def knn(self):
            start_time = time.time()
            print("--- Training Nearest Neighbours ---")

            # training knn
            knn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
            knn.fit(data)
            end_time = time.time()
            print(f"Neighest Neighbour Trained, time taken: {end_time - start_time:.04f}s")

            return knn

        self.data = data
        self.k = k
        self.knn = knn(self)

    def get_NN(self, sample):
        nn = self.knn.kneighbors(sample, self.k, return_distance=False)

        return nn
