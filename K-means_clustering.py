import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class K_Means_Clustering:
    def __init__(self, K, x, featuresCount):
        self.x = x
        self.clustersCount = K
        self.centroids = np.random.uniform(np.min(x), np.max(x), size=(self.clustersCount, featuresCount))         # Randomly initialize the centroids
        self.clusters = [[] for i in range(self.clustersCount)]    # Create an empty list to save each cluster's data

    @staticmethod
    def calDist(p1, p2):            # Calculates Euclidean distance
        p2, p1 = np.array(p2), np.array(p1)
        return np.sqrt(np.sum((p2 - p1) ** 2))

    def calNearestCentroid(self, point):     # Calculates the nearest centroid for the given point
        minDistance = self.calDist(point, self.centroids[0])
        minClusterIndex = 0                  # To save index of the nearest cluster

        for i, centroid in enumerate(self.centroids):    # Finds the smallest distance
            newDistance = self.calDist(point, centroid)
            if newDistance < minDistance:
                minDistance = newDistance
                minClusterIndex = i
        
        return minClusterIndex

    def updateCentroids(self):         # Replaces each centroid's position with the mean value of its cluster
        for i in range(len(self.centroids)):
            self.centroids[i] = np.mean(self.clusters[i])

        np.nan_to_num(self.centroids, copy=False)       # If a cluster is empty set its centroid to 0

    def fit(self, epochs):
        for epoch in tqdm(range(epochs)):
            for i, point in enumerate(self.x):          # Add each point to the nearest cluster
                self.clusters[self.calNearestCentroid(point)].append(point)

            self.updateCentroids()
            for List in self.clusters:   # Empty all the clusters 
                List.clear() 
    
    def predict(self, x):
        predictions = [self.calNearestCentroid(point) for point in x]
        return np.array(predictions)

    def getCentroids(self):
        return self.centroids
        
# ==================================================
def visualize(data, predictions):
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c=predictions)
    plt.show()

def main():
    # Generate random data (number_Of_Clusters=4)
    x_train = np.vstack((np.random.uniform(1, 10, size=(200,2)),
                        np.random.uniform(9, 20, size=(200,2)),
                        np.random.uniform(19, 30, size=(200,2)),
                        np.random.uniform(29, 40, size=(200,2))
                        ))
    x_test = np.vstack((np.random.uniform(1, 10, size=(200,2)),
                        np.random.uniform(9, 20, size=(200,2)),
                        np.random.uniform(19, 30, size=(200,2)),
                        np.random.uniform(29, 40, size=(200,2))
                        ))
    
    np.random.shuffle(x_train)
    np.random.shuffle(x_test)

    # Create the model
    K = 4       
    model = K_Means_Clustering(K, x_train, featuresCount=2)
    model.fit(200)
    print(model.getCentroids())

    # Visualize
    predictions = model.predict(x_test)
    visualize(x_test, predictions)

if __name__ == '__main__':
    main()