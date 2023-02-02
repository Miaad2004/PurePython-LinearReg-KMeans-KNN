import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class K_Nearest_Neighbor:
    def __init__(self, K, X, Y):
        self.neighborsCount = K
        self.data = [point for point in zip(X, Y)]      # Merge labels & the features into a 2D list

    @staticmethod
    def calDistance(p1, p2):          # Calculate Euclidean distance
        p2, p1 = np.array(p2), np.array(p1)
        return np.sqrt(np.sum((p2 - p1) ** 2))
        
    def predict(self, inputs):
        predictions = []
        for i in tqdm(range(len(inputs))):
            x = inputs[i]
            distances = []
            for point in self.data:   # point[0] is the features & point[1] is the label
                distances.append([self.calDistance(x, point[0]), point[1]])            # Calculate the distances between x & all the points in the dataset
            
            distances.sort(key=lambda x: x[0])     # Sort the list based on distances

            neighborsClasses = np.array(distances)[:self.neighborsCount,1]             # Get the classes of nearest neighbors(based on k value)
            classes, counts = np.unique(neighborsClasses, return_counts=True)         
            predictions.append(classes[np.argmax(counts)])

        return predictions


def visualize(x, y, predictions):
    plt.scatter(x, y, c=predictions)
    plt.show()

def main():
    # Generate data
    train_x = np.vstack((np.random.uniform(0, 15, size=(300,2)), np.random.uniform(5, 30, size=(300, 2)))).reshape(-1, 2)
    train_y = np.vstack((np.zeros(300), np.ones(300))).reshape(-1)
    test_x = np.vstack((np.random.uniform(0, 15, size=(300,2)), np.random.uniform(5, 30, size=(300, 2)))).reshape(-1, 2)
    test_y = np.vstack((np.zeros(300), np.ones(300))).reshape(-1)

    # Create the model
    K = 10
    model = K_Nearest_Neighbor(K, train_x, train_y)

    # Evaluate the model
    predictions = model.predict(test_x)
    #print(predictions)
    visualize(test_x[:, 0], test_x[:, 1], predictions)

if __name__ == '__main__':
    main()
