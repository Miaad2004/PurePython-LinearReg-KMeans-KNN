import numpy as np
import matplotlib.pyplot as plt

class OptimizerSGD:
    def __init__(self, learningRate, decay=0):
        self.LR = learningRate
        self.currentLR = learningRate

    def updateDerivatives(self, model, x, y):   
        model.dIntercept = np.sum(-2 * (y - (np.dot(x, model.slopes.T) + model.intercept)))             # Derivative w.r.t the intercept 
        model.dSlopes =  np.dot(-2 * (y - (np.dot(x, model.slopes.T) + model.intercept)), x)            # Derivative w.r.t the slopes

    def updateParameters(self, model):        
        # Gradient decent
        model.intercept -= self.currentLR * model.dIntercept
        model.slopes -= self.currentLR * model.dSlopes

class LinearRegression:
    def __init__(self, featuresCount):
        self.featuresCount = featuresCount
        # Randomly initialize parameters
        self.intercept = np.random.uniform(-50, 50)
        self.slopes = np.random.uniform(-50, 50, size=(featuresCount))
        # Derivatives
        self.dIntercept = 1
        self.dSlopes = np.ones(featuresCount)

    def calculateLoss(self, predictions, labels):     # Mean Square Error
        return np.mean((predictions - labels) ** 2)

    def predict(self, x):
        x = np.array(x)
        if self.featuresCount == 1 and x.shape[-1] != 1:
            x = x.reshape(-1,1)
        return np.dot(x, self.slopes.T) + self.intercept
    
    def train(self, x, y, LR=1e-3, epochs=1000, minLoss=-1):
        x, y = np.array(x) , np.array(y)
        if self.featuresCount == 1 and x.shape[-1] != 1:
            x = x.reshape(-1,1)

        opt = OptimizerSGD(LR)

        for i in range(epochs):
            opt.updateDerivatives(self, x, y)
            opt.updateParameters(self)
            if(i % 10 == 0):
                print(f"Epoch {i} -> Loss={self.calculateLoss(self.predict(x), y):.2f}")       # Log

            if(minLoss != -1 and self.calculateLoss(self.predict(x), y) < minLoss):        # Stop training if loss is smaller than a certain value
                break
    
    def getParams(self):
        return (self.slopes, self.intercept)


# ==================================================
def visualize(x, y, slope, intercept):
    fig, axis = plt.subplots()
    plt.scatter(x, y)
    axis.axline((0, intercept), slope=slope, color='C3')
    plt.show()

def main():
    # Hyperparameters
    epoch = 100
    minLoss = 1e-4
    lr = 1e-2

    # Generate random data
    randomNums = np.random.uniform(0, 1, size=70)
    x_train = np.hstack((randomNums,
                         np.random.uniform(0, 1, size=5)
                         ))
    y_train = np.hstack(((randomNums * 3) + 2,
                         np.random.uniform(0, 1, size=5)
                         ))

    x_test = np.random.uniform(0, 1, size=50)
    y_test = (x_test * 3) + 2

    # training
    model = LinearRegression(featuresCount=1)
    model.train(x_train, y_train, LR=lr, epochs=epoch, minLoss=minLoss)
    print(f"\nSlopes={model.getParams()[0]} Intercept={model.getParams()[1]:.3f}")

    # Evaluate
    predictions = model.predict(x_test)
    print(f"Test_Loss= {model.calculateLoss(predictions, y_test):.3f}")
    visualize(x_train, y_train, model.getParams()[0][0], model.getParams()[1])

if __name__ == '__main__':
    main()