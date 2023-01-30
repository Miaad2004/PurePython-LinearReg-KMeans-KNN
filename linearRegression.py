import numpy as np

class OptimizerSGD:
    def __init__(self, learningRate, decay=0):
        self.LR = learningRate
        self.currentLR = learningRate

    def updateDerivatives(self, model, x, y):   
        model.dIntercept = np.sum(-2 * (y - (model.slope * x + model.intercept)))     # Derivative w.r.t the intercept 
        model.dSlope =  np.sum(-2 * x * (y - (model.slope * x + model.intercept)))    # Derivative w.r.t the slope

    def updateParameters(self, model):        
        # Gradient decent
        model.intercept -= self.currentLR * model.dIntercept
        model.slope -= self.currentLR * model.dSlope

class LinearRegression:
    def __init__(self):
        # Randomly initialize parameters
        self.intercept = np.random.randint(100)
        self.slope = np.random.randint(100)
        # Derivatives
        self.dIntercept = 1
        self.dSlope = 1

    def calculateLoss(self, predictions, labels):     # Mean Square Error
        return np.mean((predictions - labels) ** 2)

    def predict(self, x):
        return self.slope * x + self.intercept
    
    def train(self, x, y, LR=1e-3, epochs=1000, minLoss=-1):
        x, y = np.array(x), np.array(y)
        opt = OptimizerSGD(LR)

        for i in range(epochs):
            opt.updateDerivatives(self, x, y)
            opt.updateParameters(self)
            print(f"Epoch {i} -> Loss={self.calculateLoss(self.predict(x), y):.2f}")       # Log

            if(minLoss != -1 and self.calculateLoss(self.predict(x), y) < minLoss):        # Stop training if loss is lower than a certain value
                break
    
    def getParams(self):
        return (self.slope, self.intercept)

def main():
    # Hyperparameters
    epoch = 10000
    minLoss = 1e-8     
    lr = 1e-3

    # data
    x = [1,2,3,4,5,6]
    y = [3,5,7,9,11,13]

    # training
    model = LinearRegression()
    model.train(x, y, LR=lr, epochs=epoch, minLoss=minLoss)
    print(f"Slope={model.getParams()[0]:.3f} Intercept={model.getParams()[1]:.3f}")
    print(model.predict(10))

if __name__ == '__main__':
    main()