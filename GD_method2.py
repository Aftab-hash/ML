class GradientDescent:
    def __init__(self, learning_rate, iterations,theta):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = theta

    def fit(self,X,Y):
        for i in range (self.iterations):
            self.theta = self.theta - self.learning_rate * (X @ (Y-(X @ self.theta.T)))
            #theta(t+1) = theta(t) - alpha[ sumation (i=0 to n)  X(i) * ( theta.T * X(i) - Y(i)]
    def predict(self,X):
        return X @ self.theta.T


GD = GradientDescent(0.001,50,0)
