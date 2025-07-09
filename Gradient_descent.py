class GradientDescent:
    def __init__(self,learning_rate,iterationn,M,B):
        self.learning_rate = learning_rate
        self.iterationn = iterationn
        self.M = M
        self.B = B


    def fit(self,X,Y):
        #calculatin B and M using GD
        for i in range(self.iterationn):
            loss_slope_b = -2 * np.sum(Y - self.M*X.ravel() - self.B)
            loss_slope_m= -2 * np.sum((Y - self.M*X.ravel() - self.B) * X.ravel())

            self.M = self.B - (self.learning_rate * loss_slope_b)
            self.B = self.M - (self.learning_rate * loss_slope_m)

            print(self.M, self.B)

    def predict(self,X):
        return self.M * X - self.B

#creating object of class GD
GD = GradientDescent(0.001,50,100, 120)

#now we can use function fit by passing training dataset and function predict by passing test datasets



