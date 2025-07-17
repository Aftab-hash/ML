
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)
plt.scatter(X,y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)
print(r2_score(y_test,y_pred))

class GradientDescent:
    def __init__(self,learning_rate,iteration,M,B):
        self.M = M
        self.B = B
        self.learning_rate = learning_rate
        self.iteration = iteration


    def fit(self,X,Y):
        #calculatin B and M using GD
        for i in range(self.iteration):
            loss_slope_b = -2 * np.sum(Y - self.M*X.ravel() - self.B)
            loss_slope_m = -2 * np.sum((Y - self.M*X.ravel() - self.B) * X.ravel())

            self.B = self.B - (self.learning_rate * loss_slope_b)
            self.M = self.M - (self.learning_rate * loss_slope_m)

        print(self.M, self.B)

    def predict(self,X):
        return self.M * X + self.B

#creating object of class GD
GD = GradientDescent(0.001,50,100, -120)

#now we can use function fit by passing training dataset and function predict by passing test datasets

GD.fit(X_train,y_train)
y_pred = GD.predict(X_test)
r2_score(y_test,y_pred)
print(r2_score(y_test,y_pred))

