from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)
plt.scatter(X,y)

# Add bias term to X
X_b = np.c_[np.ones((X.shape[0], 1)), X]
X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)
r2_score(y_test,y_pred)
print(r2_score(y_test,y_pred))

class GradientDescent:
    def __init__(self, learning_rate, iterations, theta):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = theta

    def fit(self, X, Y):
        for i in range(self.iterations):
            gradient = X.T @ (X @ self.theta - Y)
            self.theta = self.theta - self.learning_rate * gradient
        print("Trained theta:", self.theta)

    def predict(self, X):
        return X @ self.theta.T

initial_theta = np.zeros(X_train.shape[1])
GD = GradientDescent(0.01,50,initial_theta)


GD.fit(X_train,y_train)
y_pred1 = GD.predict(X_test)
r2_score(y_test,y_pred1)
print(r2_score(y_test,y_pred1))
