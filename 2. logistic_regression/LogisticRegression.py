import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class LogisticRegression:

    def __init__(self, alpha=1, tol=0.0001, max_iter=100):
        self.beta = np.zeros(1)
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    # X should be (n, m), y should be (n,1), beta should be (m, 1)
    def fit(self, X, y):

        # Vectorization
        def cost(beta, X, y):
            return np.mean(-y * np.log(self.sigmoid(X @ beta)) - (1 - y) * np.log(1 - self.sigmoid(X @ beta)))

        # Vectorization
        def gradient(beta, X, y):
            return (1 / len(X)) * X.T @ (self.sigmoid(X @ beta) - y)

        beta = np.zeros((X.shape[1], 1))     # (m,1)
        loss = cost(beta, X, y)
        beta = beta - self.alpha * gradient(beta, X, y)
        loss_new = cost(beta, X, y)
        i = 1
        while np.abs(loss_new - loss) > self.tol:       # using gradient descent
            loss = loss_new
            beta = np.array(beta) - self.alpha * gradient(beta, X, y)
            loss_new = cost(beta, X, y)
            i += 1
            if i > self.max_iter:
                break
        self.beta = beta

    def predict(self, X):
        prob = self.sigmoid(X @ self.beta)
        return (prob >= 0.5).astype(int)

    def predict_prob(self, X):
        return self.sigmoid(X @ self.beta)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data.data
    x = StandardScaler().fit_transform(x)
    y = data.target.reshape(-1, 1)
    clf = LogisticRegression()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    print(accuracy_score(y, y_pred))


