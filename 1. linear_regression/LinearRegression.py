import numpy as np
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.metrics import mean_squared_error


def mse_score(y_predict, y_test):
    """
    input:y_predict(ndarray):预测值
          y_test(ndarray):真实值
    ouput:mse(float):mse损失函数值
    """

    return 1 / len(y_predict) * sum([np.square(y - p) for y, p in zip(y_test, y_predict)])


class LinearRegression:
    """
    通过normal equation来完成线性回归
    """
    def __init__(self):
        """初始化线性回归模型"""
        self.theta = None

    def fit_normal(self, train_data, train_label):
        """
        input:train_data(ndarray):训练样本
              train_label(ndarray):训练标签
        """

        ones = np.ones((len(train_data), 1))
        train_data = np.column_stack((train_data, ones))
        # normal equation
        self.theta = np.linalg.inv(train_data.T @ train_data) @ train_data.T @ train_label

        return self

    def predict(self, test_data):
        """
        input:test_data(ndarray):测试样本
        """

        ones = np.ones((len(test_data), 1))
        test_data = np.column_stack((test_data, ones))
        return test_data @ self.theta


if __name__ == '__main__':
    housing = fetch_california_housing()
    x = housing.data
    y = housing.target
    clf = LinearRegression()
    clf.fit_normal(x, y)
    y_pred = clf.predict(x)
    print(mean_squared_error(y, y_pred))