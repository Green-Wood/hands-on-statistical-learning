import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    """
    input:n_estimators(int):迭代轮数
          learning_rate(float):弱分类器权重缩减系数
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator = DecisionTreeClassifier()

    def fit(self, X, y):
        sample_weight = np.empty(X.shape[0], dtype=np.float64)
        sample_weight[:] = 1. / X.shape[0]

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        for iboost in range(self.n_estimators):
            sample_weight, estimator_weight, estimator_error, estimator = self._boost(
                X, y, sample_weight
            )

            if estimator_error > 0.5:
                break

            self.estimators_.append(estimator)
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            if estimator_error <= 0:
                break
        return self


        estimator.fit(X, y, sample_weight=sample_weight)
        y_pred = estimator.predict(X)
        estimator_error = np.mean(y_pred != y)

        # estimator is perfect
        if estimator_error <= 0:
            return 0, 1, estimator_error, estimator

        # estimator is weak
        if estimator_error > 0.5:
            return 0, 0, estimator_error, estimator

