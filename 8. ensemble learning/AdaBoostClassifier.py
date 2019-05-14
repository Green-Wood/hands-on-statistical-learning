from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd


class MyAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        y = deepcopy(y)
        y[y == 0] = -1

        sample_weight = np.empty(X.shape[0], dtype=np.float64)
        sample_weight[:] = 1. / X.shape[0]

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        for iboost in range(self.n_estimators):
            sample_weight, estimator_weight, estimator_error, estimator = self._boost(
                X, y, sample_weight
            )
            self.estimators_.append(estimator)
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            if estimator_error > 0.5 or estimator_error <= 0:
                break
        return self

    def _boost(self, X, y, sample_weight):
        estimator = deepcopy(self.base_estimator)

        estimator.fit(X, y, sample_weight=sample_weight)
        y_pred = estimator.predict(X)
        estimator_error = np.mean(y_pred != y)

        if estimator_error > 0.5 or estimator_error <= 0:
            return 0, 1, estimator_error, estimator

        estimator_weight = 1 / 2 * np.log((1 - estimator_error) / estimator_error)

        sample_weight[y_pred == y] = sample_weight[y_pred == y] * np.exp(-estimator_weight)
        sample_weight[y_pred != y] = sample_weight[y_pred != y] * np.exp(estimator_weight)
        sample_weight = sample_weight / sum(sample_weight)
        return sample_weight, estimator_weight, estimator_error, estimator

    def predict(self, X):
        pred_sum = sum(weight * estimator.predict(X)
                       for weight, estimator
                       in zip(self.estimator_weights_, self.estimators_))
        return list(map(lambda s: 1 if s > 0 else 0, pred_sum))

    def predict_proba(self, X):
        proba = sum(estimator.predict_proba(X) * w
                    for estimator, w
                    in zip(self.estimators_, self.estimator_weights_))
        proba /= np.sum(self.estimator_weights_)
        return proba


def data_clean(data, test):
    train_y = data['target'].apply(lambda x: 0 if x == ' <=50K' else 1)
    test_y = test['target'].apply(lambda x: 0 if x == ' <=50K.' else 1)
    raw_data_x = data.drop(['target'], axis=1)
    raw_test_x = test.drop(['target'], axis=1)
    train_x = pd.get_dummies(raw_data_x)
    test_x = pd.get_dummies(raw_test_x)
    # fill value that doesn't exist in test dataset
    test_x['native-country_ Holand-Netherlands'] = 0
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    data = pd.read_csv('adult_data.csv')
    test = pd.read_csv('adult_test.csv')
    train_x, train_y, test_x, test_y = data_clean(data, test)
    random_clf = MyAdaBoostClassifier(
        DecisionTreeClassifier(max_depth=40, min_samples_split=0.008, max_features='auto'),
        n_estimators=150)
    random_clf.fit(train_x.values, train_y.values)
    random_pred_score = random_clf.predict_proba(test_x.values)[:, 1]
    score = roc_auc_score(test_y, random_pred_score)
    print(score)
