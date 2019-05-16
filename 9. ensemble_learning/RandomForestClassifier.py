from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from copy import deepcopy
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd


class MyRandomForestClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=50, max_depth=None, min_samples_split=2, max_features=None):
        # 可替换为其他的基学习器，来得到BaggingClassifier
        self.base_estimator = DecisionTreeClassifier(max_depth=max_depth,
                                                     min_samples_split=min_samples_split,
                                                     max_features=max_features)
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators_ = []
        l = X.shape[0]
        for i in range(self.n_estimators):
            # random sample index of dataset
            sample = np.floor(l * np.random.random_sample(l)).astype(int)
            sample_X = X[sample]
            sample_y = y[sample]
            estimator = deepcopy(self.base_estimator)
            estimator.fit(sample_X, sample_y)
            self.estimators_.append(estimator)
        return self

    def predict(self, X):

        def mode(l):
            c = Counter(l)
            return sorted(c.items(), key=lambda x: x[1], reverse=True)[0][0]

        predict_matrix = np.array([estimator.predict(X) for estimator in self.estimators_])
        return [mode(predict_matrix[:, i]) for i in range(predict_matrix.shape[1])]

    def predict_proba(self, X):
        proba = sum(estimator.predict_proba(X)
                    for estimator in self.estimators_)
        proba /= self.n_estimators
        return proba


def data_clean(data, test):
    train_y = data['target'].apply(lambda x: 0 if x == ' <=50K' else 1)
    test_y = test['target'].apply(lambda x: 0 if x == ' <=50K.' else 1)
    raw_data_x = data.drop(['target'], axis=1)
    raw_test_x = test.drop(['target'], axis=1)
    train_x = pd.get_dummies(raw_data_x)
    test_x = pd.get_dummies(raw_test_x)
    test_x['native-country_ Holand-Netherlands'] = 0
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    data = pd.read_csv('adult_data.csv')
    test = pd.read_csv('adult_test.csv')
    train_x, train_y, test_x, test_y = data_clean(data, test)
    random_forest_clf = MyRandomForestClassifier(
        max_depth=40, min_samples_split=0.01, max_features='auto',
        n_estimators=150)
    random_forest_clf.fit(train_x.values, train_y.values)
    random_pred_score = random_forest_clf.predict_proba(test_x.values)[:, 1]
    score = roc_auc_score(test_y, random_pred_score)
    print(score)