import numpy as np
from itertools import combinations
from statistics import mode
from statistics import StatisticsError
from random import choice


class OvO(object):
    def __init__(self, model):
        # 用于保存训练时各种模型的list
        self.models = []
        self.model = model

    def fit(self, train_data, train_labels):
        """
        OvO的训练阶段，将模型保存到self.models中
        :param train_data: 训练集数据，类型为ndarray
        :param train_labels: 训练集标签，标签值为0,1,2之类的整数，类型为ndarray，shape为(-1,)
        :return:None
        """

        unique_labels = list(set(train_labels))
        # 可使用np.where来简化
        for a, b in combinations(unique_labels, 2):
            train_a, label_a = [], []
            train_b, label_b = [], []
            for data, label in zip(train_data, train_labels):
                if label == a:
                    train_a.append(data)
                    label_a.append(a)
                elif label == b:
                    train_b.append(data)
                    label_b.append(b)

            train_a = np.array(train_a)
            train_b = np.array(train_b)
            train_batch_data = np.concatenate((train_a, train_b), axis=0)
            train_batch_label = np.concatenate((label_a, label_b))
            model = self.model()
            model.fit(train_batch_data, train_batch_label)
            self.models.append(model)

        return self

    def predict(self, test_datas):
        """
        OvO的预测阶段
        :param test_datas:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray
        """

        preds = np.array([model.predict(test_datas) for model in self.models]).T
        ans = []
        for line in preds:
            try:
                ans.append(mode(line))
            except StatisticsError:
                ans.append(choice(line))
        return np.array(ans)

