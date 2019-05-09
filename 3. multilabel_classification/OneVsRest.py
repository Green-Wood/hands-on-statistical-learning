import numpy as np


class OvR(object):
    def __init__(self, model):
        # 用于保存训练时各种模型的list
        self.model = model
        self.models = []
        # 用于保存models中对应的正例的真实标签
        # 例如第1个模型的正例是2，则real_label[0]=2
        self.real_label = []

    def fit(self, train_datas, train_labels):
        """
        OvR的训练阶段，将模型保存到self.models中
        :param train_datas: 训练集数据，类型为ndarray
        :param train_labels: 训练集标签，标签值为0,1,2之类的整数，类型为ndarray，shape为(-1,)
        :return:None
        """

        unique_label = list(set(train_labels))
        # 可使用np.where来简化
        self.real_label = unique_label.copy()
        for l in self.real_label:
            p_train, p_label = [], []
            n_train, n_label = [], []
            for data, label in zip(train_datas, train_labels):
                if label == l:
                    p_train.append(data)
                    p_label.append(1)
                else:
                    n_train.append(data)
                    n_label.append(0)
            train_batch_data = np.concatenate((p_train, n_train), axis=0)
            train_batch_label = np.concatenate((p_label, n_label))
            model = self.model()
            model.fit(train_batch_data, train_batch_label)
            self.models.append(model)
        return self

    def predict(self, test_datas):
        """
        OvR的预测阶段
        :param test_datas:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray
        """

        probs = np.array([model.predict_proba(test_datas) for model in self.models]).T
        ans = [self.real_label[np.argmax(line)] for line in probs]
        return np.array(ans)
