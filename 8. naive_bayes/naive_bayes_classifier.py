import numpy as np
from collections import Counter


class NaiveBayesClassifier(object):
    def __init__(self):
        """
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        """
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def _Laplace_smooth(self):
        pass

    def fit(self, feature, label):
        """
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: self
        """
        m, n = feature.shape
        label_count = Counter(label)
        for key, val in label_count.items():
            # 拉普拉斯平滑
            self.label_prob[key] = (val + 1) / (m + len(label_count.keys()))

        for label_key in label_count.keys():
            # condition
            self.condition_prob[label_key] = dict()
            for j in range(n):
                feature_unique = np.unique(feature[:, j])
                feature_condition = feature[label == label_key, j]
                feature_count = Counter(feature_condition)
                self.condition_prob[label_key][j] = dict()
                for feature_key in feature_unique:
                    # 拉普拉斯平滑
                    try:
                        rate = (feature_count[feature_key] + 1) / (len(feature_condition) + len(feature_unique))
                    except KeyError:
                        rate = 1 / len(feature_unique)
                    self.condition_prob[label_key][j][feature_key] = rate
        return self

    def _predict_proba(self, label_key, single_feature):
        r = self.label_prob[label_key]
        for i, feature_val in enumerate(single_feature):
            r *= self.condition_prob[label_key][i][feature_val]
        return r

    def _predict(self, single_feature):
        return max(self.label_prob.keys(), key=lambda x: self._predict_proba(x, single_feature))

    def predict(self, feature):
        """
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        """
        return [self._predict(x) for x in feature]


if __name__ == '__main__':
    feature = np.array(
        [[2, 1, 1],
         [1, 2, 2],
         [2, 2, 2],
         [2, 1, 2],
         [1, 2, 3],
         [2, 1, 3],
         [1, 1, 3],
         [1, 2, 1],
         [2, 2, 1]]
    )
    label = np.array([1, 0, 1, 1, 1, 0, 1, 1, 1])
    clf = NaiveBayesClassifier()
    clf.fit(feature, label)
    test_x = [
        [1, 1, 3],
    ]
    print(clf.predict(test_x))
