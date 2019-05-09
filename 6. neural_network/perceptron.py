import numpy as np


# 构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        """
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        """
        def f_dataset(data, label):
            pred = self.predict(data)
            f_data = data[np.where(pred != label)]
            f_label = label[np.where(pred != label)]
            return f_data, f_label

        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.] * data.shape[1])
        self.b = np.array([1.])

        for i in range(self.max_iter):
            f_data, f_label = f_dataset(data, label)
            if len(f_data) == 0:
                break
            x_i, y_i = f_data[0], f_label[0]
            self.w = self.w + self.lr * y_i * x_i
            self.b = self.b + self.lr * y_i

    def predict(self, data):
        """
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        """
        # ********* Begin *********#
        predict = data.dot(self.w) + self.b
        predict = np.where(predict > 0, 1, -1)
        # ********* End *********#
        return predict


if __name__ == '__main__':
    data = np.array([
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4]
    ])
    label = np.array([
        1, 1, 1, 1, -1, -1, -1, -1
    ])
    test_data = np.array([
        [100, 2],
        [2.5, 0],
        [0, 1.3]
    ])

    clf = Perceptron()
    clf.fit(data, label)
    print(clf.predict(test_data))