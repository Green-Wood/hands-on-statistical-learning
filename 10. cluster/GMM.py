import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris


class GMM(object):
    def __init__(self, n_components, max_iter=100):
        """
        构造函数
        :param n_components: 想要划分成几个簇，类型为int
        :param max_iter: EM的最大迭代次数
        """
        self.n_components = n_components
        self.max_iter = max_iter

    def _cal_prob(self, l, data):
        mu = self.mu[l]
        sigma = self.sigma[l]
        return multivariate_normal(mean=mu, cov=sigma).pdf(data)

    def fit(self, train_data):
        """
        训练，将模型参数分别保存至self.alpha，self.mu，self.sigma中
        :param train_data: 训练数据集，类型为ndarray
        :return: self
        """
        row, col = train_data.shape
        # 初始化每个高斯分布的响应系数
        self.alpha = np.array([1.0 / self.n_components] * self.n_components)
        # 初始化每个高斯分布的均值向量
        self.mu = np.random.rand(self.n_components, col)
        # 初始化每个高斯分布的协方差矩阵
        self.sigma = np.array([np.eye(col)] * self.n_components)

        prob = np.zeros((row, self.n_components))

        iter = 0
        while iter < self.max_iter:
            # Expectation step
            for j in range(row):
                lower = sum(self.alpha[l] * self._cal_prob(l, train_data[j]) for l in range(self.n_components))
                for i in range(self.n_components):
                    prob[j, i] = self.alpha[i] * self._cal_prob(i, train_data[j]) / lower

            # maximization step
            for i in range(self.n_components):
                self.mu[i] = sum(prob[j, i] * train_data[j] for j in range(row)) / sum(prob[:, i])

                new_sigma = np.zeros((col, col))
                for j in range(row):
                    # 一维向量点乘会得到数，要先转化为矩阵才能dot出矩阵
                    vec = (train_data[j] - self.mu[i]).reshape((1, -1))
                    vec_trans = vec.reshape((-1, 1))
                    new_sigma += prob[j, i] * np.dot(vec_trans, vec)
                new_sigma /= sum(prob[:, i])
                self.sigma[i] = new_sigma

                self.alpha[i] = np.mean(prob[:, i])

            iter += 1

        return self

    def predict(self, test_data):
        """
        预测，根据训练好的模型参数将test_data进行划分。
        注意：划分的标签的取值范围为[0,self.n_components-1]，即若self.n_components为3，则划分的标签的可能取值为0,1,2。
        :param test_data: 测试集数据，类型为ndarray
        :return: 划分结果，类型为list
        """

        ans = []
        for test_x in test_data:
            predict_label = np.argmax([self.alpha[i] * self._cal_prob(i, test_x) for i in range(self.n_components)])
            ans.append(predict_label)
        return ans


if __name__ == '__main__':
    data = load_iris()
    train_x = data.data[:100]
    train_y = data.target[:100]
    gmm = GMM(n_components=2)
    gmm.fit(train_x)
    print(gmm.predict(train_x))
    print(train_y)
