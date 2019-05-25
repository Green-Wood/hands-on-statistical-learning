# encoding=utf8
import numpy as np
from random import sample
from sklearn.datasets import load_wine


class Kmeans():
    """Kmeans聚类算法.
    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数.
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon,
        则说明算法已经收敛
    """

    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        np.random.seed(1)

    # ********* Begin *********#
    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        index = sample(range(X.shape[0]), self.k)
        return X[index]

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        dist_list = [np.linalg.norm(sample - c) for c in centroids]
        return np.argmin(dist_list)

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        cluster = [set() for i in range(self.k)]
        for i, sample in enumerate(X):
            k = self._closest_centroid(sample, centroids)
            cluster[k].add(i)
        return cluster

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        centroids = []
        for i, c in enumerate(clusters):
            index = np.array(list(c))
            centroids.append(np.mean(X[index], axis=0))
        return np.array(centroids)

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        cluster_label = np.zeros(X.shape[0])
        for i, c in enumerate(clusters):
            index = np.array(list(c))
            cluster_label[index] = i
        return cluster_label

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        new_centroids = self.init_random_centroids(X)
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        while True:
            centroids = new_centroids
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            cluster = self.create_clusters(centroids, X)
            # 计算新的聚类中心
            new_centroids = self.update_centroids(cluster, X)
            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            if all(np.linalg.norm(old - new) for old, new in zip(centroids, new_centroids)) < self.varepsilon:
                break
        return self.get_cluster_labels(cluster, X)


if __name__ == '__main__':
    data = load_wine()
    x = data.data
    k_means = Kmeans()
    label = k_means.predict(x)
    print(label)
