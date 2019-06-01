# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_breast_cancer


def mds(data, d):
    """
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
    output:Z(ndarray):降维后的数据
    """
    # ********* Begin *********#
    # 计算dist矩阵
    m = data.shape[0]
    dist2 = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            dist2[i, j] = np.linalg.norm(data[i] - data[j]) ** 2
            dist2[j, i] = dist2[i, j]
    # 计算B
    B = np.zeros((m, m))
    dist2ij = np.mean(dist2)
    dist2i = np.mean(dist2, axis=1)
    dist2j = np.mean(dist2, axis=0)
    for i in range(m):
        for j in range(m):
            B[i, j] = -0.5 * (dist2[i, j] - dist2j[j] - dist2i[i] + dist2ij)

    # 矩阵分解得到特征值与特征向量
    value, vector = np.linalg.eigh(B)
    V = vector[:, : -d - 1: -1]
    A = value[: -d - 1: -1]
    # 计算Z
    Z = np.dot(V, np.sqrt(np.diag(A)))
    # ********* End *********#
    return Z


if __name__ == '__main__':
    data = load_breast_cancer().data
    d = 2
    print(mds(data, d))
