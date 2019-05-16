# encoding=utf8
import numpy as np
from numpy.linalg import inv


def lda(X, y):
    """
    input:X(ndarray):待处理数据 (n, 2)
          y(ndarray):待处理数据标签，标签分别为0和1 (1, n)
    output:X_new(ndarray):处理后的数据
    """
    # ********* Begin *********#
    p_data = X[np.where(y == 1)]
    n_data = X[np.where(y == 0)]

    p_data = p_data.T
    n_data = n_data.T

    p_cov = np.cov(p_data)
    n_cov = np.cov(n_data)
    S_w = p_cov + n_cov

    p_mu = np.mean(p_data, axis=1)
    n_mu = np.mean(n_data, axis=1)
    w = inv(S_w).dot(n_mu - p_mu)
    X_new = X.dot(w).reshape(-1, 1)
    return X_new


