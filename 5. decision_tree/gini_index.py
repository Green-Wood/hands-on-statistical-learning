import numpy as np
from collections import Counter


def cal_gini(l):
    c = Counter(l)
    total = len(l)
    return 1 - sum([(val / total) ** 2 for val in c.values()])


def calc_total_gini(feature, label, index):
    """
    计算基尼系数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:基尼系数，类型float
    """

    # ********* Begin *********#
    feature_values = feature[:, index].flatten()
    c = Counter(feature_values)
    gini = 0
    total = len(feature_values)
    for val, count in c.items():
        gini += count / total * cal_gini(label[np.where(feature_values == val)])
    return gini
    # ********* End *********#
