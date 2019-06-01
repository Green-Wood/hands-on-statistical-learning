import numpy as np


def calc_min_dist(cluster1, cluster2):
    """
    计算簇间最小距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最小距离
    """

    # ********* Begin *********#
    return min([np.linalg.norm(np.array(i1) - np.array(i2)) for i1 in cluster1 for i2 in cluster2])
    # ********* End *********#


def find_min_cluster(data):
    m = len(data)
    min_dist = calc_min_dist(data[0], data[1])
    min_i, min_j = 0, 1
    for i in range(m):
        for j in range(i + 1, m):
            curr_dist = calc_min_dist(data[i], data[j])
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_i = i
                min_j = j
    return min_i, min_j


def AGNES(feature, k):
    """
    AGNES聚类并返回聚类结果，量化距离时请使用簇间最大欧氏距离
    假设数据集为`[1, 2], [10, 11], [1, 3]]，那么聚类结果可能为`[[1, 2], [1, 3]], [[10, 11]]]
    :param feature:数据集，类型为ndarray
    :param k:表示想要将数据聚成`k`类，类型为`int`
    :return:聚类结果，类型为list
    """

    # ********* Begin *********#
    all_class = []
    for f in feature:
        Ci = []
        Ci.append(f)
        all_class.append(Ci)
    while len(all_class) > k:
        i, j = find_min_cluster(all_class)
        for f in all_class[j]:
            all_class[i].append(f)
        all_class.pop(j)
    return all_class
    # ********* End *********#
