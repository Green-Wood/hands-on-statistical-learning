import numpy as np
from collections import Counter


def calInfo(l):
    c = Counter(l)
    total = len(l)
    ans = sum([-val / total * np.log2(val / total) for val in c.values()])
    return ans


def calcInfoGain(feature, label, index):
    '''
    计算信息增益
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益，类型float
    '''

    #*********** Begin ***********#
    total_info = calInfo(label)
    feature_list = feature[:, index].flatten()
    c = Counter(feature_list)
    length = len(feature_list)
    curr_info = 0
    for key, val in c.items():
        target_label = label[np.where(feature_list == key)]
        curr_info += val / length * calInfo(target_label)
    return total_info - curr_info


    #*********** End *************#

# feature = np.array(
#     [[0, 1], [1, 0], [1, 2], [0, 0], [1, 1]]
# )
label = np.array(
    [0, 1, 0, 0, 1]
)
# print(calcInfoGain(feature, label, 0))
print(np.where(label == 1))