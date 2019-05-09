import numpy as np
from copy import deepcopy
from collections import Counter


class DecisionTree(object):
    """
    具备后剪枝能力的决策树
    """
    def __init__(self):
        # 决策树模型
        self.tree = {}

    def calcInfoGain(self, feature, label, index):
        """
        计算信息增益
        :param feature:测试用例中字典里的feature，类型为ndarray
        :param label:测试用例中字典里的label，类型为ndarray
        :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        :return:信息增益，类型float
        """

        # 计算熵
        def calcInfoEntropy(feature, label):
            """
            计算信息熵
            :param feature:数据集中的特征，类型为ndarray
            :param label:数据集中的标签，类型为ndarray
            :return:信息熵，类型float
            """

            label_set = set(label)
            result = 0
            for l in label_set:
                count = 0
                for j in range(len(label)):
                    if label[j] == l:
                        count += 1
                # 计算标签在数据集中出现的概率
                p = count / len(label)
                # 计算熵
                result -= p * np.log2(p)
            return result

        # 计算条件熵
        def calcHDA(feature, label, index, value):
            '''
            计算信息熵
            :param feature:数据集中的特征，类型为ndarray
            :param label:数据集中的标签，类型为ndarray
            :param index:需要使用的特征列索引，类型为int
            :param value:index所表示的特征列中需要考察的特征值，类型为int
            :return:信息熵，类型float
            '''
            count = 0
            # sub_feature和sub_label表示根据特征列和特征值分割出的子数据集中的特征和标签
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][index] == value:
                    count += 1
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            pHA = count / len(feature)
            e = calcInfoEntropy(sub_feature, sub_label)
            return pHA * e

        base_e = calcInfoEntropy(feature, label)
        f = np.array(feature)
        # 得到指定特征列的值的集合
        f_set = set(f[:, index])
        sum_HDA = 0
        # 计算条件熵
        for value in f_set:
            sum_HDA += calcHDA(feature, label, index, value)
        # 计算信息增益
        return base_e - sum_HDA

    def fit(self, train_feature, train_label, val_feature, val_label):
        """
        :param train_feature:训练集数据，类型为ndarray
        :param train_label:训练集标签，类型为ndarray
        :param val_feature:验证集数据，类型为ndarray
        :param val_label:验证集标签，类型为ndarray
        :return: None
        """

        # ************* Begin ************#
        self.tree = self.build_tree(train_feature, train_label, range(train_feature.shape[1]))
        self.prune(self.tree, [], val_feature, val_label)
        return self
        # ************* End **************#

    def build_tree(self, feature, label, valid_index):
        """
        递归创建一个决策树
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :param valid_index: 哪些属性是可用的，类型为list
        :return:
        """
        def get_mode_label(label):
            """
            返回出现次数最多的标签
            :param label: 标签值
            :return:
            """
            c = Counter(label)
            return sorted(c.items(), key=lambda x: x[1], reverse=True)[0][0]

        def is_all_equal(feature):
            line = feature[0]
            for l in feature:
                if not np.array_equal(line, l):
                    return False
            return True

        if len(set(label)) == 1:
            return label[0]

        # 样本中只剩下一个属性或者所有的属性数值都相同
        if len(valid_index) == 0 or is_all_equal(feature):
            return get_mode_label(label)

        best_feature_index = valid_index[np.argmax([self.calcInfoGain(feature, label, i) for i in valid_index])]
        tree = {
            best_feature_index: {},
            'mode_label': get_mode_label(label)
        }
        # 删除作为根结点的属性
        sub_valid_index = np.delete(valid_index, np.where(valid_index == best_feature_index), axis=0)
        best_feature_val = feature[:, best_feature_index].flatten()
        for val in set(best_feature_val):
            # 划分子属性和子标签
            sub_feature = feature[np.where(best_feature_val == val)]
            sub_label = label[np.where(best_feature_val == val)]
            tree[best_feature_index][val] = self.build_tree(sub_feature, sub_label, sub_valid_index)
        return tree

    def prune(self, tree, path, val_feature, val_label):
        """
        对self.tree进行剪枝
        :param tree: 当前结点
        :param path: 当前的路径
        :param val_feature: 验证集属性
        :param val_label: 验证集标签
        :return: null
        """

        def accuracy(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            return np.mean(y_true == y_pred)

        def test_tree_accuracy(tree, val_feature, val_label):
            y_pred = np.array([self.predict_recursive(single_feature, tree) for single_feature in val_feature])
            return accuracy(val_label, y_pred)

        if type(tree) is not dict:
            return

        feature_list = list(tree.keys())
        feature_list.remove('mode_label')
        node_feature = feature_list[0]    # 获取该结点的判断属性
        path.append(node_feature)
        val_dict = tree[node_feature]       # 获取属性值和子树的字典

        for line_val, subtree in val_dict.items():
            path.append(line_val)
            self.prune(subtree, deepcopy(path), val_feature, val_label)
            path.pop()

        # 判断该结点是否应该进行剪枝
        # 获取备份的剪枝树
        tree_pruned = deepcopy(self.tree)
        tree_pruned_root = tree_pruned
        if len(path) > 2:
            val_to_curr_node = path[-2]
            for step in path[:-2]:
                tree_pruned = tree_pruned[step]
            tree_pruned[val_to_curr_node] = tree['mode_label']
        else:
            tree_pruned_root = tree['mode_label']

        acc_before_prune = test_tree_accuracy(self.tree, val_feature, val_label)
        acc_after_prune = test_tree_accuracy(tree_pruned_root, val_feature, val_label)
        # 判断是否应该对原树进行剪枝
        if acc_before_prune < acc_after_prune:
            self.tree = tree_pruned_root
        path.pop()

    def predict(self, feature):
        """
        :param feature:测试集数据，类型为ndarray
        :return:预测结果，如np.array([0, 1, 2, 2, 1, 0])
        """

        # ************* Begin ************#
        ans = np.array([self.predict_recursive(single_feature, self.tree) for single_feature in feature])
        return ans
        # ************* End **************#

    def predict_recursive(self, feature, tree):
        """
        :param feature: 一维数组
        :param tree: 字典或者数值
        :return: 一个测试样例的结果
        """
        if type(tree) is not dict:
            return tree

        if type(tree) is not dict:
            return tree

        keys = list(tree.keys())
        keys.remove('mode_label')
        test_feature = keys[0]
        value = feature[test_feature]
        return self.predict_recursive(feature, tree[test_feature][value])


if __name__ == '__main__':
    feature = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])
    label = np.array([
        1, 0, 0, 1, 1, 0, 1, 0
    ])

    val_feature = np.array([
        [0, 1, 0]
    ])
    val_label = np.array([
        1
    ])

    test_feature = np.array([
        [1, 1, 1]
    ])
    tree = DecisionTree()
    tree.fit(feature, label, val_feature, val_label)
    print(tree.predict(test_feature))