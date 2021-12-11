import numpy as np
from functools import reduce


class Tree:
    def __init__(self, data, children=None):
        self.data = data
        self.children = children


def get_gini(selections):
    total = np.sum(selections)
    return 1 - reduce(
        lambda acc, curr: acc + (curr / total) ** 2,
        selections,
        0)


def get_class_error(selections):
    return 1 - (np.max(selections) / np.sum(selections))


def get_entropy(selections):
    total = np.sum(selections)
    return 1 - reduce(
        lambda acc, curr: acc + (curr / total) * np.log2(curr / total),
        selections,
        0)


def get_purity_gain(data, method_string='gini'):
    """
    Calculate purity gain for tree

    :param method_string: 'gini' | 'class_error' | 'entropy'
    :param data: e.g Tree([32, 24],
                        [Tree([23, 8]),
                        Tree([9, 16])])
    :return: purity gain e.g. 0.125
    """
    method = {
        "gini": get_gini,
        "class_error": get_class_error,
        "entropy": get_class_error
    }[method_string]

    def calculate_purity(tree):
        total = sum(tree.data)
        if tree.children is None:
            return method(tree.data) * total

        return method(tree.data) - sum([calculate_purity(child) / total for child in tree.children])

    return calculate_purity(data)


# d = Tree([225,81], [Tree([108,62]), Tree([117,19])])

# print(get_purity_gain(d, 'gini'))

d = Tree([5,10,10], [Tree([0,8,2]), Tree([5,2,8])])
print(get_purity_gain(d, 'class_error'))
