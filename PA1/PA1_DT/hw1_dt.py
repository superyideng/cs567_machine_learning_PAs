import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        # num_cls is number of unique labels
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splittable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

        self.used_attr = []

    # split current node
    def split(self):
        # compute the entropy of this tree node
        if self.splittable:
            # find the entropy of current node
            node_entropy = 0
            for label in np.unique(self.labels):
                num_label = self.labels.count(label)
                prop = num_label / len(self.labels)
                node_entropy += -prop * np.log2(prop)

            # find which feature leads to largest information gain
            max_inf_gain = -1
            best_attr_size = -1
            uni_label_class = np.unique(self.labels)
            all_attr = [i for i in range(len(self.features[0]))]
            available_attr = [j for j in all_attr if j not in self.used_attr]
            for i in available_attr:
                cur_feature = np.array(self.features)[:, i]  # extract certain feature row
                cur_branches = []
                # find out # of attributes for current feature
                uni_cur_attr = np.unique(cur_feature)
                # for each attribute find the labels
                for attr in uni_cur_attr:
                    attr_index = np.where(cur_feature == attr)
                    attr_label = np.array(self.labels)[attr_index]
                    cur_row = []
                    for label in uni_label_class:
                        num = attr_label.tolist().count(label)
                        cur_row.append(num)
                    cur_branches.append(cur_row)
                cur_inf_gain = Util.Information_Gain(node_entropy, cur_branches)
                if cur_inf_gain > max_inf_gain or \
                        (cur_inf_gain == max_inf_gain and len(uni_cur_attr) > best_attr_size):
                    max_inf_gain = cur_inf_gain
                    self.dim_split = i
                    best_attr_size = len(uni_cur_attr)

            # extract the whole line of the best split feature we just computed
            split_feature = np.array(self.features)[:, self.dim_split]

            self.feature_uniq_split = np.unique(split_feature).tolist()

            # store the number of each attribute
            for attr in np.unique(split_feature):
                child_features = []
                child_labels = []
                value_indices_arr = np.array(self.features)
                value_indices = np.where(value_indices_arr[:, self.dim_split] == attr)[0]
                for index in value_indices:
                    child_features.append(self.features[index])
                    child_labels.append(self.labels[index])

                '''idx = np.where(split_feature == attr)
                child_features = np.delete(np.array(self.features)[idx], self.dim_split, axis=1).tolist()
                child_labels = np.array(self.labels)[idx].tolist()'''

                num_cls_child = np.unique(child_labels).size
                child = TreeNode(child_features, child_labels, num_cls_child)
                child.used_attr.extend(self.used_attr)
                child.used_attr.append(self.dim_split)
                if len(child.used_attr) == len(self.features[0]):
                    child.splittable = False
                self.children.append(child)

            for child in self.children:
                if child.splittable:
                    child.split()

        return

    # predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int

        if self.splittable and feature[self.dim_split] in self.feature_uniq_split:
            idx = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx].predict(feature)
        else:
            return self.cls_max
