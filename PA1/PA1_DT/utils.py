import numpy as np
import hw1_dt as dt


# Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    sum_arr = np.array([])
    for branch in branches:
        sum_arr = np.append(sum_arr, np.sum(branch))

    scaler_arr = sum_arr / np.sum(sum_arr)

    prob_arr = []
    for k in range(len(branches)):
        prob_arr.append(branches[k] / sum_arr[k])

    # prob_arr = branches / sum_arr

    # replace all 0 in probability array by 1 in order to make the arr logable
    # without changing the entropy.
    for i in range(len(prob_arr)):
        for j in range(len(prob_arr[0])):
            if prob_arr[i][j] == 0:
                prob_arr[i][j] = 1

    prob_arr = np.array(prob_arr)
    log_arr = np.log2(prob_arr)

    mul_arr = -prob_arr * log_arr

    entropy_arr = np.array([])
    for mul in mul_arr:
        entropy_arr = np.append(entropy_arr, np.sum(mul))

    inf_gain = S - np.sum(entropy_arr * scaler_arr)
    return inf_gain


# TODO: implement reduced error pruning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List

    # compute current accuracy
    predicted = decisionTree.predict(X_test)
    correct_num = 0
    for i in range(len(y_test)):
        if predicted[i] == y_test[i]:
            correct_num += 1

    orig_accuracy = correct_num / len(y_test)
    max_accuracy = orig_accuracy

    root = decisionTree.root_node
    nodes = [root]
    possible_nodes = []
    prune_node = None

    # DFS on original tree
    while nodes:
        cur_node = nodes[0]
        if cur_node.children:
            possible_nodes.append(cur_node)
        nodes = nodes[1:]
        for child in cur_node.children:
            nodes.insert(0, child)

    # find the best improved error rate
    for node in possible_nodes:
        node.splittable = False
        new_predicted = decisionTree.predict(X_test)
        new_correct_num = 0
        for i in range(len(y_test)):
            if new_predicted[i] == y_test[i]:
                new_correct_num += 1
        cur_accuracy = new_correct_num / len(y_test)

        if cur_accuracy > max_accuracy:
            max_accuracy = cur_accuracy
            prune_node = node
        # Undo current pruning
        node.splittable = True

    '''if max_accuracy > orig_accuracy:
        prune_node.children = []
        prune_node.splittable = False
        reduced_error_prunning(decisionTree, X_test, y_test)'''

    if max_accuracy >= orig_accuracy:
        prune_node.children = []
        prune_node.splittable = False
        reduced_error_prunning(decisionTree, X_test, y_test)

    return


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
