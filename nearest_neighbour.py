import heapq
from collections import Counter
import numpy as np
from scipy.spatial import distance


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return k, list(zip(x_train, y_train))

def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k = classifier[0]
    y_pred = []
    test_data = classifier[1]
    for x in x_test:
        h = []
        for s in test_data:
            d = distance.euclidean(x, s[0])
            if len(h) < k:
                heapq.heappush(h, (1 / d, s[1]))
            elif (1 / d) > h[0][0]:
                heapq.heappop(h)
                heapq.heappush(h, (1 / d, s[1]))
        most_common_label = Counter(y[1] for y in h).most_common(1)[0][0]
        y_pred.append(most_common_label)
    y_pred = np.array(y_pred).reshape(-1, 1)
    return y_pred


def simple_test():
    """test the functions with a simple example"""
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 500)

    classifer = learnknn(1, x_train, y_train)

    count = 0
    preds = predictknn(classifer, x_test)
    for y_p, y_r in zip(preds, y_test):
        print(f"expected: {y_r}, received: {y_p[0]}")
        if y_p[0] != y_r:
            count += 1
    print(preds.reshape(-1, 1) - y_test)
    print(count)
    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
