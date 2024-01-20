import numpy as np
import matplotlib.pyplot as plt
from nearest_neighbour import predictknn, learnknn, gensmallm
def testSampleSize(sampleSize: int):
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], sampleSize)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    y_preds = predictknn(classifer, x_test)
    
    error = np.mean(y_test != y_preds)
    
    return error

def changeSampleSize():
    sampleSizes = range(10,101,10)
    errorBySampleSize = [testSampleSize(size) for size in sampleSizes for _ in range(10)]
    plt.plot(sampleSizes, errorBySampleSize, marker='o', linestyle='-', color='b', label='Error Values')

    plt.title('Error vs. Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)
    
changeSampleSize()
            
    