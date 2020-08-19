import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

primeNumber = 1002527


def visualize_sample_sliced_by_velocity(samples, label):
    data = samples[:, 1:]
    merge_after = samples[:, 0].astype(int)
    # data0=data[label==0]
    data00 = data[np.logical_and(label == 0, merge_after == 0)]
    data01 = data[np.logical_and(label == 0, merge_after == 1)]
    data1 = data[label == 1]
    # ax.scatter(data0[:,1],data0[:,4],c='blue', marker='x', label='adv')
    for i in range(5):
        index00 = np.logical_and(data00[:, 0] > i * 5., data00[:, 0] <= (i + 1) * 5.)
        index01 = np.logical_and(data01[:, 0] > i * 5., data01[:, 0] <= (i + 1) * 5.)
        index1 = np.logical_and(data1[:, 0] > i * 5., data1[:, 0] <= (i + 1) * 5.)
        fig, ax = plt.subplots()
        ax.scatter(data00[index00, 1], data00[index00, 4], c='black', marker='x',
                   label='adv(merge infront)')
        ax.scatter(data01[index01, 1], data01[index01, 4], c='red', marker='x',
                   label='adv(merge after)')
        ax.scatter(data1[index1, 1], data1[index1, 4], c='blue', marker='o', label='coop')

        plt.ylabel('$\Delta x$')
        plt.xlabel('$\Delta v$')
        # plt.axis([-15, 15, -50, 100])
        plt.axis([-7, 5, -10, 50])
        ax.legend()
        plt.title('$v_{ego}\leq$' + str(i * 5 + 5))

    plt.show()


def visualize_sample(samples, label):
    dvInd = 1
    dxInd = 4
    fig, ax = plt.subplots()
    data = samples[:, 1:]
    merge_after = samples[:, 0].astype(int)
    # data0=data[label==0]
    data00 = data[np.logical_and(label == 0, merge_after == 0)]
    data01 = data[np.logical_and(label == 0, merge_after == 1)]
    data1 = data[label == 1]
    # ax.scatter(data0[:,1],data0[:,4],c='blue', marker='x', label='adv')
    ax.scatter(data00[:, dvInd], data00[:, dxInd], c='black', marker='x',
               label='adv(merge infront)')
    ax.scatter(data01[:, dvInd], data01[:, dxInd], c='red', marker='x',
               label='adv(merge after)')
    ax.scatter(data1[:, dvInd], data1[:, dxInd], c='blue', marker='o',
               label='coop')

    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    # plt.axis([-15, 15, -50, 100])
    # plt.axis([-10, 10, -10, 100])
    plt.axis([-7, 2, 0, 50])
    ax.legend()
    plt.show()


def visualize_prediction(x_test0, label, predict):
    # print('accuracy: ', np.mean(label==predict))
    dvInd = 1
    dxInd = 4
    data = x_test0[:, 1:]
    fig, ax = plt.subplots()

    data1 = data[np.logical_and(label == 1, predict == 1)]
    data0 = data[np.logical_and(label == 1, predict == 0)]
    ax.scatter(data1[:, dvInd], data1[:, dxInd], c='blue', marker='o', label='true coop')
    ax.scatter(data0[:, dvInd], data0[:, dxInd], c='blue', marker='x', label='false adv')

    merge_after = x_test0[:, 0].astype(int)
    data00 = data[np.logical_and(np.logical_and(label == 0, predict == 0),
                                 merge_after == 0)]
    data01 = data[np.logical_and(np.logical_and(label == 0, predict == 0),
                                 merge_after == 1)]
    data10 = data[np.logical_and(np.logical_and(label == 0, predict == 1),
                                 merge_after == 0)]
    data11 = data[np.logical_and(np.logical_and(label == 0, predict == 1),
                                 merge_after == 1)]
    ax.scatter(data00[:, dvInd], data00[:, dxInd], c='black', marker='o',
               label='true adv(merge infront)')
    ax.scatter(data01[:, dvInd], data01[:, dxInd], c='red', marker='o',
               label='true adv(merge after)')
    ax.scatter(data10[:, dvInd], data10[:, dxInd], c='black', marker='x',
               label='false coop(merge infront)')
    ax.scatter(data11[:, dvInd], data11[:, dxInd], c='red', marker='x',
               label='false coop(merge after)')

    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    # plt.axis([-15, 15, -50, 100])
    # plt.axis([-10, 5, -10, 50])
    plt.axis([-7, 2, 0, 50])
    ax.legend()
    plt.show()


def analysis(x_test0, y_predict, y_test):
    visualize_prediction(x_test0, y_test, y_predict)

    # for merge after
    label = y_test[x_test0[:, 0] == 1]
    predict = y_predict[x_test0[:, 0] == 1]
    accuracy = np.sum(label == predict) / label.shape[0]
    print('accuracy for merge after samples', accuracy)
    # for merge infront
    label = y_test[x_test0[:, 0] == 0]
    predict = y_predict[x_test0[:, 0] == 0]
    accuracy = np.sum(label == predict) / label.shape[0]
    print('accuracy for merge infront samples', accuracy)

    accuracy = np.sum(y_predict == y_test) / y_predict.shape[0]
    truePos = np.sum(y_test)
    trueNeg = y_test.shape[0] - np.sum(y_test)
    predictedPos = np.sum(y_predict)
    predictedNeg = y_predict.shape[0] - np.sum(y_predict)
    print('true positive ', truePos)
    print('true negative ', trueNeg)
    print('predicted positive ', predictedPos)
    print('predicted negative ', predictedNeg)
