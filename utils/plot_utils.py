import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_loss_functions():
    def logistic(x):
        return -np.log(1./(1.+np.exp(-x)))

    def svm(x):
        y=1.-x
        y[y<0]=0.
        return y

    def lorenz(x):
        y=np.log(1.+(x-1.)*(x-1.))
        y[x>1.]=0.
        return y

    plt.figure()
    x=np.arange(-5,4)
    plt.plot(x,logistic(x), label='logistic loss')
    plt.plot(x,svm(x), label='hinge loss')
    plt.plot(x,lorenz(x), label='lorenz')
    plt.legend()
    plt.axis([-5, 3, -0.5, 6])

    plt.show()

def plot_feature_selections():
    validation_acc = [0.846, 0.846, 0.8484, 0.846, 0.8434, 0.8434, 0.8409, 0.8333, 0.8080]
    features = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    plt.figure()
    plt.plot(features, validation_acc, '-o')
    plt.axis([10, 0, 0.8, 0.85])
    plt.ylabel('validation acc')
    plt.xlabel('features')
    plt.show()

plot_feature_selections()