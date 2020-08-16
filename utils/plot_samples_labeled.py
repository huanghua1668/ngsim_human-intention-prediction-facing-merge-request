import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def visualize_sample(f):
    data0=f['a']
    seqLen=10
    data0=data0[seqLen-1::seqLen] # get the last snapshot
    #data0=data0[::seqLen] # get the last snapshot
    fig, ax=plt.subplots()
    data=data0[:,3:-1]
    merge_after=data0[:,0].astype(int)
    label=data0[:,-1]
    data00=data[np.logical_and(label==0, merge_after==0)]
    data01=data[np.logical_and(label==0, merge_after==1)]
    data1=data[label==1]
    ax.scatter(data00[:,1],data00[:,4],c='black', marker='x', label='adv(merge infront)')
    ax.scatter(data01[:,1],data01[:,4],c='red',   marker='x', label='adv(merge after)')
    ax.scatter(data1[:,1], data1[:,4], c='blue',  marker='o', label='coop')

    plt.ylabel('$\Delta x \enspace [m]$', fontsize=25)
    plt.xlabel('$\Delta v \enspace [m/s^2]$', fontsize=25)
    plt.axis([-15, 15, -50, 100])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=25)

    #plt.show()

f=np.load('train_origin_us80.npz')
visualize_sample(f)
f=np.load('validate_origin_us101.npz')
visualize_sample(f)
plt.show()

