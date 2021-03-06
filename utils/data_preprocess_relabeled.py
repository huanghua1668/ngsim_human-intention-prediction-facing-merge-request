import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

primeNumber=1002527
seqLen=10

def preprocess():
    # the data, split between train and test sets
    seqLen=10
    datas=[]
    datas.append(np.genfromtxt('0400pm-0415pm/samples_snapshots_relabeled.csv', delimiter=','))
    datas.append(np.genfromtxt('0500pm-0515pm/samples_snapshots_relabeled.csv', delimiter=','))
    datas.append(np.genfromtxt('0515pm-0530pm/samples_snapshots_relabeled.csv', delimiter=','))
    #datas.append(np.genfromtxt('0400pm-0415pm/samples_merge_after_snapshots.csv', delimiter=','))
    #datas.append(np.genfromtxt('0500pm-0515pm/samples_merge_after_snapshots.csv', delimiter=','))
    #datas.append(np.genfromtxt('0515pm-0530pm/samples_merge_after_snapshots.csv', delimiter=','))

    for i in range(3):
        #0 for merge in front
        temp=np.zeros((datas[i].shape[0], datas[i].shape[1]+1))
        temp[:,1:]=datas[i]
        datas[i]=temp
    #for i in range(3,6):
    #    #1 for merge after
    #    temp=np.ones((datas[i].shape[0], datas[i].shape[1]+1))
    #    temp[:,1:]=datas[i]
    #    datas[i]=temp

    data=np.vstack((datas[0], datas[1]))
    for i in range(2,3):
        data=np.vstack((data, datas[i]))
    ys=data[:,-1]
    print(ys.shape[0]/seqLen, ' samples, and ', np.sum(ys)/ys.shape[0]*100, '% positives')

    np.random.seed(primeNumber)

    #data=data[:100]
    #print('data', data[:,:4])

    indexes=np.arange(data.shape[0]/seqLen).astype(int)
    nCoop=data[:,-1].sum()/seqLen
    nTotal=data.shape[0]/seqLen
    nAdv=nTotal-nCoop
    print('total samples', data.shape[0], 'total sequences', indexes.shape[0])
    print('coop samples', nCoop)
    print('adv samples', nAdv)
    print('before shuffle, indexes[0:10]', indexes[0:10] )
    np.random.shuffle(indexes)
    print('after shuffle, indexes[0:10]', indexes[0:10] )

    data_train=[]
    for i in range(0, indexes.shape[0]-3, 5):
        for j in range(i, i+3):
            for k in range(seqLen):
                data_train.append(data[indexes[j]*seqLen+k])
    sc_X = StandardScaler()
    data_train=np.vstack(data_train)
    np.savez("train_origin_relabeled.npz",a=data_train)
    #print('data train', data_train[:,:4])
    x_train=data_train[:, 3:-1]
    y_train=data_train[:, -1].astype(int)
    #visualize_sample(data_train[:, 0:-1] , y_train)
    #visualize_sample_sliced_by_velocity(data_train[:,0:-1], y_train)
    x_train=sc_X.fit_transform(x_train)
    np.savez("train_normalized_relabeled.npz",a=x_train, b=y_train)

    data_validate=[]
    for i in range(3, indexes.shape[0], 5):
        for j in range(seqLen):
            data_validate.append(data[indexes[i]*seqLen+j])
    data_validate=np.vstack(data_validate)
    np.savez("validate_origin_relabeled.npz",a=data_validate)
    #print('data validate', data_validate[:,:4])
    x_validate=data_validate[:,3:-1]
    x_validate=sc_X.transform(x_validate)
    y_validate=data_validate[:,-1].astype(int)
    np.savez("validate_normalized_relabeled.npz",a=x_validate, b=y_validate)

    data_test=[]
    for i in range(4, indexes.shape[0], 5):
        for j in range(seqLen):
            data_test.append(data[indexes[i]*seqLen+j])
    data_test=np.vstack(data_test)
    #print('data test', data_test[:,:4])
    np.savez("test_origin_relabeled.npz",a=data_test)
    x_test=data_test[:,3:-1]
    x_test=sc_X.transform(x_test)
    y_test=data_test[:,-1].astype(int)
    np.savez("test_normalized_relabeled.npz",a=x_test, b=y_test)

    print(y_train.shape[0], ' trainning samples, and ', 
          np.sum(y_train)/y_train.shape[0]*100, '% positives')
    print(y_validate.shape[0], ' validate samples, and ', 
          np.sum(y_validate)/y_validate.shape[0]*100, '% positives')
    print(y_test.shape[0], ' test samples, and ', 
          np.sum(y_test)/y_test.shape[0]*100, '% positives')
    return x_train, x_validate, x_test, y_train, y_validate, y_test, sc_X

def preprocess_both_dataset():
    # the data, split between train and test sets
    seqLen=10
    datas=[]
    datas.append(np.genfromtxt('0400pm-0415pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0500pm-0515pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0515pm-0530pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0400pm-0415pm/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0500pm-0515pm/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0515pm-0530pm/samples_merge_after_snapshots.csv', delimiter=','))

    for i in range(3):
        #0 for merge in front
        temp=np.zeros((datas[i].shape[0], datas[i].shape[1]+1))
        temp[:,1:]=datas[i]
        datas[i]=temp
    for i in range(3,6):
        #1 for merge after
        temp=np.ones((datas[i].shape[0], datas[i].shape[1]+1))
        temp[:,1:]=datas[i]
        datas[i]=temp

    data=np.vstack((datas[0], datas[1]))
    for i in range(2,6):
        data=np.vstack((data, datas[i]))
    ys=data[:,-1]
    print('For dataset us-80:')
    print(ys.shape[0]/seqLen, ' samples, and ', np.sum(ys)/ys.shape[0]*100, '% positives')

    np.random.seed(primeNumber)

    #data=data[:100]
    #print('data', data[:,:4])

    indexes=np.arange(data.shape[0]/seqLen).astype(int)
    nCoop=data[:,-1].sum()/seqLen
    nAfter=data[:,0].sum()/seqLen
    nTotal=data.shape[0]/seqLen
    print('total samples', data.shape[0], 'total sequences', indexes.shape[0])
    print('coop samples', nCoop)
    print('merge after samples', nAfter )
    print('adv samples of merge in front', nTotal-nAfter-nCoop)
    print('before shuffle, indexes[0:10]', indexes[0:10] )
    np.random.shuffle(indexes)
    print('after shuffle, indexes[0:10]', indexes[0:10] )

    data_train=[]
    for i in range(indexes.shape[0]):
        for k in range(seqLen):
            data_train.append(data[indexes[i]*seqLen+k])
    sc_X = StandardScaler()
    data_train=np.vstack(data_train)
    np.savez("train_origin_us80.npz",a=data_train)
    #print('data train', data_train[:,:4])
    x_train=data_train[:, 3:-1]
    y_train=data_train[:, -1].astype(int)
    #visualize_sample(data_train[:, 0:-1] , y_train)
    #visualize_sample_sliced_by_velocity(data_train[:,0:-1], y_train)
    x_train=sc_X.fit_transform(x_train)
    np.savez("train_normalized_us80.npz",a=x_train, b=y_train)

    datas=[]
    dir0='/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
    datas.append(np.genfromtxt(dir0+'0750am-0805am/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0+'0805am-0820am/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0+'0820am-0835am/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0+'0750am-0805am/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0+'0805am-0820am/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0+'0820am-0835am/samples_merge_after_snapshots.csv', delimiter=','))

    for i in range(3):
        #0 for merge in front
        temp=np.zeros((datas[i].shape[0], datas[i].shape[1]+1))
        temp[:,1:]=datas[i]
        datas[i]=temp
    for i in range(3,6):
        #1 for merge after
        temp=np.ones((datas[i].shape[0], datas[i].shape[1]+1))
        temp[:,1:]=datas[i]
        datas[i]=temp

    data=np.vstack((datas[0], datas[1]))
    for i in range(2,6):
        data=np.vstack((data, datas[i]))
    ys=data[:,-1]
    print('For dataset us-101:')
    print(ys.shape[0]/seqLen, ' samples, and ', np.sum(ys)/ys.shape[0]*100, '% positives')

    nCoop=data[:,-1].sum()/seqLen
    nAfter=data[:,0].sum()/seqLen
    nTotal=data.shape[0]/seqLen
    print('total samples', data.shape[0], 'total sequences', indexes.shape[0])
    print('coop samples', nCoop)
    print('merge after samples', nAfter )
    print('adv samples of merge in front', nTotal-nAfter-nCoop)

    np.savez("validate_origin_us101.npz",a=data)
    #print('data train', data_train[:,:4])
    x_validate=data[:,3:-1]
    x_validate=sc_X.transform(x_validate)
    y_validate=data[:,-1].astype(int)
    np.savez("validate_normalized_us101.npz",a=x_validate, b=y_validate)

    return x_train, x_validate, y_train, y_validate, sc_X

def preprocess_multiple():
    # the data, split between train and test sets
    data0=np.genfromtxt('0400pm-0415pm/samples_multiple_snapshot.csv', delimiter=',')
    data1=np.genfromtxt('0500pm-0515pm/samples_multiple_snapshot.csv', delimiter=',')
    data2=np.genfromtxt('0515pm-0530pm/samples_multiple_snapshot.csv', delimiter=',')
    data=np.vstack((data0, data1))
    data=np.vstack((data, data2))
    ys=data[:,-1]
    print(ys.shape[0], ' samples, and ', np.sum(ys)/ys.shape[0]*100, '% positives')
    
    data_train=[]
    data_validate=[]
    data_test=[]
    mixture=False
    for i in range(data.shape[0]):
        if not mixture:
            laneChange=int(data[i,0])
            if laneChange%5<3:
                data_train.append(data[i,1:])
            elif laneChange%5==3:
                data_validate.append(data[i,1:])
            else:
                data_test.append(data[i,1:])
        else:
            if i%5<3:
                data_train.append(data[i,1:])
            elif i%5==3:
                data_validate.append(data[i,1:])
            else:
                data_test.append(data[i,1:])
    sc_X = StandardScaler()
    data_train=np.vstack(data_train)
    x_train=data_train[:, :-1]
    x_train=sc_X.fit_transform(x_train)
    y_train=data_train[:, -1].astype(int)

    data_validate=np.vstack(data_validate)
    x_validate=data_validate[:,:-1]
    x_validate=sc_X.transform(x_validate)
    y_validate=data_validate[:,-1].astype(int)

    data_test=np.vstack(data_test)
    x_test=data_test[:,:-1]
    x_test=sc_X.transform(x_test)
    y_test=data_test[:,-1].astype(int)
    print(y_train.shape[0], ' trainning samples, and ', 
          np.sum(y_train)/y_train.shape[0]*100, '% positives')
    print(y_validate.shape[0], ' validate samples, and ', 
          np.sum(y_validate)/y_validate.shape[0]*100, '% positives')
    print(y_test.shape[0], ' test samples, and ', 
          np.sum(y_test)/y_test.shape[0]*100, '% positives')
    return x_train, x_validate, x_test, y_train, y_validate, y_test

def visualize_sample(f):
    #preprocess()
    data0=f['a']
    data0=data0[seqLen-1::seqLen] # get the last snapshot
    #data0=data0[::seqLen] # get the first snapshot
    fig, ax=plt.subplots()
    data=data0[:,3:-1]
    merge_after=data0[:,0].astype(int)
    label=data0[:,-1]
    #data0=data[label==0]
    data00=data[np.logical_and(label==0, merge_after==0)]
    data1=data[label==1]
    #ax.scatter(data0[:,1],data0[:,4],c='blue', marker='x', label='adv')
    ax.scatter(data00[:,1],data00[:,4],c='black', marker='x', label='adv')
    ax.scatter(data1[:,1], data1[:,4], c='blue',  marker='o', label='coop')

    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-15, 15, -50, 100])
    #plt.axis([-7, 2, 0, 50])
    ax.legend()
    plt.show()

preprocess()
f=np.load('train_origin_relabeled.npz')
visualize_sample(f)
#preprocess_both_dataset()
#f=np.load('train_origin_us80.npz')
#visualize_sample(f)
#f=np.load('validate_origin_us101.npz')
#visualize_sample(f)
