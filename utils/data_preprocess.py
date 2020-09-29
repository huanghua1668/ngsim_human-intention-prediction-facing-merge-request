import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

primeNumber = 3
seqLen = 10


def preprocess():
    # the data, split between train and test sets
    datas = []
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    datas.append(np.genfromtxt(dir+'0400pm-0415pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir+'0500pm-0515pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir+'0515pm-0530pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir+'0400pm-0415pm/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir+'0500pm-0515pm/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir+'0515pm-0530pm/samples_merge_after_snapshots.csv', delimiter=','))

    for i in range(3):
        # 0 for merge in front
        temp = np.zeros_like(datas[i])
        temp[:, 1:] = datas[i][:, 1:]
        datas[i] = temp
    for i in range(3, 6):
        # 1 for merge after
        temp = np.ones_like(datas[i])
        temp[:, 1:] = datas[i][:, 1:]
        datas[i] = temp

    data = np.vstack((datas[0], datas[1]))
    for i in range(2, 6):
        data = np.vstack((data, datas[i]))
    # data = data[:, 1:]  # delete index of lane changes
    ys = data[:, -1]
    print(ys.shape[0], ' samples, and ', np.mean(ys) * 100, '% positives')

    np.random.seed(primeNumber)
    np.random.shuffle(data)
    nCoop = data[:, -1].sum()
    nAfter = data[:, 0].sum()
    nTotal = data.shape[0]
    print('coop samples', nCoop)
    print('merge after samples', nAfter)
    print('adv samples of merge in front', nTotal - nAfter - nCoop)

    sz = data.shape[0]
    trainRatio = 0.75

    sc_X = StandardScaler()
    data_train = data[:int(sz * trainRatio)]
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    np.savez(dir + "train_origin.npz", a=data_train)
    x_train = data_train[:, 1:-1]
    y_train = data_train[:, -1].astype(int)
    x_train = sc_X.fit_transform(x_train)
    np.savez(dir + "train_normalized.npz", a=x_train, b=y_train)

    data_validate = data[int(sz * trainRatio):]
    np.savez(dir + "validate_origin.npz", a=data_validate)
    x_validate = data_validate[:, 1:-1]
    x_validate = sc_X.transform(x_validate)
    y_validate = data_validate[:, -1].astype(int)
    np.savez(dir + "validate_normalized.npz", a=x_validate, b=y_validate)

    print(y_train.shape[0], ' trainning samples, and ',
          np.mean(y_train) * 100, '% positives')
    print(y_validate.shape[0], ' validate samples, and ',
          np.mean(y_validate) * 100, '% positives')
    return x_train, x_validate, y_train, y_validate, sc_X


def preprocess_both_dataset():
    # the data, split between train and test sets
    seqLen = 10
    datas = []
    datas.append(np.genfromtxt('0400pm-0415pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0500pm-0515pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0515pm-0530pm/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0400pm-0415pm/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0500pm-0515pm/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt('0515pm-0530pm/samples_merge_after_snapshots.csv', delimiter=','))

    for i in range(3):
        # 0 for merge in front
        temp = np.zeros((datas[i].shape[0], datas[i].shape[1] + 1))
        temp[:, 1:] = datas[i]
        datas[i] = temp
    for i in range(3, 6):
        # 1 for merge after
        temp = np.ones((datas[i].shape[0], datas[i].shape[1] + 1))
        temp[:, 1:] = datas[i]
        datas[i] = temp

    data = np.vstack((datas[0], datas[1]))
    for i in range(2, 6):
        data = np.vstack((data, datas[i]))
    ys = data[:, -1]
    print('For dataset us-80:')
    print(ys.shape[0] / seqLen, ' samples, and ', np.sum(ys) / ys.shape[0] * 100, '% positives')

    np.random.seed(primeNumber)

    # data=data[:100]
    # print('data', data[:,:4])

    indexes = np.arange(data.shape[0] / seqLen).astype(int)
    nCoop = data[:, -1].sum() / seqLen
    nAfter = data[:, 0].sum() / seqLen
    nTotal = data.shape[0] / seqLen
    print('total samples', data.shape[0], 'total sequences', indexes.shape[0])
    print('coop samples', nCoop)
    print('merge after samples', nAfter)
    print('adv samples of merge in front', nTotal - nAfter - nCoop)
    print('before shuffle, indexes[0:10]', indexes[0:10])
    np.random.shuffle(indexes)
    print('after shuffle, indexes[0:10]', indexes[0:10])

    data_train = []
    for i in range(indexes.shape[0]):
        for k in range(seqLen):
            data_train.append(data[indexes[i] * seqLen + k])
    sc_X = StandardScaler()
    data_train = np.vstack(data_train)
    np.savez("train_origin_us80.npz", a=data_train)
    # print('data train', data_train[:,:4])
    x_train = data_train[:, 3:-1]
    y_train = data_train[:, -1].astype(int)
    # visualize_sample(data_train[:, 0:-1] , y_train)
    # visualize_sample_sliced_by_velocity(data_train[:,0:-1], y_train)
    x_train = sc_X.fit_transform(x_train)
    np.savez("train_normalized_us80.npz", a=x_train, b=y_train)

    datas = []
    dir0 = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
    datas.append(np.genfromtxt(dir0 + '0750am-0805am/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0 + '0805am-0820am/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0 + '0820am-0835am/samples_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0 + '0750am-0805am/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0 + '0805am-0820am/samples_merge_after_snapshots.csv', delimiter=','))
    datas.append(np.genfromtxt(dir0 + '0820am-0835am/samples_merge_after_snapshots.csv', delimiter=','))

    for i in range(3):
        # 0 for merge in front
        temp = np.zeros((datas[i].shape[0], datas[i].shape[1] + 1))
        temp[:, 1:] = datas[i]
        datas[i] = temp
    for i in range(3, 6):
        # 1 for merge after
        temp = np.ones((datas[i].shape[0], datas[i].shape[1] + 1))
        temp[:, 1:] = datas[i]
        datas[i] = temp

    data = np.vstack((datas[0], datas[1]))
    for i in range(2, 6):
        data = np.vstack((data, datas[i]))
    ys = data[:, -1]
    print('For dataset us-101:')
    print(ys.shape[0] / seqLen, ' samples, and ', np.sum(ys) / ys.shape[0] * 100, '% positives')

    nCoop = data[:, -1].sum() / seqLen
    nAfter = data[:, 0].sum() / seqLen
    nTotal = data.shape[0] / seqLen
    print('total samples', data.shape[0], 'total sequences', indexes.shape[0])
    print('coop samples', nCoop)
    print('merge after samples', nAfter)
    print('adv samples of merge in front', nTotal - nAfter - nCoop)

    np.savez("validate_origin_us101.npz", a=data)
    # print('data train', data_train[:,:4])
    x_validate = data[:, 3:-1]
    x_validate = sc_X.transform(x_validate)
    y_validate = data[:, -1].astype(int)
    np.savez("validate_normalized_us101.npz", a=x_validate, b=y_validate)

    return x_train, x_validate, y_train, y_validate, sc_X


def preprocess_multiple():
    # the data, split between train and test sets
    data0 = np.genfromtxt('0400pm-0415pm/samples_multiple_snapshot.csv', delimiter=',')
    data1 = np.genfromtxt('0500pm-0515pm/samples_multiple_snapshot.csv', delimiter=',')
    data2 = np.genfromtxt('0515pm-0530pm/samples_multiple_snapshot.csv', delimiter=',')
    data = np.vstack((data0, data1))
    data = np.vstack((data, data2))
    ys = data[:, -1]
    print(ys.shape[0], ' samples, and ', np.sum(ys) / ys.shape[0] * 100, '% positives')

    data_train = []
    data_validate = []
    data_test = []
    mixture = False
    for i in range(data.shape[0]):
        if not mixture:
            laneChange = int(data[i, 0])
            if laneChange % 5 < 3:
                data_train.append(data[i, 1:])
            elif laneChange % 5 == 3:
                data_validate.append(data[i, 1:])
            else:
                data_test.append(data[i, 1:])
        else:
            if i % 5 < 3:
                data_train.append(data[i, 1:])
            elif i % 5 == 3:
                data_validate.append(data[i, 1:])
            else:
                data_test.append(data[i, 1:])
    sc_X = StandardScaler()
    data_train = np.vstack(data_train)
    x_train = data_train[:, :-1]
    x_train = sc_X.fit_transform(x_train)
    y_train = data_train[:, -1].astype(int)

    data_validate = np.vstack(data_validate)
    x_validate = data_validate[:, :-1]
    x_validate = sc_X.transform(x_validate)
    y_validate = data_validate[:, -1].astype(int)

    data_test = np.vstack(data_test)
    x_test = data_test[:, :-1]
    x_test = sc_X.transform(x_test)
    y_test = data_test[:, -1].astype(int)
    print(y_train.shape[0], ' trainning samples, and ',
          np.sum(y_train) / y_train.shape[0] * 100, '% positives')
    print(y_validate.shape[0], ' validate samples, and ',
          np.sum(y_validate) / y_validate.shape[0] * 100, '% positives')
    print(y_test.shape[0], ' test samples, and ',
          np.sum(y_test) / y_test.shape[0] * 100, '% positives')
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def visualize_sample(f):
    # preprocess()
    data0 = f['a']
    fig, ax = plt.subplots()
    data = data0[:, 1:-1]
    # print(data[:5])
    merge_after = data0[:, 0].astype(int)
    label = data0[:, -1]
    data00 = data[np.logical_and(label == 0, merge_after == 0)]
    data01 = data[np.logical_and(label == 0, merge_after == 1)]
    data1 = data[label == 1]
    ax.scatter(data00[:, 1], data00[:, 4], c='black', marker='x', label='adv(merge infront)')
    ax.scatter(data01[:, 1], data01[:, 4], c='red', marker='x', label='adv(merge after)')
    ax.scatter(data1[:, 1], data1[:, 4], c='blue', marker='o', label='coop')
    # print(data00[:10, [1,4]])

    plt.xlabel('$\Delta v$')
    plt.ylabel('$\Delta x$')
    plt.axis([-15, 15, -50, 100])
    # plt.axis([-7, 2, 0, 50])
    # plt.axis([-10, 10, -10, 50])
    ax.legend()
    # plt.show()


def visualize_ood_sample(x, x_validate, xGenerated):
    xIndistribution = np.concatenate((x, x_validate))
    fig, ax = plt.subplots()
    ax.scatter(xIndistribution[:,0], xIndistribution[:,2], c='blue', marker='o', label='In-distribution')
    ax.scatter(xGenerated[::10,0], xGenerated[::10,2], c='orange', marker='x', label='OOD')
    ax.set(xlim=(-30.2, 21.3), ylim=(-50., 100.))
    plt.legend()
    plt.show()


def get_threshold(x, percentage):
    from collections import Counter
    from numpy import linalg as LA

    print('In total, ', x.shape[0], ' samples')
    minDis = []
    for i in range(x.shape[0]):
        mask = np.ones(x.shape[0], dtype=np.bool)
        mask[i] = False
        diff = x - x[i]
        diff = LA.norm(diff, ord=2, axis=1)
        diff = min(diff[mask])
        minDis.append(diff)
    minDis = Counter(minDis)
    minDis = sorted(minDis.items())
    dis = []
    frequency = []
    count = 0.
    for p in minDis:
        dis.append(p[0])
        count += p[1]
        frequency.append(count)
    frequency = np.array(frequency)
    dis = np.array(dis)
    frequency /= count
    np.savez("/home/hh/data/ngsim/in_distribution_sample_minDis_distribution.npz", a=dis, b=frequency)
    plt.plot(dis, frequency, '-o')
    plt.axis([0, 3.0, 0., 1.1])
    plt.ylabel('percentage')
    plt.xlabel('minimum distance to in-distribution samples')
    plt.show()
    for i in range(dis.shape[0]-1, 0, -1):
        if frequency[i]>percentage and frequency[i-1]<=percentage:
            minDis = dis[i-1]
            break
    print('min dis ', minDis, 'for percentage ', percentage)
    return minDis


def extract_ood(x_train, x_validate, xGenerated, percentage = 0.99):
    from numpy import linalg as LA
    minDis = []
    xInDistribution = np.concatenate((x_train, x_validate))
    # minDis = get_threshold(xInDistribution, percentage)
    # minDis = 1.2846  # for percentage 0.98
    minDis = 1.4967  # for percentage 0.99
    mask = np.zeros(xGenerated.shape[0], dtype=np.bool)
    for i in range(xGenerated.shape[0]):
        diff = xInDistribution - xGenerated[i]
        diff = LA.norm(diff, ord=2, axis=1)
        diff = min(diff)
        if diff >= minDis:
            mask[i] = True
        if i % 10000 == 0: print('done for ', i+1, ', distance ', diff )
    xOOD = xGenerated[mask]
    print('from ', xGenerated.shape[0], 'samples, extracted ', xOOD.shape[0], 'OOD samples with threshold ', minDis)
    np.savez("/home/hh/data/ood_sample.npz", a=xOOD)


def generate_ood_ngsim():
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'train_origin.npz')
    data0 = f['a']
    f = np.load(dir + 'validate_origin.npz')
    data1 = f['a']
    data1 = data1[:, 1:]
    merge_after = data0[:, 0].astype(int)
    label = data0[:, -1]
    x = data0[:, 1:-1]
    feature_selected = [1, 2, 4, 5]
    x= x [:, feature_selected]
    x_validate = data1[:, feature_selected]
    print('dv0 range ', min(x[:,0]), max(x[:,0]))
    print('dv1 range ', min(x[:,1]), max(x[:,0]))
    print('dx0 range ', min(x[:,2]), max(x[:,2]))
    print('dx1 range ', min(x[:,3]), max(x[:,3]))
    dv0_min = 1.5 * min(x[:,0])
    dv0_max = 1.5 * max(x[:,0])
    dv1_min = 1.5 * min(x[:,1])
    dv1_max = 1.5 * max(x[:,1])
    detectionRange = 100.
    pointsInEachDim = 20
    uniformGrid = False
    if uniformGrid:
        dv0 = np.linspace(dv0_min, dv0_max + 0.5, pointsInEachDim)
        dv1 = np.linspace(dv1_min, dv1_max + 0.5, pointsInEachDim)
        dx0 = np.linspace(-detectionRange/2., detectionRange + 0.5, pointsInEachDim)
        dx1 = np.linspace(-detectionRange, detectionRange/2., pointsInEachDim)
        v0, v1, x0, x1 = np.meshgrid(dv0, dv1, dx0, dx1)
        xGenerated = np.column_stack([v0.flatten(), v1.flatten(), x0.flatten(), x1.flatten()])
    else:
        l = [dv0_min, dv1_min, -detectionRange/2., -detectionRange]
        h = [dv0_max, dv1_max, detectionRange, detectionRange/2.]
        xGenerated = np.random.uniform(low=l, high=h, size = (pointsInEachDim**4, 4))

    visualize_ood_sample(x, x_validate, xGenerated)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x)
    print('feature mean ', sc_X.mean_)
    print('feature std ', np.sqrt(sc_X.var_))
    x_validate = sc_X.transform(x_validate)
    xGenerated = sc_X.transform(xGenerated)
    # minDis, x_ood = ood_label(x_train, x_validate, xGenerated)
    extract_ood(x_train, x_validate, xGenerated)
    # f = np.load("/home/hh/data/ngsim/generated_sample_minDis.npz")
    # minDis = f['a']



# preprocess()
generate_ood_ngsim()
# dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
# dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
# f = np.load(dir+'train_origin.npz')
# visualize_sample(f)
# f = np.load(dir+'validate_origin.npz')
# visualize_sample(f)
# plt.show()
