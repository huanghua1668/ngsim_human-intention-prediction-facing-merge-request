import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

primeNumber = 3
seqLen = 10

def transform_both_dataset():
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'us80.npz')
    us80 = f['a']

    dir = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'us101.npz')
    us101 = f['a']

    mask = np.array([1, 2, 4, 5]) # already been chosen to delete
    us80_x = us80[:, 1:-1]
    us80_x = us80_x[:, mask]
    us80_y = us80[:, -1]
    us80_y = us80_y.astype(int)
    us101_x = us101[:, 1:-1]
    us101_x = us101_x[:, mask]
    us101_y = us101[:, -1]
    us101_y = us101_y.astype(int)

    dir = '/home/hh/data/ngsim/'
    # train us80, validate us101
    sc_X = StandardScaler()
    us80 = sc_X.fit_transform(us80_x)
    np.savez(dir + "us80_train.npz", a=us80, b=us80_y)
    us101 = sc_X.transform(us101_x)
    np.savez(dir + "us101_validate.npz", a=us101, b=us101_y)
    # train us101, validate us80
    sc_X = StandardScaler()
    us101 = sc_X.fit_transform(us101_x)
    np.savez(dir + "us101_train.npz", a=us101, b=us101_y)
    us80 = sc_X.transform(us80_x)
    np.savez(dir + "us80_validate.npz", a=us80, b=us80_y)


def load_data_both_dataset(trainUS80 = True):
    transformed = False
    if not transformed:
        transform_both_dataset()
    dir = '/home/hh/data/ngsim/'
    if trainUS80:
        f = np.load(dir + "us80_train.npz")
        us80_x = f['a']
        us80_y = f['b']
        f = np.load(dir + "us101_validate.npz")
        us101_x = f['a']
        us101_y = f['b']
        return us80_x, us80_y, us101_x, us101_y
    else:
        f = np.load(dir + "us101_train.npz")
        us101_x = f['a']
        us101_y = f['b']
        f = np.load(dir + "us80_validate.npz")
        us80_x = f['a']
        us80_y = f['b']
        return us101_x, us101_y, us80_x, us80_y

    # return (x_train, y_train, x_validate, y_validate, x_ood)


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
    # np.savez(dir + "train_origin.npz", a=data_train)
    np.savez(dir + "train_origin_cleaned.npz", a=data_train)
    x_train = data_train[:, 1:-1]
    y_train = data_train[:, -1].astype(int)
    x_train = sc_X.fit_transform(x_train)
    # np.savez(dir + "train_normalized.npz", a=x_train, b=y_train)
    np.savez(dir + "train_normalized_cleaned.npz", a=x_train, b=y_train)

    data_validate = data[int(sz * trainRatio):]
    # np.savez(dir + "validate_origin.npz", a=data_validate)
    np.savez(dir + "validate_origin_cleaned.npz", a=data_validate)
    x_validate = data_validate[:, 1:-1]
    x_validate = sc_X.transform(x_validate)
    y_validate = data_validate[:, -1].astype(int)
    # np.savez(dir + "validate_normalized.npz", a=x_validate, b=y_validate)
    np.savez(dir + "validate_normalized_cleaned.npz", a=x_validate, b=y_validate)

    print(y_train.shape[0], ' trainning samples, and ',
          np.mean(y_train) * 100, '% positives')
    print(y_validate.shape[0], ' validate samples, and ',
          np.mean(y_validate) * 100, '% positives')
    return x_train, x_validate, y_train, y_validate, sc_X


def load_data():
    mask = np.array([1, 2, 4, 5]) # already been chosen to delete
    # dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    dir = '/home/hh/Dropbox/Hua/csnn/data/'
    f = np.load(dir + "train.npz")
    # f = np.load(dir + "train_normalized.npz")
    x_train = f['a']
    # x_train = x_train[:, mask]
    y_train = f['b']
    # f = np.load(dir + "validate_normalized_cleaned.npz")
    f = np.load(dir + "validate.npz")
    x_validate = f['a']
    # x_validate = x_validate[:, mask]
    y_validate = f['b']
    f = np.load(dir + "ood_sample.npz")
    # f = np.load("/home/hh/data/ood_sample_cleaned.npz")
    # f = np.load("/home/hh/data/ood_sample.npz")
    xOOD = f['a']
    return x_train, y_train, x_validate, y_validate, xOOD


def cal_min_dis(a, b):
    '''for every sample in a, calculate the shortest distance between it and samples in b'''
    from numpy import linalg as LA
    minDis = []

    for i in range(a.shape[0]):
        sample = a[i]
        diff = b - sample
        diff = LA.norm(diff, ord=2, axis=1)
        diff = min(diff)
        minDis.append(diff)
    return np.array(minDis)


def cal_distance():
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'us80.npz')
    us80 = f['a']

    dir = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'us101.npz')
    us101 = f['a']

    mask = np.array([1, 2, 4, 5]) # already been chosen to delete
    us80_x = us80[:, 1:-1]
    us80_x = us80_x[:, mask]
    us101_x = us101[:, 1:-1]
    us101_x = us101_x[:, mask]
    num_us80 = us80_x.shape[0]

    # scale
    sc_X = StandardScaler()
    x = np.concatenate((us80_x, us101_x))
    x = sc_X.fit_transform(x)
    us80 = x[:num_us80]
    us101 = x[num_us80:]

    # get the threshold
    percentage = 0.99
    dir = '/home/hh/data/ngsim/'
    minDis_us80 = get_threshold(us80, percentage, dir+'min_dis_within_us80.npz')
    minDis_us101 = get_threshold(us101, percentage, dir+'min_dis_within_us101.npz')
    minDis = np.array([minDis_us80, minDis_us101])
    # for each sample in us80, calculate min dis between it and samples in us101
    dis_us80_to_us101 = cal_min_dis(us80, us101)
    dis_us101_to_us80 = cal_min_dis(us101, us80)

    ood_us80_wrt_us101 = dis_us80_to_us101 > minDis_us101
    ood_us80_wrt_us101 = np.mean(ood_us80_wrt_us101)

    ood_us101_wrt_us80 = dis_us101_to_us80 > minDis_us80
    ood_us101_wrt_us80 = np.mean(ood_us101_wrt_us80)

    print('percentage of us80 as OOD wrt us101 {:.4f}'.format(ood_us80_wrt_us101))
    print('percentage of us101 as OOD wrt us80 {:.4f}'.format(ood_us101_wrt_us80))
    np.savez(dir + "min_dis.npz", a=us80, b=us101, c=minDis, d=dis_us80_to_us101, e=dis_us101_to_us80)


def preprocess_both_dataset():
    # the data, split between train and test sets
    primeNumber = 3
    datas = []
    I80 = True
    # I80 = False

    # for i-80
    if I80:
        dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
        datas.append(np.genfromtxt(dir+'0400pm-0415pm/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0500pm-0515pm/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0515pm-0530pm/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0400pm-0415pm/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0500pm-0515pm/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0515pm-0530pm/samples_merge_after_snapshots.csv', delimiter=','))
    # for i-101
    else:
        dir = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
        datas.append(np.genfromtxt(dir + '0750am-0805am/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0805am-0820am/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0820am-0835am/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0750am-0805am/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0805am-0820am/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0820am-0835am/samples_merge_after_snapshots.csv', delimiter=','))

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
    if I80:
        dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
        np.savez(dir + "us80.npz", a=data)
        f = np.load(dir+'us80.npz')
        visualize_sample(f)
    else:
        dir = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
        np.savez(dir + "us101.npz", a=data)
        f = np.load(dir+'us101.npz')
        visualize_sample(f)


def inspect_abnormal():
    '''check whether the abnormal samples make sense'''
    vehicleLength = 5.
    detectionRange = 100
    laneWidth = 3.7

    # for i-80
    dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    # dir = dir0 + '0400pm-0415pm/'
    # dir = dir0 + '0500pm-0515pm/'
    dir = dir0 + '0515pm-0530pm/'


    data = np.genfromtxt(dir+'lane_changes.csv', delimiter=',')
    # output = open(dir+'samples_snapshots.csv', 'w')

    # writer = csv.writer(output)
    count = 0
    minTimeBeforeLC = 1.5  # from decision to cross lane divider
    minTimeAfterLC = 1.5  # from cross lane divider to end of lane change
    observationLength = 5
    cooperates = 0

    fig, ax = plt.subplots()
    ax.set(xlim=(-15, 15), ylim=(-50., 100.))
    for i in range(0, int(data[-1, 0]) + 1):
        # for i in range(0, 5):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.):
            # lag vehicle not show up as x_lag=0 (lateral position)
            start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -minTimeBeforeLC: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < minTimeAfterLC: continue
        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        # print(count, 'after trim ', i, start0, end0, end0 - start0, data[start0, 1], data[end0, 1])

        # handle missing vehicle values
        for j in range(start0, start0 + observationLength):
            if data[j, 11] == 0:  # no corresponding preceding obstacle for target lane
                data[j, 12] = data[j, 4]
                data[j, 11] = data[j, 3] + detectionRange
                data[j, 10] = 0.5 * laneWidth
                if data[j, 14] < 0.:
                    data[j, 10] *= -1.
                # print('no leading vehilce at ', j)
            if data[j, 7] == 0:  # no corresponding obstacle for leading in old lane
                data[j, 8] = data[j, 4]
                data[j, 7] = data[j, 3] + detectionRange
                data[j, 6] = 0.5 * laneWidth
                if data[j, 2] < 0.:
                    data[j, 6] *= -1.
                # print('no leading vehilce at original lane at ', j)

        j = start0 + observationLength -1
        dx0 = data[j, 3] - data[j, 15]
        dx1 = data[j, 3] - data[j, 11]
        dx2 = data[j, 3] - data[j, 7]
        dy0 = data[j, 2] - data[j, 14]
        dy1 = data[j, 2] - data[j, 10]
        dy2 = data[j, 2] - data[j, 6]
        u0 = np.mean(data[start0:start0 + observationLength, 4])
        du0 = np.mean(data[start0:start0 + observationLength, 4] - data[start0:start0 + observationLength, 16])
        du1 = np.mean(data[start0:start0 + observationLength, 4] - data[start0:start0 + observationLength, 12])
        du2 = np.mean(data[start0:start0 + observationLength, 4] - data[start0:start0 + observationLength, 8])
        y = data[start0, -1]
        cooperates += y

        sample = [count, u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]
        ###
        if y == 0 and du0 > 3. and  dx0 >= 14.:
            print('abnormal adv at i {}, dv {:.2f}, dx {:.2f}'.format(i, du0, dx0 ))
            plt.scatter(du0, dx0, color='red', marker='o')
        if y == 1 and du0 < -2. and  dx0 <= 10.:
            print('abnormal coop at i {}, dv {:.2f}, dx {:.2f}'.format(i, du0, dx0 ))
            plt.scatter(du0, dx0, color='blue', marker='o')

        # writer.writerow(np.around(sample, decimals=3))
        count += 1

# def preprocess_multiple():
#     # the data, split between train and test sets
#     data0 = np.genfromtxt('0400pm-0415pm/samples_multiple_snapshot.csv', delimiter=',')
#     data1 = np.genfromtxt('0500pm-0515pm/samples_multiple_snapshot.csv', delimiter=',')
#     data2 = np.genfromtxt('0515pm-0530pm/samples_multiple_snapshot.csv', delimiter=',')
#     data = np.vstack((data0, data1))
#     data = np.vstack((data, data2))
#     ys = data[:, -1]
#     print(ys.shape[0], ' samples, and ', np.sum(ys) / ys.shape[0] * 100, '% positives')
#
#     data_train = []
#     data_validate = []
#     data_test = []
#     mixture = False
#     for i in range(data.shape[0]):
#         if not mixture:
#             laneChange = int(data[i, 0])
#             if laneChange % 5 < 3:
#                 data_train.append(data[i, 1:])
#             elif laneChange % 5 == 3:
#                 data_validate.append(data[i, 1:])
#             else:
#                 data_test.append(data[i, 1:])
#         else:
#             if i % 5 < 3:
#                 data_train.append(data[i, 1:])
#             elif i % 5 == 3:
#                 data_validate.append(data[i, 1:])
#             else:
#                 data_test.append(data[i, 1:])
#     sc_X = StandardScaler()
#     data_train = np.vstack(data_train)
#     x_train = data_train[:, :-1]
#     x_train = sc_X.fit_transform(x_train)
#     y_train = data_train[:, -1].astype(int)
#
#     data_validate = np.vstack(data_validate)
#     x_validate = data_validate[:, :-1]
#     x_validate = sc_X.transform(x_validate)
#     y_validate = data_validate[:, -1].astype(int)
#
#     data_test = np.vstack(data_test)
#     x_test = data_test[:, :-1]
#     x_test = sc_X.transform(x_test)
#     y_test = data_test[:, -1].astype(int)
#     print(y_train.shape[0], ' trainning samples, and ',
#           np.sum(y_train) / y_train.shape[0] * 100, '% positives')
#     print(y_validate.shape[0], ' validate samples, and ',
#           np.sum(y_validate) / y_validate.shape[0] * 100, '% positives')
#     print(y_test.shape[0], ' test samples, and ',
#           np.sum(y_test) / y_test.shape[0] * 100, '% positives')
#     return x_train, x_validate, x_test, y_train, y_validate, y_test


def visualize_sample():
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'us80.npz')
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
    ax.scatter(data00[:, 1], data00[:, 4], c='black', marker='x', label='adv (merge infront)')
    ax.scatter(data01[:, 1], data01[:, 4], c='red', marker='x', label='adv (merge after)')
    ax.scatter(data1[:, 1], data1[:, 4], c='blue', marker='o', label='coop')
    # print(data00[:10, [1,4]])

    plt.ylabel('$\Delta x \enspace [m]$', fontsize=25)
    plt.xlabel('$\Delta v \enspace [m/s^2]$', fontsize=25)
    plt.axis([-15, 15, -50, 100])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=25)
    ax.legend(prop={'size': 25})
    # plt.show()


def visualize_sample_labeled_by_dx():
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    records0=np.genfromtxt(dir+'0400pm-0415pm/samples_relabeled_by_decrease_in_dx.csv', delimiter=',')
    records1=np.genfromtxt(dir+'0500pm-0515pm/samples_relabeled_by_decrease_in_dx.csv', delimiter=',')
    records=np.concatenate((records0, records1), axis=0)

    samples0=records[records[:,-1]==0]
    samples1=records[records[:,-1]==1]
    plt.scatter(samples0[:,1], samples0[:,0], color="black", marker='x',
            label='adv')
    plt.scatter(samples1[:,1], samples1[:,0], color="blue", marker='o',
            label='coop')
    plt.ylabel('$\Delta x \enspace [m]$', fontsize=25)
    plt.xlabel('$\Delta v \enspace [m/s^2]$', fontsize=25)
    plt.axis([-15, 15, -50, 100])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=25)
    plt.show()

def visualize_ood_sample(x, x_validate, xGenerated):
    xIndistribution = np.concatenate((x, x_validate))
    fig, ax = plt.subplots()
    ax.scatter(xIndistribution[:,0], xIndistribution[:,2], c='blue', marker='o', label='In-distribution')
    ax.scatter(xGenerated[::10,0], xGenerated[::10,2], c='orange', marker='x', label='OOD')
    ax.set(xlim=(-30.2, 21.3), ylim=(-50., 100.))
    plt.legend()
    plt.show()


def get_threshold(x, percentage, dir):
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
    minDis0 = minDis.copy()

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
    # np.savez(dir, a=dis, b=frequency)
    # plt.plot(dis, frequency, '-o')
    # plt.axis([0, 3.0, 0., 1.1])
    # plt.ylabel('percentage')
    # plt.xlabel('minimum distance to in-distribution samples')
    # plt.show()
    for i in range(dis.shape[0]-1, 0, -1):
        if frequency[i]>percentage and frequency[i-1]<=percentage:
            minDis = dis[i-1]
            break
    print('min dis ', minDis, 'for percentage ', percentage)
    oodLabel = minDis0>=minDis
    return minDis, oodLabel


def extract_ood(xInDistribution, xGenerated, minDis):
    # minDis = 1.4967  # for percentage 0.99
    from numpy import linalg as LA
    # minDis = get_threshold(xInDistribution, percentage)
    # minDis = 1.2846  # for percentage 0.98
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
    # np.savez("/home/hh/data/ood_sample.npz", a=xOOD)
    # np.savez("/home/hh/data/ood_sample_cleaned.npz", a=xOOD)
    return mask, xOOD


def generate_ood_ngsim():
    percentage = 0.99
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    # f = np.load(dir + 'train_origin.npz')
    f = np.load(dir + 'train_origin_cleaned.npz')
    data0 = f['a']
    # f = np.load(dir + 'validate_origin.npz')
    f = np.load(dir + 'validate_origin_cleaned.npz')
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
    x0 = x.copy()
    x_validate0 = x_validate.copy()
    xGenerated0 = xGenerated.copy()
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x)
    print('feature mean ', sc_X.mean_)
    print('feature std ', np.sqrt(sc_X.var_))
    x_validate = sc_X.transform(x_validate)
    xGenerated = sc_X.transform(xGenerated)
    # minDis, x_ood = ood_label(x_train, x_validate, xGenerated)
    minDis, _ = get_threshold(x_train, percentage, dir+'min_dis.png')
    xInDistribution = np.concatenate((x_train, x_validate))
    xInDistribution0 = np.concatenate((x0, x_validate0))
    # extract_ood(xInDistribution, xGenerated, minDis = 1.4967)
    mask, x_ood = extract_ood(xInDistribution, xGenerated, minDis)
    # f = np.load("/home/hh/data/ngsim/generated_sample_minDis.npz")
    # minDis = f['a']
    visualize_ood_sample(x0, x_validate0, xGenerated0[mask])
    print('for ood:')
    print('dv0 range ', min(xGenerated0[mask][:,0]), max(xGenerated0[mask][:,0]))
    print('dv1 range ', min(xGenerated0[mask][:,1]), max(xGenerated0[mask][:,0]))
    print('dx0 range ', min(xGenerated0[mask][:,2]), max(xGenerated0[mask][:,2]))
    print('dx1 range ', min(xGenerated0[mask][:,3]), max(xGenerated0[mask][:,3]))


def generate_ood(x):
    '''input x, output generated OOD;
       should not be normalized '''
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
    return xGenerated


def prepare_validate_and_generate_ood():
    percentage = 0.99
    featureMask = np.array([1, 2, 4, 5]) # already been chosen to delete
    np.random.seed(0)

    dir = '/home/hh/data/ngsim/'
    f = np.load(dir + "us80.npz")
    us80 = f['a']
    f = np.load(dir + "us101.npz")
    us101 = f['a']
    print('us80: ', us80.shape[0], ' samples,', np.sum(us80[:,-1]), 'coop', np.sum(us80[:,0]), 'merge after')
    print('us101: ', us101.shape[0], ' samples,', np.sum(us101[:,-1]), 'coop', np.sum(us101[:,0]), 'merge after')
    combined = np.concatenate((us80, us101))
    np.random.seed(primeNumber)
    np.random.shuffle(combined)
    x_combined = combined[:, 1:-1]
    y_combined = combined[:, -1]
    x_combined = x_combined[:, featureMask]
    x_combined0 = x_combined.copy()
    sc_X = StandardScaler()
    x_combined = sc_X.fit_transform(x_combined)
    # need to scale first before any calculating of distance

    # clean data, 99% dis to in-dis, 1% to OOD
    minDis, oodLabel = get_threshold(x_combined, percentage, dir)
    x_inDis = x_combined[~oodLabel]
    y_inDis = y_combined[~oodLabel]
    x_ood = x_combined[oodLabel]
    print('total sample ', x_combined.shape[0], ', in dis ', x_inDis.shape[0], ', ood ', x_ood.shape[0])

    # generate ood samples
    combined_ood = generate_ood(x_combined0[~oodLabel])
    combined_ood0 = combined_ood.copy()
    combined_ood = sc_X.transform(combined_ood)
    mask, xGenerated = extract_ood(x_inDis, combined_ood, minDis)

    # split to train and test
    sz = x_inDis.shape[0]
    trainRatio = 0.75
    x_train = x_inDis[:int(sz * trainRatio)]
    y_train = y_inDis[:int(sz * trainRatio)].astype(int)
    x_test = x_inDis[int(sz * trainRatio):]
    y_test = y_inDis[int(sz * trainRatio):].astype(int)
    print(y_train.shape[0], ' trainning samples, and ',
          np.mean(y_train) * 100, '% positives')
    print(y_test.shape[0], ' validate samples, and ',
          np.mean(y_test) * 100, '% positives')

    x_train0 = x_combined0[~oodLabel][:int(sz * trainRatio)]
    print('for training:')
    print('dv0 range ', min(x_train0[:,0]), max(x_train0[:,0]))
    print('dv1 range ', min(x_train0[:,1]), max(x_train0[:,0]))
    print('dx0 range ', min(x_train0[:,2]), max(x_train0[:,2]))
    print('dx1 range ', min(x_train0[:,3]), max(x_train0[:,3]))
    print('for ood:')
    print('dv0 range ', min(combined_ood0[mask][:,0]), max(combined_ood0[mask][:,0]))
    print('dv1 range ', min(combined_ood0[mask][:,1]), max(combined_ood0[mask][:,0]))
    print('dx0 range ', min(combined_ood0[mask][:,2]), max(combined_ood0[mask][:,2]))
    print('dx1 range ', min(combined_ood0[mask][:,3]), max(combined_ood0[mask][:,3]))
    # x_ood = sc_X.transform(xGenerated)
    # np.savez(dir + "combined_dataset.npz", a=x_train, b=y_train, c=x_test, d=y_test, e=x_ood)
    np.savez(dir + "combined_dataset.npz", a=x_train, b=y_train, c=x_test, d=y_test, e=xGenerated)


def generate_two_gaussian():
    from plot_utils import plot_two_gaussian
    np.random.seed(0)
    variance = 2.
    n_train = 1500
    n_test = 1000
    mean0 = [-2., 0.]
    mean1 = [2., 0.]
    cov = [[variance, 0], [0, variance]]
    x0 = np.random.multivariate_normal(mean0, cov, n_train)
    x0 = np.hstack((x0, np.zeros((x0.shape[0],1))))
    x1 = np.random.multivariate_normal(mean1, cov, n_train)
    x1 = np.hstack((x1, np.ones((x1.shape[0],1))))
    x_train = np.concatenate((x0, x1))
    np.random.shuffle(x_train)
    plot_two_gaussian(x_train)

    x0 = np.random.multivariate_normal(mean0, cov, n_test)
    x0 = np.hstack((x0, np.zeros((x0.shape[0],1))))
    x1 = np.random.multivariate_normal(mean1, cov, n_test)
    x1 = np.hstack((x1, np.ones((x1.shape[0],1))))
    x_test = np.concatenate((x0, x1))
    np.random.shuffle(x_test)
    dir = '/home/hh/data/two_gaussian/'
    np.savez(dir+'two_gaussian_train_test.npz', a=x_train, b=x_test)

if __name__ == "__main__":
    # preprocess()
    # preprocess_both_dataset()
    # generate_ood_ngsim()
    visualize_sample()
    # visualize_sample_labeled_by_dx()
    # visualize_sample(f)
    # f = np.load(dir+'validate_origin.npz')
    # visualize_sample(f)
    # inspect_abnormal()
    # cal_distance()
    # prepare_validate_and_generate_ood()
    # generate_two_gaussian()
    # prepare_validate_and_generate_ood()
    plt.show()
