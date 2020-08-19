import numpy as np
import csv
import matplotlib.pyplot as plt

detectionRange = 100
laneWidth = 3.7


def extract_sample():
    vehicleLength = 5.
    data = np.genfromtxt('lane_changes.csv', delimiter=',')
    output = open('samples.csv', 'w')
    writer = csv.writer(output)
    for i in range(0, int(data[-1, 0]) + 1):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division

        du0 = data[start0, 4] - data[start0, 16]
        du1 = data[start0, 4] - data[start0, 12]
        du2 = data[start0, 4] - data[start0, 8]
        dx0 = data[start0, 3] - data[start0, 15]
        dx1 = data[start0, 3] - data[start0, 11]
        dx2 = data[start0, 3] - data[start0, 7]
        dy0 = data[start0, 2] - data[start0, 14]
        dy1 = data[start0, 2] - data[start0, 10]
        dy2 = data[start0, 2] - data[start0, 6]

        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        y = data[start0, -1]
        sample = [data[start0, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

        if y == 1:
            plt.scatter(du0, dx0, color='blue', marker='o')
        else:
            plt.scatter(du0, dx0, color='red', marker='o')

        writer.writerow(np.around(sample, decimals=3))

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-10, 10, -30, 70])
    plt.show()


def extract_sample_3seconds():
    vehicleLength = 5.
    data = np.genfromtxt('lane_changes.csv', delimiter=',')
    output = open('samples_3seconds_begin_after.csv', 'w')
    writer = csv.writer(output)
    for i in range(0, int(data[-1, 0]) + 1):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division
        # print('before trim ', i, start0, end0, end0-start0)
        while data[start0, 1] < -3.: start0 += 1
        while data[end0, 1] > 3.:   end0 -= 1
        # print('after trim ', i, start0, end0, end0-start0)

        # handle missing vehicle values
        if data[start0, 11] == 0:  # no corresponding preceding obstacle for target lane
            data[start0, 12] = data[start0, 4]
            data[start0, 11] = data[start0, 3] + detectionRange
            data[start0, 10] = 0.5 * laneWidth
            if data[start0, 14] < 0.:
                data[start0, 10] *= -1.
            print('no leading vehilce at ', i)
        if data[start0, 7] == 0:  # no corresponding obstacle for leading in old lane
            data[start0, 8] = data[start0, 4]
            data[start0, 7] = data[start0, 3] + detectionRange
            data[start0, 6] = 0.5 * laneWidth
            if data[start0, 2] < 0.:
                data[start0, 6] *= -1.
            print('no leading vehilce at original lane at ', i)

        du0 = data[start0, 4] - data[start0, 16]
        du1 = data[start0, 4] - data[start0, 12]
        du2 = data[start0, 4] - data[start0, 8]
        dx0 = data[start0, 3] - data[start0, 15]
        dx1 = data[start0, 3] - data[start0, 11]
        dx2 = data[start0, 3] - data[start0, 7]
        dy0 = data[start0, 2] - data[start0, 14]
        dy1 = data[start0, 2] - data[start0, 10]
        dy2 = data[start0, 2] - data[start0, 6]

        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        y = data[start0, -1]
        sample = [data[start0, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

        if y == 1:
            plt.scatter(du0, dx0, color='blue', marker='o')
        else:
            plt.scatter(du0, dx0, color='red', marker='o')

        writer.writerow(np.around(sample, decimals=3))

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-15, 15, -50, 100])
    plt.show()


def extract_snapshots_3seconds():
    vehicleLength = 5.
    data = np.genfromtxt('lane_changes.csv', delimiter=',')
    output = open('samples_snapshots.csv', 'w')
    writer = csv.writer(output)
    count = 0
    for i in range(0, int(data[-1, 0]) + 1):
        # for i in range(0, 5):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division
        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        # print('before trim ', i, start0, end0, end0-start0)
        # while data[start0,1]<-3.: start0+=1
        while data[end0, 1] > -2.:   end0 -= 1
        print(count, 'after trim ', i, start0, end0, end0 - start0,
              data[start0, 1], data[end0, 1])
        for j in range(start0, end0 + 1):
            # handle missing vehicle values
            if data[j, 11] == 0:  # no corresponding preceding obstacle for target lane
                data[j, 12] = data[j, 4]
                data[j, 11] = data[j, 3] + detectionRange
                data[j, 10] = 0.5 * laneWidth
                if data[j, 14] < 0.:
                    data[j, 10] *= -1.
                print('no leading vehilce at ', j)
            if data[j, 7] == 0:  # no corresponding obstacle for leading in old lane
                data[j, 8] = data[j, 4]
                data[j, 7] = data[j, 3] + detectionRange
                data[j, 6] = 0.5 * laneWidth
                if data[j, 2] < 0.:
                    data[j, 6] *= -1.
                print('no leading vehilce at original lane at ', j)
            dt = data[j, 1]
            du0 = data[j, 4] - data[j, 16]
            du1 = data[j, 4] - data[j, 12]
            du2 = data[j, 4] - data[j, 8]
            dx0 = data[j, 3] - data[j, 15]
            dx1 = data[j, 3] - data[j, 11]
            dx2 = data[j, 3] - data[j, 7]
            dy0 = data[j, 2] - data[j, 14]
            dy1 = data[j, 2] - data[j, 10]
            dy2 = data[j, 2] - data[j, 6]

            y = data[j, -1]
            sample = [count, dt, data[j, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

            if y == 1:
                plt.scatter(du0, dx0, color='blue', marker='o')
            else:
                plt.scatter(du0, dx0, color='red', marker='o')

            writer.writerow(np.around(sample, decimals=3))
        count += 1

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-15, 15, -50, 100])
    plt.show()


def extract_samples_modified_loss_function():
    vehicleLength = 5.
    dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    dir = dir0 + '0400pm-0415pm/'
    # dir = dir0 + '0500pm-0515pm/'
    # dir = dir0 + '0515pm-0530pm/'
    data = np.genfromtxt(dir+'lane_changes.csv', delimiter=',')
    output = open(dir+'samples_snapshots.csv', 'w')
    writer = csv.writer(output)
    count = 0
    minTimeBeforeLC = 1.5  # from decision to cross lane divider
    minTimeAfterLC = 1.5  # from cross lane divider to end of lane change
    observationLength = 5
    cooperates = 0
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
        print(count, 'after trim ', i, start0, end0, end0 - start0, data[start0, 1], data[end0, 1])

        # handle missing vehicle values
        for j in range(start0, start0 + observationLength):
            if data[j, 11] == 0:  # no corresponding preceding obstacle for target lane
                data[j, 12] = data[j, 4]
                data[j, 11] = data[j, 3] + detectionRange
                data[j, 10] = 0.5 * laneWidth
                if data[j, 14] < 0.:
                    data[j, 10] *= -1.
                print('no leading vehilce at ', j)
            if data[j, 7] == 0:  # no corresponding obstacle for leading in old lane
                data[j, 8] = data[j, 4]
                data[j, 7] = data[j, 3] + detectionRange
                data[j, 6] = 0.5 * laneWidth
                if data[j, 2] < 0.:
                    data[j, 6] *= -1.
                print('no leading vehilce at original lane at ', j)

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
            print('abnormal adv at i=', i, 'dv=', du0, ', dx=', dx0)

        ###
        if y == 1:
            plt.scatter(du0, dx0, color='blue', marker='o')
        else:
            plt.scatter(du0, dx0, color='red', marker='o')

        writer.writerow(np.around(sample, decimals=3))
        count += 1
    print('coops, total, coop rate', cooperates, count, cooperates/count)
    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-15, 15, -50, 100])
    plt.show()


def extract_samples_modified_loss_function_relabel_as_change_in_distance():
    vehicleLength = 5.
    data = np.genfromtxt('lane_changes.csv', delimiter=',')
    output = open('samples_snapshots_relabeled.csv', 'w')
    writer = csv.writer(output)
    count = 0
    for i in range(0, int(data[-1, 0]) + 1):
        # for i in range(0, 5):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division
        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        # print('before trim ', i, start0, end0, end0-start0)
        # while data[start0,1]<-3.: start0+=1
        while data[start0, 1] < -4.: start0 += 1
        time0 = start0
        while data[time0, 1] < -3.: time0 += 1

        y = 1
        lcTime = start0
        while data[lcTime, 1] < 0.: lcTime += 1
        # if data[end0, 3]-data[end0,15] < data[start0, 3]-data[start0,15]:
        if data[lcTime, 3] - data[lcTime, 15] < data[time0, 3] - data[time0, 15]:
            y = 0

        # while data[end0,1]  > -2.:   end0-=1
        print(count, 'after trim ', i, start0, end0, end0 - start0,
              data[start0, 1], data[end0, 1])
        # for j in range(start0, end0+1):
        for j in range(start0, start0 + 10):
            # handle missing vehicle values
            if data[j, 11] == 0:  # no corresponding preceding obstacle for target lane
                data[j, 12] = data[j, 4]
                data[j, 11] = data[j, 3] + detectionRange
                data[j, 10] = 0.5 * laneWidth
                if data[j, 14] < 0.:
                    data[j, 10] *= -1.
                print('no leading vehilce at ', j)
            if data[j, 7] == 0:  # no corresponding obstacle for leading in old lane
                data[j, 8] = data[j, 4]
                data[j, 7] = data[j, 3] + detectionRange
                data[j, 6] = 0.5 * laneWidth
                if data[j, 2] < 0.:
                    data[j, 6] *= -1.
                print('no leading vehilce at original lane at ', j)
            dt = data[j, 1]
            du0 = data[j, 4] - data[j, 16]
            du1 = data[j, 4] - data[j, 12]
            du2 = data[j, 4] - data[j, 8]
            dx0 = data[j, 3] - data[j, 15]
            dx1 = data[j, 3] - data[j, 11]
            dx2 = data[j, 3] - data[j, 7]
            dy0 = data[j, 2] - data[j, 14]
            dy1 = data[j, 2] - data[j, 10]
            dy2 = data[j, 2] - data[j, 6]

            # y=data[j, -1]
            sample = [count, dt, data[j, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

            if y == 1:
                if j == start0:
                    plt.scatter(du0, dx0, color='blue', marker='o')
            else:
                if j == start0:
                    plt.scatter(du0, dx0, color='red', marker='o')

            writer.writerow(np.around(sample, decimals=3))
        count += 1

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-15, 15, -50, 100])
    plt.show()


def extract_sample_multiple_snapshot():
    vehicleLength = 5.
    data = np.genfromtxt('lane_changes.csv', delimiter=',')
    output = open('samples_multiple_snapshot.csv', 'w')
    writer = csv.writer(output)
    laneChanges = 0
    for i in range(0, int(data[-1, 0]) + 1):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division
        print('before trim ', i, start0, end0, end0 - start0)
        while data[start0, 1] < -3.5: start0 += 1
        # while data[end0,1]  > 3.5:   end0-=1
        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        print('after trim ', i, start0, end0, end0 - start0)

        while data[start0, 1] <= -3.:
            du0 = data[start0, 4] - data[start0, 16]
            du1 = data[start0, 4] - data[start0, 12]
            du2 = data[start0, 4] - data[start0, 8]
            dx0 = data[start0, 3] - data[start0, 15]
            dx1 = data[start0, 3] - data[start0, 11]
            dx2 = data[start0, 3] - data[start0, 7]
            dy0 = data[start0, 2] - data[start0, 14]
            dy1 = data[start0, 2] - data[start0, 10]
            dy2 = data[start0, 2] - data[start0, 6]

            y = data[start0, -1]
            sample = [laneChanges, data[start0, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

            if y == 1:
                plt.scatter(du0, dx0, color='blue', marker='o')
            else:
                plt.scatter(du0, dx0, color='red', marker='o')

            writer.writerow(np.around(sample, decimals=3))
            start0 += 1
        laneChanges += 1

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-15, 15, -50, 100])
    plt.show()


# extract_sample_3seconds()
# extract_snapshots_3seconds()
# extract_sample_multiple_snapshot()

extract_samples_modified_loss_function()
# extract_samples_modified_loss_function_relabel_as_change_in_distance()
