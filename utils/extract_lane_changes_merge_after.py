# detect case when leading vehicle in target lane close window
# namely bypass from behind

import numpy as np
import csv

def extract_surrounding_vehicles(id, name, data, i, start, end, record, x0, y0, carOnly=True):
    nonCar = False
    m = data.shape[0]
    nameToInd = {'precedingIDold': 6, 'leadingID':10, 'precedingID': 14, 'followingID': 19}
    ind = nameToInd[name]
    row = np.searchsorted(data[:, 0], id)

    if (name == 'followingID' and row < m and data[row, 0] == id and
            data[row, 3] >= data[i, 3]):
        # only when/after cross lane division starts seeing
        # lag vehicle, discard
        return True, record, -1, -1, -1

    # eliminate examples involved with non-car
    if carOnly:
        v_class = data[row, 10]
        if (v_class != 2.):
            if (v_class == 1.):
                print('non car for leading vehicle at old lane at', row, ", motorcycle")
            else:
                print('non car for leading vehicle at old lane at', row, ", truck")
            i = i + 1
            nonCar = True
            return nonCar, record, -1, -1, -1

    while (row < m and data[row, 0] == id and
           data[row, 3] < data[i, 3]):
        row += 1
    if row < m and data[row, 0] == id and data[row, 3] == data[i, 3]:
        s = row
        e = row
        while (s >= 0 and data[s, 0] == id and
               row - s < i - start):
            s -= 1
        if data[s, 0] != id: s += 1
        while (e < m and data[e, 0] == id and
               e - row < end - i):
            e += 1
        if data[e, 0] != id: e -= 1
        # if (e - s < end - start):
        #     print(i, 'does not have long enough data for ' + name + ', ',
        #           id, 'len [', row - s, ',', e - row, '], ',
        #           'len of ego [', i - start, ',', end - i, ']')
        if s < row:
            for j in range(s, e + 1):
                record[j - row + i - start, ind:ind + 2] = data[j, 4:6]
                record[j - row + i - start, ind] -= x0
                record[j - row + i - start, ind + 1] -= y0
                record[j - row + i - start, ind + 2:ind + 4] = data[j, 11:13]

        return nonCar, record, s, row, e
    return True, record, -1, -1, -1

laneWidth = 3.7 * 3.2808  # 3.7m to feet
vehicleLength = 5. * 3.2808
carOnly = True
observeLength = 5
t = 80  # lane change decision(abortion here) make time

# for i-80
# dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
# dir = dir0 + '0400pm-0415pm/'
# data = np.loadtxt(dir + 'trajectories-0400-0415.txt')
# dir = dir0 + '0500pm-0515pm/'
# data = np.loadtxt(dir + 'trajectories-0500-0515.txt')
# dir = dir0 + '0515pm-0530pm/'
# data = np.loadtxt(dir + 'trajectories-0515-0530.txt')

# for i-101
dir0 = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
# dir = dir0 + '0750am-0805am/'
# data = np.loadtxt(dir + 'trajectories-0750am-0805am.txt')
# dir = dir0 + '0805am-0820am/'
# data = np.loadtxt(dir + 'trajectories-0805am-0820am.txt')
dir = dir0 + '0820am-0835am/'
data = np.loadtxt(dir + 'trajectories-0820am-0835am.txt')

f_lane_change_index = open(dir+'lane_changes_merge_after_index.csv', 'w')
f_lane_change = open(dir+'lane_changes_merge_after.csv', 'w')
writerInd = csv.writer(f_lane_change_index)
writer = csv.writer(f_lane_change)

m = data.shape[0]
print('load data successfully, data shape', data.shape)
records = []
counts = 0
i = 1
iPriorLC = -1
while (i < m):
    mergeAfter = False
    if i % 10000 == 0: print(i)
    if (data[i, 0] == data[i - 1, 0] and data[i, 13] != data[i - 1, 13]
            and data[i, 13] != 7 and data[i - 1, 13] != 7):
        # exlude merge to lane 7, which is ramp
        # same car, just changed the lane

        # eliminate examples involved with non-car
        if carOnly:
            v_class = data[i, 10]
            if (v_class != 2.):
                if v_class == 1:
                    print('non car do lane change at ', i, ", motocycle")
                else:
                    print('non car do lane change at ', i, ", truck")
                i = i + 1
                continue

        x0 = int((data[i, 4] + laneWidth / 2.) / laneWidth) * laneWidth
        i0 = i

        # judge "actual" cross point of lane division
        if data[i, 13] > data[i - 1, 13]:  # turn right
            if data[i, 4] < x0:
                while i < m and data[i, 4] < x0:
                    i += 1
            elif data[i - 1, 4] >= x0:
                while i >= 1 and data[i - 1, 4] >= x0:
                    i -= 1
        else:
            if data[i, 4] > x0:
                while i < m and data[i, 4] > x0:
                    i += 1
            elif data[i - 1, 4] <= x0:
                while i >= 1 and data[i - 1, 4] <= x0:
                    i -= 1
        if (i == m): break

        if (i0 - i > 20 or i - i0 > 20):
            # too large disparity, discard
            i = i0 + 1
            continue

        precedingID = int(data[i0, -4])
        followingID = int(data[i0, -3])
        precedingIDold = int(data[i0 - 1, -4])
        lcTime = data[i, 3]

        start = i - 1
        end = i
        if data[i0, 13] > data[i0 - 1, 13]:  # turn right
            while (start >= 0 and data[start, 0] == data[i, 0]
                   # and data[start,4]<x0 and data[start,4]>x0-laneWidth and i-start<50):
                   and data[start, 4] < x0 and data[start, 4] > x0 - laneWidth
                   and i - start < t + observeLength-1):
                start -= 1
            if data[start, 0] != data[i, 0] or data[start, 4] >= x0 or data[start, 4] <= x0 - laneWidth:
                start += 1
            while (end < m and data[end, 0] == data[i, 0]
                   and data[end, 4] > x0 and data[end, 4] < x0 + laneWidth and end - i < 50):
                end += 1
            if data[end, 0] != data[i, 0] or data[end, 4] <= x0 or data[end, 4] >= x0 + laneWidth:
                end -= 1
        else:  # turn left
            while (start >= 0 and data[start, 0] == data[i, 0]
                   # and data[start,4]>x0 and data[start,4]<x0+laneWidth and i-start<50):
                   and data[start, 4] > x0 and data[start, 4] < x0 + laneWidth
                   and i - start < t + observeLength-1):
                start -= 1
            if data[start, 0] != data[i, 0] or data[start, 4] <= x0 or data[start, 4] >= x0 + laneWidth:
                start += 1
            while (end < m and data[end, 0] == data[i, 0]
                   and data[end, 4] < x0 and data[end, 4] > x0 - laneWidth and end - i < 50):
                end += 1
            if data[end, 0] != data[i, 0] or data[end, 4] >= x0 or data[end, 4] <= x0 - laneWidth: end -= 1
        if (data[end, 4] <= x0 - laneWidth): print('Error, i, i0, start, end', i, i0, start, end)
        # if(i-start<20 or end-i<20):
        if (i - start < t + observeLength-1 or end - i < 20):
            print(i, 'short before or after lane change period', i - start,
                  end - i)
            i = i0 + 1
            continue

        if (i == iPriorLC):
            # in case go back to already tracked one
            i = i0 + 1
            continue
        iPriorLC = i

        print('i0, i-i0, i-start, end-i, xStart, x0, xEnd', i0, i - i0,
              i - start, end - i, data[start, 4], x0, data[end, 4])

        # interpolate
        t0 = data[i - 1, 3] + (x0 - data[i - 1, 4]) / (data[i, 4] - data[i - 1, 4]) * 100
        y0 = data[i - 1, 5] + ((x0 - data[i - 1, 4]) / (data[i, 4] - data[i - 1, 4])
                               * (data[i, 5] - data[i - 1, 5]))
        print('t, t0, t1', t0, data[i - 1, 3], data[i, 3])
        print('x, x0, x1', x0, data[i - 1, 4], data[i, 4])
        if ((x0 - data[i - 1, 4]) * (data[i, 4] - x0) < 0):
            print('Error, x does not locate between x0 and x1!!!!!')
        print('y, y0, y1', y0, data[i - 1, 5], data[i, 5])

        record = np.zeros((end - start + 1, 23))
        # ID, time, local_x_ego, local_y_ego, v_ego, a_ego, //5
        #          local_x_f,   local_y_f, v_f, a_f,       //9
        #          local_x_ft1,  local_y_ft1, v_ft1, a_ft1,    //13, leading
        #          local_x_ft2,  local_y_ft2, v_ft2, a_ft2, label //18, right front
        #          local_x_rt,  local_y_rt, v_rt, a_rt //22, right back
        record[:, 0] = counts

        index = [counts, i0, i - i0, i - start, end - i, 0, 0, 0, 0, 0, 0, 1]

        # for ego
        for j in range(start, end + 1):
            record[j - start, 1] = data[j, 3] - t0
            record[j - start, 2:4] = data[j, 4:6]
            record[j - start, 4:6] = data[j, 11:13]
            record[j - start, 2] -= x0
            record[j - start, 3] -= y0

        # for leading car in original lane
        if precedingIDold != 0:  # there is one
            nonCar, record, s, row, e = extract_surrounding_vehicles(precedingIDold, 'precedingIDold', data, i, start,
                                                                     end, record, x0, y0)
            index[5] = row - s
            index[6] = e - row
            if nonCar:
                i = i + 1
                continue

        # for leading car in target lane
        s = -1
        e = -1
        if precedingID != 0:  # there is one
            nonCar, record, s, row, e = extract_surrounding_vehicles(precedingID, 'precedingID', data, i, start,
                                                                     end, record, x0, y0)
            index[7] = row - s
            index[8] = e - row
            if nonCar:
                i = i + 1
                continue
            if (row - s < observeLength):
                continue
        else:
            print('no preceding for row ', i, ' for old lane')
            continue

        if (s > 0 and e > 0 and data[i - row + s + observeLength-1, 5] > data[s + observeLength-1, 5] + vehicleLength
                and data[i - row + e, 5] < data[e, 5] - vehicleLength):
            # at time s+observationLength, there is indeed a window, and precedingID is trying to close it
            print('merge after!')
            mergeAfter = True
        else:
            continue

        # now find car in front of leading car in target lane
        # ego choose to merge after precedingID
        row = np.searchsorted(data[:, 0], precedingID)

        # now data[row-1]<precedingID<=data[row]
        while (row < m and data[row, 0] == precedingID and
               data[row, 3] < data[i, 3]):
            row += 1
        if (row < m and data[row, 0] == precedingID and data[row, 3] == data[i, 3]):
            leadingID = int(data[row, -4])
            # ego choose not to merge after leadingID
            if leadingID != 0:
                nonCar, record, s, row, e = extract_surrounding_vehicles(leadingID, 'leadingID', data, i, start,
                                                                         end, record, x0, y0)
                index[9] = row - s
                index[10] = e - row
                index[11] = 1
                if nonCar:
                    i = i + 1
                    continue
            else:
                print('no front most vehicle at target lane for row ', i)

        # for following car in target lane
        if followingID != 0:  # there is one
            nonCar, record, s, row, e = extract_surrounding_vehicles(followingID, 'followingID', data, i, start,
                                                                     end, record, x0, y0)
            if nonCar:
                i = i + 1
                continue

        print('lane changed at row', i, counts, 'th change',
              'from lane ', data[i - 1, -5], ' change to ', data[i, -5])
        writerInd.writerow(index)
        records.append(record)
        counts += 1
        i = i0 + 1
    i += 1
records = np.vstack(records)

# normalize by minuse min values
records[:, 1] /= 1000  # time: to seconds
records[:, 2:18] /= 3.2808  # x: foot to meter
records[:, 19:23] /= 3.2808  # x: foot to meter

for i in range(records.shape[0]):
    writer.writerow(np.around(records[i], decimals=3))
f_lane_change_index.close()
f_lane_change.close()
