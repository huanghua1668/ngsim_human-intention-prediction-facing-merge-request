#####################################################################
# here at index i, many cases it's not just crossed lane division,
# slight differences, some cases actually need few more steps to cross
# lane division based on 3.7m lane width
# --> so add few more steps and judge lane change when car cross 3.7*lane
# counts
######################################################################
import numpy as np
import csv
from label import score


def extract_surrounding_vehicles(id, name, data, i, start, end, record, x0, y0, carOnly=True):
    nonCar = False
    m = data.shape[0]
    nameToInd = {'precedingIDold': 6, 'precedingID': 10, 'followingID': 14}
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

        if e > s and name == 'followingID':
            y = score(data[max(s, row - (i - t)):e + 1, 12] / 3.2808)
            for j in range(start, end + 1):
                record[j - start, -1] = y
        return nonCar, record, s, row, e
    return True, record, -1, -1, -1


laneWidth = 3.7 * 3.2808  # 3.7m to feet
carOnly = True
maxTurnTime = 3.
minTurnTime = 1.5  # begin turn no later than minTurnTime seconds before cross lane division
vTurn = 0.7 * 3.2808 / 10
observationLength = 5
shortestDuration = 15

dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
# dir = dir0 + '0400pm-0415pm/'
# data = np.loadtxt(dir + 'trajectories-0400-0415.txt')
# dir = dir0 + '0500pm-0515pm/'
# data = np.loadtxt(dir + 'trajectories-0500-0515.txt')
dir = dir0 + '0515pm-0530pm/'
data = np.loadtxt(dir + 'trajectories-0515-0530.txt')

f_lane_change_index = open(dir + 'lane_change_row_index.csv', 'w')
f_lane_change = open(dir + 'lane_changes.csv', 'w')
writerInd = csv.writer(f_lane_change_index)
writer = csv.writer(f_lane_change)

m = data.shape[0]
print('load data successfully, data shape', data.shape)
records = []
counts = 0
print('m=', m)
i = 1
iPriorLC = -1
while (i < m):
    if i % 10000 == 0: print(i)
    if (data[i, 0] == data[i - 1, 0] and data[i, 13] != data[i - 1, 13]
            and data[i, 13] != 7 and data[i - 1, 13] != 7):
        # exlude merge to lane 7, which is ramp
        # same car, just changed the lane

        # eliminate examples involved with non-car
        if carOnly:
            v_class = data[i, 10]
            if (v_class != 2.):
                # if v_class == 1:
                #     print('non car do lane change at ', i, ", motocycle")
                # else:
                #     print('non car do lane change at ', i, ", truck")
                i = i + 1
                continue

        x0 = int((data[i, 4] + laneWidth / 2.) / laneWidth) * laneWidth
        i0 = i

        # judge "actual" cross point of lane division
        precedingIDold = int(data[i0 - 1, -4])
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

        # better use original i0
        precedingID = int(data[i0, -4])
        followingID = int(data[i0, -3])

        t = i - 1  # t for time to begin the lateral move
        if data[i0, 13] > data[i0 - 1, 13]:  # turn right
            while (t >= 0 and data[t, 0] == data[i, 0]
                   and data[t, 4] < x0 and data[t, 4] > x0 - laneWidth and i - t <= int(maxTurnTime / 0.1)):
                t -= 1
            if not (t >= 0 and data[t, 0] == data[i, 0]
                    and data[t, 4] < x0 and data[t, 4] > x0 - laneWidth and i - t <= int(maxTurnTime / 0.1)):
                t += 1
            turnStarted = False
            while (t < i - int(minTurnTime / 0.1) and (not turnStarted)):
                if data[t + 1, 4] - data[t, 4] >= vTurn:
                    turnStarted = True;
                else:
                    t += 1
        else:
            while (t >= 0 and data[t, 0] == data[i, 0]
                   and data[t, 4] > x0 and data[t, 4] < x0 + laneWidth and i - t <= int(maxTurnTime / 0.1)):
                t -= 1
            if not (t >= 0 and data[t, 0] == data[i, 0]
                    and data[t, 4] > x0 and data[t, 4] < x0 + laneWidth and i - t <= int(maxTurnTime / 0.1)):
                t += 1
            turnStarted = False
            while (t < i - int(minTurnTime / 0.1) and (not turnStarted)):
                if data[t + 1, 4] - data[t, 4] <= -vTurn:
                    turnStarted = True;
                else:
                    t += 1
        if (i - t < int(minTurnTime / 0.1)):
            print('no turn velocity larger then 0.7 at row', i0)
            i = i0 + 1
            continue

        start = t - 1
        end = i + 1
        if data[i0, 13] > data[i0 - 1, 13]:  # turn right
            while (start >= 0 and data[start, 0] == data[i, 0]
                   and data[start, 4] < x0 and data[start, 4] > x0 - laneWidth and t - start < observationLength - 1):
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
                   and data[start, 4] > x0 and data[start, 4] < x0 + laneWidth and t - start < observationLength - 1):
                start -= 1
            if data[start, 0] != data[i, 0] or data[start, 4] <= x0 or data[start, 4] >= x0 + laneWidth:
                start += 1
            while (end < m and data[end, 0] == data[i, 0]
                   and data[end, 4] < x0 and data[end, 4] > x0 - laneWidth and end - i < 50):
                end += 1
            if data[end, 0] != data[i, 0] or data[end, 4] >= x0 or data[end, 4] <= x0 - laneWidth:
                end -= 1
        if (data[end, 4] <= x0 - laneWidth): print('Error, i, i0, start, end', i, i0, start, end)
        if (i - start < shortestDuration or end - i < shortestDuration):
            print(i, 'short before or after lane change period for ego', i - start,
                  end - i)
            i = i0 + 1
            continue

        if (i == iPriorLC):
            # in case go back to already tracked one
            i = i0 + 1
            continue
        iPriorLC = i

        # interpolate
        t0 = data[i - 1, 3] + (x0 - data[i - 1, 4]) / (data[i, 4] - data[i - 1, 4]) * 100
        y0 = data[i - 1, 5] + ((x0 - data[i - 1, 4]) / (data[i, 4] - data[i - 1, 4])
                               * (data[i, 5] - data[i - 1, 5]))
        # print('t, t0, t1', t0, data[i - 1, 3], data[i, 3])
        # print('x, x0, x1', x0, data[i - 1, 4], data[i, 4])
        if ((x0 - data[i - 1, 4]) * (data[i, 4] - x0) < 0): print('Error, x does not locate between x0 and x1!!!!!')
        # print('y, y0, y1', y0, data[i - 1, 5], data[i, 5])

        record = np.zeros((end - start + 1, 19))
        # ID, time, local_x_ego, local_y_ego, v_ego, a_ego, //5
        #          local_x_f,   local_y_f, v_f, a_f,       //9
        #          local_x_ft,  local_y_ft, v_ft, a_ft,    //13
        #          local_x_rt,  local_y_rt, v_rt, a_rt, label //18
        record[:, 0] = counts

        # index=[counts, i, data[i,0], data[i-1,-5], data[i,-5],
        #        i-start, end-i]
        index = [counts, i0, i - i0, i - start, end - i, 0, 0, 0, 0, 0, 0, 1, (t - start), (i - t), (end - i)]

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
        if precedingID != 0:  # there is one
            nonCar, record, s, row, e = extract_surrounding_vehicles(precedingID, 'precedingID', data, i, start, end,
                                                                 record, x0, y0)
            index[7] = row - s
            index[8] = e - row
            if nonCar:
                i = i + 1
                continue
        # for following car in target lane
        if followingID != 0:  # there is one
            nonCar, record, s, row, e = extract_surrounding_vehicles(followingID, 'followingID', data, i, start, end,
                                                                 record, x0, y0)
            index[9] = row - s
            index[10] = e - row
            index[11] = record[0, -1]
            if nonCar:
                i = i + 1
                continue
        else:
            i = i + 1
            continue

        writerInd.writerow(index)
        records.append(record)
        counts += 1
        i = i0 + 1
        print('i0, i-i0, i-start, end-i, xStart, x0, xEnd', i0, i - i0,
              i - start, end - i, data[start, 4], x0, data[end, 4])
        print('lane changed at row', i, counts, 'th change',
              'from lane ', data[i - 1, -5], ' change to ', data[i, -5])

        # if counts>=2: break
    i += 1
records = np.vstack(records)

# normalize by minuse min values
records[:, 1] /= 1000  # time: to seconds
records[:, 2:-1] /= 3.2808  # x: foot to meter

for i in range(records.shape[0]):
    writer.writerow(np.around(records[i], decimals=3))
f_lane_change_index.close()
f_lane_change.close()
