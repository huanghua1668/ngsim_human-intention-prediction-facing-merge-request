import numpy as np
import csv

f_lane_change_index=open('lane_change_row_index.csv', 'w')
f_lane_change=open('lane_changes.csv', 'w')
writerInd=csv.writer(f_lane_change_index)
writer=csv.writer(f_lane_change)

data=np.loadtxt('trajectories-0400-0415.txt')
m=data.shape[0]
print('load data successfully, data shape', data.shape)
records=[]
counts=0
timeMin=data[0, 3]
xMin=data[0,6]
yMin=data[0,7]
print('m=', m)
for i in range(1, m):
    if i%10000==0: print(i)
    if (data[i,0]==data[i-1,0] and data[i,13]!=data[i-1,13]
            and data[i,13]!=7):
        # exlude merge to lane 7, which is ramp
        # same car, just changed the lane
        print('lane changed at row', i, counts, 'th change',
                'from lane ', data[i-1,-5], ' change to ', data[i,-5])

        precedingID=int(data[i,-4])
        followingID=int(data[i,-3])
        precedingIDold=int(data[i-1,-4])
        lcTime=data[i,3]

        timeMin=min(timeMin, data[i,3])
        xMin=min(xMin, data[i,6])
        yMin=min(yMin, data[i,7])

        start=i
        while (start>=0 and data[start,0]==data[i,0] and
                              i-start<50):
            start-=1
        if data[start,0]!=data[i,0]: start+=1
        #end=min(i+51, m)
        end=i
        while (end<m and data[end,0]==data[i,0] and
                              end-i<50):
            end+=1
        if data[end,0]!=data[i,0]: end-=1
        record=np.zeros((end-start+1, 18))
        #ID, time, local_x_ego, local_y_ego, v_ego, a_ego, 
        #          local_x_f,   local_y_f, v_f, a_f, 
        #          local_x_ft,  local_y_ft, v_ft, a_ft, 
        #          local_x_rt,  local_y_rt, v_rt, a_rt, 
        record[:,0]=counts

        index=[counts, i, data[i,0], data[i-1,-5], data[i,-5],
                i-start, end-i]
        writerInd.writerow(index)

        #for ego
        for j in range(start, end+1):
            record[j-start,1]=data[j,3]
            record[j-start,2:4]=data[j,4:6]
            record[j-start,4:6]=data[j,11:13]

        #for leading car in original lane
        if precedingIDold!=0: # there is one
            row=np.searchsorted(data[:,0], precedingIDold)
            #now data[row-1]<precedingIDold<=data[row]
            while (row<m and data[row,0]==precedingIDold and
                              data[row,3]!=data[i,3]):
                row+=1
            s=row
            e=row
            while (s>=0 and data[s,0]==precedingIDold and
                              row-s<i-start):
                s-=1
            if data[s,0]!=precedingIDold: s+=1
            while (e<m and data[e,0]==precedingIDold and
                              e-row<end-i):
                e+=1
            if data[e,0]!=precedingIDold: e-=1
            if row==m:
                print(i, 'error! can not find leading car in original lane')
                continue
            if (e-s<end-start):
                print(i, 'does not have long enough data for leading car in old lane ', 
                        precedingIDold, 'len of old leading car [', row-s,',',e-row,'], ', 
                        'len of ego [', i-start, ',', end-i,']')
            for j in range(s, e+1):
                record[j-row+i-start,6:8]=data[j,4:6]
                record[j-row+i-start,8:10]=data[j,11:13]
        else:
            print('no preceding for row ', i, ' for old lane')

        #for leading car in target lane
        if precedingID!=0: # there is one
            row=np.searchsorted(data[:,0], precedingID)
            while (row<m and data[row,0]==precedingID and
                              data[row,3]!=data[i,3]):
                row+=1
            s=row
            e=row
            while (s>=0 and data[s,0]==precedingID and
                              row-s<i-start):
                s-=1
            if data[s,0]!=precedingID: s+=1
            while (e<m and data[e,0]==precedingID and
                              e-row<end-i):
                e+=1
            if data[e,0]!=precedingID: e-=1
            if row==m:
                print(i, 'error! can not find leading car')
                continue
            if (e-s<end-start):
                print(i, 'does not have long enough data for leading car ', 
                        precedingID, 'len of leading car [ ', row-s,',',e-row,'],', 
                        'len of ego [ ', i-start, ',', end-i,']')
            for j in range(s, e+1):
                record[j-row+i-start,10:12]=data[j,4:6]
                record[j-row+i-start,12:14]=data[j,11:13]
        else:
            print('no preceding for row ', i)

        #for following car in target lane
        if followingID!=0: # there is one
            row=np.searchsorted(data[:,0], followingID)
            while (row<m and data[row,0]==followingID and
                              data[row,3]!=data[i,3]):
                row+=1
            s=row
            e=row
            while (s>=0 and data[s,0]==followingID and
                              row-s<i-start):
                s-=1
            if data[s,0]!=followingID: s+=1
            while (e<m and data[e,0]==followingID and
                              e-row<end-i):
                e+=1
            if data[e,0]!=followingID: e-=1
            if row==m:
                print(i, 'error! can not find following car')
                continue
            if (e-s<end-start):
                print(i, 'does not have long enough data for following car ', 
                        followingID, 'len of following car [ ', row-s,',',e-row,'],', 
                        'len of ego [ ',i-start,',', end-i,']')
            for j in range(s, e+1):
                record[j-row+i-start,14:16]=data[j,4:6]
                record[j-row+i-start,16:18]=data[j,11:13]
        else:
            print('no following for row ', i)

        records.append(record)
        counts+=1
records=np.vstack(records)

#normalize by minuse min values
records[:,1]-=timeMin
records[:,1]/=100 #time: to seconds
#records[:, 2:15:4]-=xMin
records[:, 2:15:4]/=3.2808 #x: foot to meter
#records[:, 3:16:4]-=yMin
records[:, 3:16:4]/=3.2808  #y: foot to meter
records[:, 4:17:4]/=3.2808 #v: feet/second to meter/second
records[:, 5:18:4]/=3.2808 #a: feet/second^2 to meter/second^2

for i in range(records.shape[0]):
    writer.writerow(np.around(records[i], decimals=3))
f_lane_change_index.close()
f_lane_change.close()

