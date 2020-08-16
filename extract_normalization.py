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

laneWidth=3.7*3.2808 # 3.7m to feet

f_lane_change_index=open('lane_change_row_index.csv', 'w')
f_lane_change=open('lane_changes.csv', 'w')
writerInd=csv.writer(f_lane_change_index)
writer=csv.writer(f_lane_change)

data=np.loadtxt('trajectories-0400-0415.txt')
m=data.shape[0]
print('load data successfully, data shape', data.shape)
records=[]
counts=0
print('m=', m)
i=1
iPriorLC=-1
while(i< m):
    if i%10000==0: print(i)
    if (data[i,0]==data[i-1,0] and data[i,13]!=data[i-1,13]
            and data[i,13]!=7 and data[i-1,13]!=7):
        # exlude merge to lane 7, which is ramp
        # same car, just changed the lane
        x0=int((data[i,4]+laneWidth/2.)/laneWidth)*laneWidth
        i0=i

        # judge "actual" cross point of lane division
        precedingIDold=int(data[i0-1,-4])
        if data[i,13]>data[i-1,13]: #turn right
            if data[i, 4]<x0:
                while i<m and data[i, 4]<x0:
                    i+=1
            elif data[i-1, 4]>=x0:
                while i>=1 and data[i-1, 4]>=x0:
                    i-=1
        else:
            if data[i,4]>x0:
                while i<m and data[i, 4]>x0:
                    i+=1
            elif data[i-1,4]<=x0:
                while i>=1 and data[i-1, 4]<=x0:
                    i-=1
        if(i==m): break

        if(i0-i>20 or i-i0>20):
            # too large disparity, discard
            i=i0+1
            continue

        # better use original i0
        precedingID=int(data[i0,-4])
        followingID=int(data[i0,-3])

        start=i-1
        end=i
        if data[i0,13]>data[i0-1,13]: #turn right
            while (start>=0 and data[start,0]==data[i,0] 
                   and data[start,4]<x0 and data[start,4]>x0-laneWidth and i-start<50):
                start-=1
            if data[start,0]!=data[i,0] or data[start,4]>=x0 or data[start,4]<=x0-laneWidth :
                start+=1
            while (end<m and data[end,0]==data[i,0] 
                   and data[end,4]>x0 and data[end,4]<x0+laneWidth and  end-i<50):
                end+=1
            if data[end,0]!=data[i,0] or data[end,4]<=x0 or data[end,4]>=x0+laneWidth:
                end-=1
        else:#turn left
            while (start>=0 and data[start,0]==data[i,0]
                   and data[start,4]>x0 and data[start,4]<x0+laneWidth and i-start<50):
                start-=1
            if data[start,0]!=data[i,0] or data[start,4]<=x0 or data[start,4]>=x0+laneWidth: 
                start+=1
            while (end<m and data[end,0]==data[i,0] 
                   and data[end,4]<x0 and data[end,4]>x0-laneWidth and  end-i<50):
                end+=1
            if data[end,0]!=data[i,0] or data[end,4]>=x0 or data[end,4]<=x0-laneWidth : end-=1
        if(data[end,4]<=x0-laneWidth): print('Error, i, i0, start, end', i, i0, start, end)
        if(i-start<10 or end-i<10):
            print(i, 'short before or after lane change period', i-start,
                    end-i)
            i=i0+1
            continue

        if(i==iPriorLC):
            # in case go back to already tracked one
            i=i0+1
            continue
        iPriorLC=i
        
        maxdiff=np.amax(np.absolute(data[start+1:end+1,4]-data[start:end,4]))
        #if maxdiff>laneWidth:
            #print("large lateral velocity at ", i0, ', with maximum ',maxdiff)

        print('i0, i-i0, i-start, end-i, xStart, x0, xEnd', i0, i-i0,
                i-start, end-i, data[start,4], x0, data[end,4])
        print('lane changed at row', i, counts, 'th change',
                'from lane ', data[i-1,-5], ' change to ', data[i,-5])

        #interpolate
        t0=data[i-1,3]+ (x0-data[i-1,4])/(data[i,4]-data[i-1,4])*100
        y0=data[i-1,5]+((x0-data[i-1,4])/(data[i,4]-data[i-1,4])
                                        *(data[i,5]-data[i-1,5]))
        print('t, t0, t1', t0, data[i-1,3], data[i,3])
        print('x, x0, x1', x0, data[i-1,4], data[i,4])
        if((x0-data[i-1,4])*(data[i,4]-x0)<0): print('Error, x does not locate between x0 and x1!!!!!')
        print('y, y0, y1', y0, data[i-1,5], data[i,5])

        record=np.zeros((end-start+1, 19))
        #ID, time, local_x_ego, local_y_ego, v_ego, a_ego, //5 
        #          local_x_f,   local_y_f, v_f, a_f,       //9
        #          local_x_ft,  local_y_ft, v_ft, a_ft,    //13
        #          local_x_rt,  local_y_rt, v_rt, a_rt, label //18
        record[:,0]=counts

        #index=[counts, i, data[i,0], data[i-1,-5], data[i,-5],
        #        i-start, end-i]
        index=[counts, i0, i-i0, i-start, end-i, 0,0,0,0,0,0,1]

        #for ego
        for j in range(start, end+1):
            record[j-start,1]=data[j,3]-t0
            record[j-start,2:4]=data[j,4:6]
            record[j-start,4:6]=data[j,11:13]
            record[j-start,2]-=x0
            record[j-start,3]-=y0

        #for leading car in original lane
        if precedingIDold!=0: # there is one
            row=np.searchsorted(data[:,0], precedingIDold)
            #now data[row-1]<precedingIDold<=data[row]
            while (row<m and data[row,0]==precedingIDold and
                              data[row,3]<data[i,3]):
                row+=1
            if row<m and data[row,0]==precedingIDold and data[row,3]==data[i,3]:
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
                    i=i0+1
                    continue
                if (e-s<end-start):
                    print(i, 'does not have long enough data for leading car in old lane ', 
                            precedingIDold, 'len of old leading car [', row-s,',',e-row,'], ', 
                            'len of ego [', i-start, ',', end-i,']')
                if s<row:
                    for j in range(s, e+1):
                        record[j-row+i-start,6:8]=data[j,4:6]
                        record[j-row+i-start,6]-=x0
                        record[j-row+i-start,7]-=y0
                        record[j-row+i-start,8:10]=data[j,11:13]

                index[5]=row-s
                index[6]=e-row
        else:
            print('no preceding for row ', i, ' for old lane')

        #for leading car in target lane
        if precedingID!=0: # there is one
            row=np.searchsorted(data[:,0], precedingID)
            while (row<m and data[row,0]==precedingID and
                              data[row,3]<data[i,3]):
                row+=1
            if (row<m and data[row,0]==precedingID and data[row,3]==data[i,3]):
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
                    i=i0+1
                    continue
                if (e-s<end-start):
                    print(i, 'does not have long enough data for leading car ', 
                            precedingID, 'len of leading car [ ', row-s,',',e-row,'],', 
                            'len of ego [ ', i-start, ',', end-i,']')
                if s<row:
                    for j in range(s, e+1):
                        record[j-row+i-start,10:12]=data[j,4:6]
                        record[j-row+i-start,10]-=x0
                        record[j-row+i-start,11]-=y0
                        record[j-row+i-start,12:14]=data[j,11:13]
                index[7]=row-s
                index[8]=e-row
        else:
            print('no preceding for row ', i)
        y=1
        #for following car in target lane
        if followingID!=0: # there is one
            row=np.searchsorted(data[:,0], followingID)
            if (row<m and data[row,0]==followingID and
                              data[row,3]>=data[i,3]):
                # only when/after cross lane division starts seeing
                # lag vehicle, discard 
                continue
            while (row<m and data[row,0]==followingID and
                              data[row,3]<data[i,3]):
                row+=1
            if (row<m and data[row,0]==followingID and data[row,3]==data[i,3]):
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
                    i=i0+1
                    continue
                if (e-s<end-start):
                    print(i, 'does not have long enough data for following car ', 
                            followingID, 'len of following car [ ', row-s,',',e-row,'],', 
                            'len of ego [ ',i-start,',', end-i,']')

                if e>s:
                    y=score(data[s:e+1, 12]/3.2808)

                for j in range(s, e+1):
                    record[j-row+i-start,14:16]=data[j,4:6]
                    record[j-row+i-start,14]-=x0
                    record[j-row+i-start,15]-=y0
                    record[j-row+i-start,16:18]=data[j,11:13]
                index[9]=row-s
                index[10]=e-row
                index[11]=y
        else:
            print('no following for row ', i)

        for j in range(start, end+1):
            record[j-start,-1]=y

        writerInd.writerow(index)

        records.append(record)
        counts+=1
        i=i0+1
        #if counts>=2: break
    i+=1
records=np.vstack(records)

#normalize by minuse min values
records[:,1]/=1000 #time: to seconds
records[:, 2:15:4]/=3.2808 #x: foot to meter
records[:, 3:16:4]/=3.2808  #y: foot to meter
records[:, 4:17:4]/=3.2808 #v: feet/second to meter/second
records[:, 5:18:4]/=3.2808 #a: feet/second^2 to meter/second^2

for i in range(records.shape[0]):
    writer.writerow(np.around(records[i], decimals=3))
f_lane_change_index.close()
f_lane_change.close()

