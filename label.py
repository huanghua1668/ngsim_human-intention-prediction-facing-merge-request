import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

comfortableDecRate=-3.4
def score(accelerations):
    minDec=np.amin(accelerations)
    if minDec>comfortableDecRate:
        return 1
    temp=np.sum(accelerations<=comfortableDecRate)
    if temp>=10:
        return 0
    return 1

def label():
    vehicleLength=5.
    data=np.genfromtxt('lane_changes.csv', delimiter=',')
    f_lane_change=open('lane_changes_begin_end_labeled.csv', 'w')
    writer=csv.writer(f_lane_change)
    u0=[]
    x0=[]
    y0=[]
    du=[]
    dx=[]
    dy=[]
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    for i in range(0, int(data[-1,0])+1):
        start=np.searchsorted(data[:,0], i)
        if i==int(data[-1,0]):
            end=data.shape[0]-1
        else:
            end=np.searchsorted(data[:,0], i+1)-1
        #if abs(data[end,2]-data[start,2])<1.85:
        #    plt.plot(data[start:end+1, 3]-data[start,3],data[start:end+1,2])
        #plt.plot(data[start:end+1, 1]-data[start,1],
        #        data[start:end+1,2]-data[start:end+1,-4], label='$\Delta y$')
        start0=start
        end0=end
        while(start0<end0 and data[start0, -4]==0.): start0+=1
        if start0==end0: continue
        while(start0<end0 and data[end0, -4]==0.): end0-=1
        y=score(data[start0:end0+1, -1])
        print('arrow', i, start0, end0, y)
        du0=data[start0, 4]-data[start0,-2]
        dx0=data[start0, 3]-data[start0,-3]
        dy0=data[start0, 2]-data[start0,-4]
        du1=data[end0, 4]-  data[end0,-2]
        dx1=data[end0, 3]-  data[end0,-3]
        dy1=data[end0, 2]-  data[end0,-4]
        if dx1<vehicleLength: continue
        print(du1-du0, dx1-dx0, dy1-dy0)
        record=[du0, dx0, dy0, du1, dx1, dy1, y]
        writer.writerow(np.around(record, decimals=3))
        u0.append(du0)
        x0.append(dx0)
        y0.append(dy0)
        du.append(du1-du0)
        dx.append(dx1-dx0)
        dy.append(dy1-dy0)
        #plt.arrow(du0, dx0, du1-du0, dx1-dx0)
        ax.scatter(du0,dx0,dy0,color='black', marker='o')
        ax.scatter(du1,dx1,dy1,color='black', marker='x')
        #if i>100: break
    ax.quiver(u0, x0, y0, du, dx, dy, arrow_length_ratio=0.005)
    ax.set_xlabel('$\Delta v$')
    ax.set_ylabel('$\Delta x$')
    ax.set_zlabel('$\Delta y$')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-50, 100)
    ax.set_zlim(-5, 5)
    #plt.axis([-15, 15, -50, 100])
    #plt.axis([-10, 10, -30, 70, -5, 5])
    
    plt.show()
    f_lane_change.close()

#label()
