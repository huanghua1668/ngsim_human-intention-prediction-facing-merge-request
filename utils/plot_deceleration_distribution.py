import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

vehicleLength=5.
comfortableDecRate=-3.4
def plot_deceleration():
    vehicleLength=5.
    records=[]

    f=open('decelerations.csv', 'w')
    writer=csv.writer(f)
    files=['0400pm-0415pm/lane_changes.csv',
           '0500pm-0515pm/lane_changes.csv',
           '0515pm-0530pm/lane_changes.csv' ]
    for f in files:
        data=np.genfromtxt(f, delimiter=',')
        #data[:,1]/=10.
        for i in range(data.shape[0]):
            if data[i,-1]==1 or data[i,-2]>=comfortableDecRate:
                continue
            record=np.zeros(4)
            record[0]=data[i,1]
            record[1]=data[i,2]-data[i,-5]
            record[2]=data[i,3]-data[i,-4]
            if data[i,-3]-data[i,4]==0:
                record[3]=15*60.
            else:
                record[3]=(data[i,3]-data[i,-4]-vehicleLength)/(data[i,-3]-data[i,4])
            records.append(record)
            writer.writerow(record)

    records=np.vstack(records)
    print(records.shape[0], 'harsh brakes for adversarial lane change')

    figInd=0
    plt.figure(figInd)
    plt.hist(records[:,0])
    plt.xlabel('$\Delta t$')

    figInd+=1
    plt.figure(figInd)
    plt.hist(records[:,1])
    plt.xlabel('$\Delta y$')

    figInd+=1
    plt.figure(figInd)
    plt.hist(records[:,2], range=(0, 100))
    plt.xlabel('$\Delta x$')

    figInd+=1
    plt.figure(figInd)
    plt.hist(records[:,3], range=(-30,30))
    plt.xlabel('$Time To Collision$')

    figInd+=1
    plt.figure(figInd)
    r=np.array([[-4,4],[-20,20]])
    plt.hist2d(records[:,0], records[:,3], range=r)
    plt.xlabel('$\Delta t$')
    plt.ylabel('$Time To Collision$')


    #if i==400: break
    #plt.legend()
    #plt.axis([-15, 15, -50, 100])
    
    plt.show()

plot_deceleration()
