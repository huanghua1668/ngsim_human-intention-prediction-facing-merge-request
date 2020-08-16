import numpy as np
import csv
import matplotlib.pyplot as plt

def extract_sample():
    vehicleLength=5.
    data=np.genfromtxt('lane_changes.csv', delimiter=',')
    output=open('samples.csv', 'w')
    writer=csv.writer(output)
    for i in range(0, int(data[-1,0])+1):
        start=np.searchsorted(data[:,0], i)
        if i==int(data[-1,0]):
            end=data.shape[0]-1
        else:
            end=np.searchsorted(data[:,0], i+1)-1

        start0=start
        end0=end
        while(start0<end0 and data[start0, -5]==0.): start0+=1
        if start0==end0: continue
        if data[start0,1]>-2.: continue 
        # find lag at time shorter than 2 seconds before cross lane division
        while(start0<end0 and data[end0, -5]==0.): end0-=1
        if data[end0,1]<2.: continue 
        # lag disappear at time shorter than 2 seconds after cross lane division

        du0=data[start0, 4]-data[start0,16]
        du1=data[start0, 4]-data[start0,12]
        du2=data[start0, 4]-data[start0,8]
        dx0=data[start0, 3]-data[start0,15]
        dx1=data[start0, 3]-data[start0,11]
        dx2=data[start0, 3]-data[start0,7]
        dy0=data[start0, 2]-data[start0,14]
        dy1=data[start0, 2]-data[start0,10]
        dy2=data[start0, 2]-data[start0,6]

        if data[end0, 3]-data[end0,15] <vehicleLength: continue
        y=data[start0, -1]
        sample=[data[start0, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

        if y==1:
            plt.scatter(du0,dx0,color='blue', marker='o')
        else:
            plt.scatter(du0,dx0,color='red', marker='o')

        writer.writerow(np.around(sample, decimals=3))
        
    output.close() 
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-10, 10, -30, 70])
    plt.show()

extract_sample()

