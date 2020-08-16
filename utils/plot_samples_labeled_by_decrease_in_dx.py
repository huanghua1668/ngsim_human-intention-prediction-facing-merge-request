#####################################################################
# here at index i, many cases it's not just crossed lane division,
# slight differences, some cases actually need few more steps to cross
# lane division based on 3.7m lane width
# --> so add few more steps and judge lane change when car cross 3.7*lane
# counts

# 04-26-2020
# modify the code to extract the samples labeled by decrease in dx
######################################################################
import numpy as np
import csv
from label import score
import matplotlib.pyplot as plt

records0=np.genfromtxt('0400pm-0415pm/samples_relabeled_by_decrease_in_dx.csv', delimiter=',')
records1=np.genfromtxt('0500pm-0515pm/samples_relabeled_by_decrease_in_dx.csv', delimiter=',')
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

