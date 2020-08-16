import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import ImageMagickFileWriter

lengthCar=5.
widthCar=2.1
widthLane=3.7

data=np.genfromtxt('lane_changes.csv', delimiter=',')
ind=5
start=np.searchsorted(data[:,0], ind)
end=np.searchsorted(data[:,0], ind+1)-1
data=data[start:end+1]

x_e=data[:,3]-data[0,3]
y_e=data[:,2]
x_f=data[:,7]-data[0,3]
y_f=data[:,6]
x_ft=data[:,11]-data[0,3]
y_ft=data[:,10]
x_rt=data[:,15]-data[0,3]
y_rt=data[:,14]

fig = plt.figure()
plt.axis('scaled')
ax = fig.add_subplot(111)
ax.set_xlim(-25, 100)
ax.set_ylim(14, 25)
plt.gca().invert_yaxis()

if y_e[0]>y_e[-1]:
    laneLowerBound=(int(y_e[0]/widthLane)+1)*widthLane
    laneUpperBound=int(y_e[-1]/widthLane)*widthLane
else:
    laneUpperBound=int(y_e[0]/widthLane)*widthLane
    laneLowerBound=(int(y_e[-1]/widthLane)+1)*widthLane
laneDivision=laneLowerBound-widthLane
print('ub', laneUpperBound, 'lb', laneLowerBound, 'ld', laneDivision)

patch_e = patches.Rectangle((x_e[0]-lengthCar/2., y_e[0]-widthCar/2.),
                            lengthCar, widthCar, fc='r')
patch_f = patches.Rectangle((x_f[0]-lengthCar/2., y_f[0]-widthCar/2.),
                            lengthCar, widthCar, fc='b')
patch_ft = patches.Rectangle((x_ft[0]-lengthCar/2., y_ft[0]-widthCar/2.),
                            lengthCar, widthCar, fc='b')
patch_rt = patches.Rectangle((x_rt[0]-lengthCar/2., y_rt[0]-widthCar/2.),
                            lengthCar, widthCar, fc='b')

def init():
    ax.add_patch(patch_e)
    ax.plot(np.arange(-50, 151, 1), laneLowerBound*np.ones(201))
    ax.plot(np.arange(-50, 151, 1), laneUpperBound*np.ones(201))
    ax.plot(np.arange(-50, 151, 1), laneDivision*np.ones(201), linestyle='--')
    return patch_e,

def animate(i):
    patch_e.set_xy([x_e[i], y_e[i]])
    if y_f[i]!=0:
        patch_f.set_xy([x_f[i]-lengthCar/2., y_f[i]-widthCar/2.])
        ax.add_patch(patch_f)
    if y_ft[i]!=0:
        patch_ft.set_xy([x_ft[i]-lengthCar/2., y_ft[i]-widthCar/2.])
        ax.add_patch(patch_ft)
    if y_rt[i]!=0:
        patch_rt.set_xy([x_rt[i]-lengthCar/2., y_rt[i]-widthCar/2.])
        ax.add_patch(patch_rt)

    return patch_e,patch_f, patch_ft,patch_rt

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=x_e.shape[0],
                               interval=100,
                               blit=True)
writer=ImageMagickFileWriter(fps=15, bitrate=3000)
anim.save('lc_id5.gif', writer=writer)
plt.show()
