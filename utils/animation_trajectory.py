from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import ImageMagickFileWriter

lengthCar = 5.
widthCar = 2.1
widthLane = 3.7


def makeMovie(data, ind, dir):
    start = np.searchsorted(data[:, 0], ind)
    end = np.searchsorted(data[:, 0], ind + 1) - 1
    # if data[start,-1]==1: return
    data = data[start:end + 1]

    x_e = data[:, 3]
    y_e = data[:, 2]
    x_f = data[:, 7]
    y_f = data[:, 6]
    x_ft = data[:, 11]
    y_ft = data[:, 10]
    x_rt = data[:, 15]
    y_rt = data[:, 14]

    xMin = min(x_e)
    xMax = max(x_e)
    xMin = int(min(min(min(x_e), min(x_f)), min(min(x_ft), min(x_rt)))) - 5
    xMax = int(max(max(max(x_e), max(x_f)), max(max(x_ft), max(x_rt)))) + 5

    laneLowerBound = widthLane
    laneUpperBound = -widthLane
    # for ramp
    laneDivision = laneUpperBound + widthLane

    fig = plt.figure()
    plt.axis('scaled')
    ax = fig.add_subplot(111)
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(laneUpperBound - 2, laneLowerBound + 2)
    plt.gca().invert_yaxis()

    # (x,y) bottom and left corner coordinates
    patch_e = patches.Rectangle((x_e[0] - lengthCar / 2., y_e[0] - widthCar / 2.),
                                lengthCar, widthCar, fc='r')
    patch_f = patches.Rectangle((x_f[0] - lengthCar / 2., y_f[0] - widthCar / 2.),
                                lengthCar, widthCar, fc='b')
    patch_ft = patches.Rectangle((x_ft[0] - lengthCar / 2., y_ft[0] - widthCar / 2.),
                                 lengthCar, widthCar, fc='b')
    patch_rt = patches.Rectangle((x_rt[0] - lengthCar / 2., y_rt[0] - widthCar / 2.),
                                 lengthCar, widthCar, fc='b')

    def init():
        ax.add_patch(patch_e)
        ax.plot(np.arange(xMin, xMax, 1), laneLowerBound * np.ones(xMax - xMin))
        ax.plot(np.arange(xMin, xMax, 1), laneUpperBound * np.ones(xMax - xMin))
        ax.plot(np.arange(xMin, xMax, 1), laneDivision * np.ones(xMax - xMin), linestyle='--')
        return patch_e,

    def animate(i):
        patch_e.set_xy([x_e[i] - lengthCar / 2., y_e[i] - widthCar / 2.])

        if y_f[i] == 0.:
            # front obstacle does not exist, -100 to remove it out the animation y limit
            patch_f.set_xy([x_f[i] - lengthCar / 2., y_f[i] - widthCar / 2. - 100.])
        else:
            patch_f.set_xy([x_f[i] - lengthCar / 2., y_f[i] - widthCar / 2.])
        ax.add_patch(patch_f)

        if y_ft[i] == 0.:
            patch_ft.set_xy([x_ft[i] - lengthCar / 2., y_ft[i] - widthCar / 2. - 100.])
        else:
            patch_ft.set_xy([x_ft[i] - lengthCar / 2., y_ft[i] - widthCar / 2.])
        ax.add_patch(patch_ft)

        if y_rt[i] == 0.:
            patch_rt.set_xy([x_rt[i] - lengthCar / 2., y_rt[i] - widthCar / 2. - 100.])
        else:
            patch_rt.set_xy([x_rt[i] - lengthCar / 2., y_rt[i] - widthCar / 2.])
        ax.add_patch(patch_rt)

        return patch_e, patch_f, patch_ft, patch_rt

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=x_e.shape[0],
                                   interval=100,
                                   blit=True)
    writer = ImageMagickFileWriter(fps=15, bitrate=3000)
    anim.save(dir + 'lc_id' + str(ind) + '.gif', writer=writer)


dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
dir = dir0 + '0400pm-0415pm/'
# dir = dir0 + '0500pm-0515pm/'
# dir = dir0 + '0515pm-0530pm/'
data = np.genfromtxt(dir + 'lane_changes.csv', delimiter=',')
# for i in range(int(data[-1,0])+1):
idx = [135, 139, 175, 217, 298]
for i in idx:
    makeMovie(data, i, dir)
    print(i)
# plt.show()
