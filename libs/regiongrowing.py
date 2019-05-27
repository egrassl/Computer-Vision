import math

from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random

def ex3():

    im = Image.open('image/avocado.jpg').convert('L')
    arr = np.asarray(im)

    out = Image.open('image/avocado1.jpg').convert('L')
    arr_out = np.asarray(out)

    rows, columns = np.shape(arr)
    # print '\nrows',rows,'columns',columns
    plt.figure()
    plt.imshow(im)
    plt.gray()


    pseed = plt.ginput(1)
    # pseed
    # print pseed[0][0],pseed[0][1]

    x = int(pseed[0][0])
    y = int(pseed[0][1])
    # x = int(179)
    # y = int(86)
    seed_pixel = []
    seed_pixel.append(x)
    seed_pixel.append(y)


    # closing figure
    plt.close()

    img_rg = np.zeros((rows + 1, columns + 1))
    img_rg[seed_pixel[0]][seed_pixel[1]] = 255.0
    img_display = np.zeros((rows, columns))

    region_points = []
    region_points.append([x, y])


    def find_region():
        # print 'starting points',i,j
        count = 0
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]

        while (len(region_points) > 0):

            if count == 0:
                point = region_points.pop(0)
                i = point[0]
                j = point[1]
            # print 'count',count
            val = arr[i][j]
            lt = val - 8
            ht = val + 8
            # print 'value of pixel',val
            for k in range(8):
                # print '\ncomparison val:',val, 'ht',ht,'lt',lt
                if img_rg[i + x[k]][j + y[k]] != 1:
                    try:
                        if arr[i + x[k]][j + y[k]] > lt and arr[i + x[k]][j + y[k]] < ht:
                            # print '\nbelongs to region',arr[i+x[k]][j+y[k]]
                            img_rg[i + x[k]][j + y[k]] = 1
                            p = [0, 0]
                            p[0] = i + x[k]
                            p[1] = j + y[k]
                            if p not in region_points:
                                if 0 < p[0] < rows and 0 < p[1] < columns:
                                    region_points.append([i + x[k], j + y[k]])
                        else:
                            # print 'not part of region'
                            img_rg[i + x[k]][j + y[k]] = 0
                    except IndexError:
                        continue

            # print '\npoints list',region_points
            point = region_points.pop(0)
            i = point[0]
            j = point[1]
            count = count + 1
        # find_region(point[0], point[1])


    find_region()

    ground_out = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            if arr_out[i][j] > 125:
                ground_out[i][j] = int(1)

            else:
                ground_out[i][j] = int(0)

    tp = 0
    tn = 0
    fn = 0
    fp = 0

    for i in range(rows):
        for j in range(columns):
            if ground_out[i][j] == 1 and img_rg[i][j] == 1:
                tp = tp + 1
            if ground_out[i][j] == 0 and img_rg[i][j] == 0:
                tn = tn + 1
            if ground_out[i][j] == 1 and img_rg[i][j] == 0:
                fn = fn + 1
            if ground_out[i][j] == 0 and img_rg[i][j] == 1:
                fp = fp + 1
    ''' ********************************** Calculation of Tpr, Fpr, F-Score ***************************************************'''


    plt.figure()
    plt.imshow(img_rg, cmap="Greys_r")
    plt.colorbar()
    plt.show()

