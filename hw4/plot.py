import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def read_data(filename):
    with open(filename) as f:
        xdim = int(f.readline().strip())
        ydim = int(f.readline().strip())
        raw = f.readline().strip().split(",")
    
    data = []
    for x in raw:
        if (x == ""):
            continue;
        data.append(float(x))

    return xdim, ydim, data

def main():

    filename = sys.argv[1]
    out_dir = os.path.join('docs')
    xdim, ydim, data = read_data(filename)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    X = np.arange(0, xdim, 1)
    Y = np.arange(0, ydim, 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(data).reshape((xdim, ydim))

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, 
                    linewidth=0, antialiased=False)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("T")

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.01f}')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
    
    # figure = plt.gcf()
    plt.savefig(
        os.path.join(out_dir, filename[:-4] + '.png'), 
        format="png",
        dpi=100
    )
    plt.show()
    plt.close()

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python3 {} < infile name >'.format(
            sys.argv[0]))
        sys.exit(0)

    main()
