import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def plot_bandwidth_by_algorithm(data, out_dir, ord):

    algs = data["stencil"].unique()
    filtered_data = data[data["order"] == ord].copy()
    x = ['256x256', '512x512', '1024x1024', '2048x2048', '4096x4096']
    # x = np.array([256**2, 512**2, 1024**2, 2048**2, 4096**2])

    for alg in algs:
        y = filtered_data[filtered_data['stencil'] == alg]['bandwidth']
        plt.plot(x, y, label=alg)

    plt.xlabel('Grid size in MegaPoints')
    plt.ylabel('Bandwidth in Gb/sec')
    plt.legend()

    out_name = "bandwidth_by_alg_ord" + str(ord) + ".png"
    plt.savefig(
        os.path.join(out_dir, out_name), 
        format="png",
        dpi=100
    )
    plt.show()
    plt.close()


def plot_bandwidth_by_order(data, out_dir, alg):

    ords = data["order"].unique()
    filtered_data = data[data["stencil"] == alg]
    x = ['256x256', '512x512', '1024x1024', '2048x2048', '4096x4096']
    # x = np.array([256**2, 512**2, 1024**2, 2048**2, 4096**2])

    for ord in ords:
        y = filtered_data[filtered_data['order'] == ord]['bandwidth']
        plt.plot(x, y, label=ord)

    plt.xlabel('Grid size in MegaPoints')
    plt.ylabel('Bandwidth in Gb/sec')
    plt.legend()

    out_name = "bandwidth_by_order_" + alg + ".png"
    plt.savefig(
        os.path.join(out_dir, out_name), 
        format="png",
        dpi=100
    )
    plt.show()
    plt.close()

def main():

    filename = sys.argv[1]
    out_dir = os.path.join("docs")
    data = pd.read_csv(filename)
    # print(data.head())

    plt.style.use('ggplot')
    plot_bandwidth_by_algorithm(data, out_dir, ord=8)
    plot_bandwidth_by_order(data, out_dir, alg='block')


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python3 {} < infile name >'.format(
            sys.argv[0]))
        sys.exit(0)

    main()
