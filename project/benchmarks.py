import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def plot_bandwidth_by_algorithm(data, out_dir, dtypes):

    algs = data['alg'].unique()
    x = ['40', '400', '4000', '40000']
    # x = np.array([256**2, 512**2, 1024**2, 2048**2, 4096**2])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharey=False)

    for i in range(len(dtypes)):
        filtered_data = data[data['type'] == dtypes[i]].copy()
        axs[i].set_yscale('log')
        axs[i].set_title(dtypes[i])
        axs[i].set(xlabel='Problem size (N = number of columns)', ylabel='Time to execute (sec)')
        for alg in algs:
            y = filtered_data[filtered_data['alg'] == alg]['time_sec']
            axs[i].plot(x, y, label=alg)

    plt.legend()

    out_name = "speed_alg_vs_problem_size.png"
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
    print(data.head())

    plt.style.use('ggplot')
    plot_bandwidth_by_algorithm(data, out_dir, dtypes=['fp32', 'fp64'])


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python3 {} < infile name >'.format(
            sys.argv[0]))
        sys.exit(0)

    main()
