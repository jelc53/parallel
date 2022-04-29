import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():

    filename = sys.argv[1]
    out_dir = os.path.join('docs')
    data = pd.read_csv(filename)

    plt.plot(data.iloc[:,0], data.iloc[:,1])
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])

    plt.savefig(os.path.join(out_dir, filename[:-4] + '.png'))

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print('Usage:')
        print('  python3 {} < infile name > < outfile name'.format(
            sys.argv[0]))
        sys.exit(0)

    main()