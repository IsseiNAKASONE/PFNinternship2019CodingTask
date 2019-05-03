import json
import os
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from statistics import mean, stdev


EPOCH = 10

def plot2d(loss, vald, outfile):
    loss_mean = []
    loss_err = []
    vald_mean = []
    vald_err = []
    epoch = [e+1 for e in range(EPOCH)] 
    for i in range(EPOCH):
        loss_mean.append(mean(loss[:,i]))
        loss_err.append(stdev(loss[:,i]))
        vald_mean.append(mean(vald[:,i]))
        vald_err.append(stdev(vald[:,i]))
    
    fig = plt.figure(figsize=(10, 7.5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(epoch, loss_mean, linestyle='-', color='b', linewidth=0.8, marker='x', label='main/loss')
    ax.plot(epoch, vald_mean, linestyle='--', color='g', linewidth=0.8, marker='o', label='val/main/loss')
    plt.errorbar(epoch, loss_mean, linestyle='-', yerr=loss_err, linewidth=0.8, elinewidth=0.4, marker='x', capsize=2, color='b', ecolor='b', label='main/loss')
    plt.errorbar(epoch, vald_mean, yerr=vald_err, elinewidth=0.4, capsize=2, ecolor='g')

    
    plt.xlim([0, 11])
    plt.ylim([0.5, 2.0])
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    plt.grid(which='major',color='#999999',linestyle='--')

    ax.legend(loc='upper right', borderaxespad=1, fontsize=30)
    plt.savefig(outfile)



def lossplot(filepath, outfile):
    files = glob(filepath+'/*.json')
    loss = []
    vald = []

    for f in files:
        with open(f, 'r') as fd:
            data = json.load(fd)
            losp = []
            vasp = []
            for d in data:
                losp.append(d['main/loss'])
                vasp.append(d['val/main/loss'])
            loss.append(np.array(losp))
            vald.append(np.array(vasp))

    loss = np.array(loss)
    vald = np.array(vald)
    plot2d(loss, vald, outfile)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type = str, help = 'directory path of input files', required = True)
    parser.add_argument('-o', type = str, help = 'output file name', required = True)

    argv = parser.parse_args()
    lossplot(argv.i, argv.o)

