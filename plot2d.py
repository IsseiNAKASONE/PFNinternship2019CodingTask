import json
import os
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from statistics import mean, stdev

EPOCH = 100


def plot2d(train, test, filepath, ylabel):
    train_mean = []
    train_err = []
    test_mean = []
    test_err = []
    epoch = [e+1 for e in range(EPOCH)] 
    for i in range(EPOCH):
        train_mean.append(mean(train[:,i]))
        train_err.append(stdev(train[:,i]))
        test_mean.append(mean(test[:,i]))
        test_err.append(stdev(test[:,i]))
    
    with open(filepath+'/'+ylabel+'.csv', 'w') as fd:
        fd.write('train_mean,')
        for y in train_mean: fd.write(str(y)+',')
        fd.write('\ntrain_err,')
        for y in train_err: fd.write(str(y)+',')
        fd.write('\ntest_mean,')
        for y in test_mean: fd.write(str(y)+',')
        fd.write('\ntest_err,')
        for y in test_err: fd.write(str(y)+',')
 
    fig = plt.figure(figsize=(9, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    plt.errorbar(epoch, train_mean, yerr=train_err, linewidth=2, 
            elinewidth=0.6, markersize=3, capsize=2, color='#0073CC', ecolor='#0073CC', label='main/'+ylabel)
    plt.errorbar(epoch, test_mean, yerr=test_err, linewidth=2,
            elinewidth=0.6, markersize=3, capsize=2, color='#FF9400', ecolor='#FF9400', label='test/main/'+ylabel)

    
    if ylabel == 'loss':
        plt.xlim([0, EPOCH+1])
        plt.ylim([0.5, 1.2])
    else:
        plt.xlim([0, EPOCH+1])
        plt.ylim([0.3, 0.8])
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)

    plt.grid(which='major',color='#999999',linestyle='--')
    ax.legend(loc='upper right', borderaxespad=1, fontsize=16)
    plt.savefig(filepath+'/'+ylabel+'.png')


def logplot(filepath):
    files = glob(filepath+'/*.json')

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for f in files:
        with open(f, 'r') as fd:
            data = json.load(fd)
            trls = []
            tsls = []
            trac = []
            tsac = []
            for d in data:
                trls.append(d['main/loss'])
                tsls.append(d['test/main/loss'])
                trac.append(d['main/accuracy'])
                tsac.append(d['test/main/accuracy'])
            train_loss.append(np.array(trls))
            test_loss.append(np.array(tsls))
            train_acc.append(np.array(trac))
            test_acc.append(np.array(tsac))

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    plot2d(train_loss, test_loss, filepath, 'loss')
    plot2d(train_acc, test_acc, filepath, 'accuracy')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type = str, help = 'directory path of input files', required = True)

    argv = parser.parse_args()
    logplot(argv.i)

