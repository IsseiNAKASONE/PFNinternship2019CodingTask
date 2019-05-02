import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import mpl.pyplot as plt



class Reporter:

    def __init__(self):
        self._observation = {}
        self._first = False

    def report(self, key, value):
        if key in self._observation.keys():
            self._observation[key] = np.hstack((self._observation[key], value))
        else:
            self._observation[key] = np.array([value])

    def print_report(self, ignore=[]):
        if not self._first:
            for key in self._observation.keys():
                if key not in ignore: print(key+'\t', end='')
            print()
            self._first = True

        for key, values in self._observation.items():
            if key not in ignore:
                value = values[-1]
                if isinstance(value, np.int64):
                    print('{}\t'.format(value), end='')
                else:
                    print('{:.6f}\t'.format(value), end='')
        print()
            
    def plot_report(self, outfile='loss.png'):
        pass

