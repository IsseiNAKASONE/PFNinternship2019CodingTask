import numpy as np
import json



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
                    print('{}\t'.format(int(value)), end='')
                else:
                    print('{:.6f}\t'.format(value), end='')
        print()
     
    def log_report(self, outfile='log.json'):
        obj = []
        length = len(list(self._observation.values())[0])
        for i in range(length):
            tup = {} 
            for key, value in self._observation.items():
                val = value[i]
                if isinstance(val, np.int64):
                    val = int(val)
                else:
                    val = float(val)
                tup[key] = val
            obj.append(tup)
        
        with open(outfile, 'w') as fd:
            json.dump(obj, fd, sort_keys=True, indent=4)

