import datasets as D
import numpy as np



class Iterator:

    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.order_sampler = ShuffleOrderSampler() 
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        self._previous_epoch_detail = self.epoch_detail

        offset = self.current_position
        end = offset + self.batch_size
        N = self._epoch_size

        batch = [self.dataset[index] for index in self._order[offset:end]]

        if end >= N:
            rest = end - N
            if self._order is not None:
                new_order = self.order_sampler(self._order, offset)
                if len(self._order) != len(new_order): raise ValueError
                self._order = new_order
                if rest > 0:
                    if self._order is None:
                        batch.extend(self.dataset[:rest])
                    else:
                        batch.extend([self.dataset[index]
                            for index in self._order[:rest]])
                self.current_position = rest

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = end

        return batch
        next = __next__

    @property
    def epoch_detail(self):
        return (self.epoch+self.current_position)/self._epoch_size

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0: return None
        else:                               return self._previous_epoch_detail

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self._previous_epoch_detail = -1.
        self._order = self.order_sampler(np.arange(len(self.dataset)), 0)

    @property
    def _epoch_size(self):
        if self._order is None: return len(self.dataset)
        else:                   return len(self._order)

    @property
    def repeat(self):
        return self._repeat



class ShuffleOrderSampler(object):

    def __init__(self):
        self._random = np.random.random.__self__

    def __call__(self, current_order, current_position):
        return self._random.permutation(len(current_order))    
