import iterator



class Updater(object):

    def __init__(self, iterator, optimizer, loss_func=None):
        self._iterater = iterator
