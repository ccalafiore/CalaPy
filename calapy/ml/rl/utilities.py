

import math


__all__ = ['EnvironmentsIterator']


class EnvironmentsIterator:

    def __init__(self, tot_observations_per_epoch):

        """

        :type tot_observations_per_epoch: int
        """

        if isinstance(tot_observations_per_epoch, int):
            self.tot_observations_per_epoch = tot_observations_per_epoch
        else:
            raise TypeError('tot_observations_per_epoch')

    def __iter__(self):
        self.i = -1
        self.s = 0
        return self

    def __next__(self):

        self.i += 1
        if self.s < self.tot_observations_per_epoch:
            return self.i
        else:
            raise StopIteration

    def __add__(self, n_new_observations):
        self.s += n_new_observations
        return self.s

    def count_observations(self, n_new_observations):
        return self + n_new_observations
