

import math


__all__ = ['EnvironmentsIterator', 'ObservationsIterator']


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
        self.s = 0
        return self

    def __next__(self):

        if self.s < self.tot_observations_per_epoch:
            return self.s
        else:
            raise StopIteration

    def __add__(self, n_new_observations):
        self.s += n_new_observations
        return self.s

    def count_observations(self, n_new_observations):
        return self + n_new_observations


class ObservationsIterator:

    def __init__(self, T=None):

        """
        :type T: int | None
        """

        if T is None:
            self.T = math.inf
        elif isinstance(T, int):
            self.T = T
        else:
            raise TypeError('T')

    def __iter__(self):
        self.t = -1
        self.not_over = True
        return self

    def __next__(self):

        self.t += 1

        if self.not_over or (self.t < self.T):
            return self.t
        else:
            raise StopIteration
