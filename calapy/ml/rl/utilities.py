

import math


__all__ = ['TrainEpisodesIterator', 'ValEpisodesIterator', 'ObservationsIterator']


class TrainEpisodesIterator:

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
        self.j = -1
        self.b = -1
        self.s = 0
        return self

    def __next__(self):

        self.i += 1
        self.j += 1

        if self.not_over:
            return self.i, self.j
        else:
            raise StopIteration

    def __add__(self, n_new_observations):
        self.s += n_new_observations
        self.b += 1
        self.j = -1
        return self.s, self.b

    def count_observations(self, n_new_observations):
        return self + n_new_observations

    @property
    def not_over(self):
        return self.s < self.tot_observations_per_epoch


class ValEpisodesIterator:

    def __init__(self, n_episodes):

        """

        :type n_episodes: int
        """

        if isinstance(n_episodes, int):
            self.n_episodes = n_episodes
        else:
            raise TypeError('n_episodes')

    def __iter__(self):
        self.i = -1
        self.s = 0
        return self

    def __next__(self):

        if self.not_over:
            return self.i
        else:
            raise StopIteration

    def __add__(self, n_new_episodes):
        self.i += n_new_episodes
        return self.i

    def count_observations(self, n_new_observations):
        self.s += n_new_observations
        return self.s

    def count_episodes(self, n_new_episodes):
        return self + n_new_episodes

    @property
    def not_over(self):
        return self.i < self.n_episodes


class ObservationsIterator:

    def __init__(self, T=None):

        """
        :type T: int | None
        """

        if T is None:
            self.T = math.inf
        elif isinstance(T, int):
            self.T = T
        elif isinstance(T, float):
            if T == math.inf:
                self.T = T
            else:
                raise ValueError('T')
        else:
            raise TypeError('T')

    def __iter__(self):
        self.t = -1
        self.not_over = True
        return self

    def __next__(self):

        self.t += 1

        if self.not_over and (self.t < self.T):
            return self.t
        else:
            raise StopIteration
