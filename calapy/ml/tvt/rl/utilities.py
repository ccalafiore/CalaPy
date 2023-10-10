

import math


class EpochPhase:

    def __init__(self, n_environments_per_epoch, n_environments_per_batch, T=None):

        if isinstance(n_environments_per_epoch, int):
            self.n_environments_per_epoch = n_environments_per_epoch
        else:
            raise TypeError('n_environments_per_epoch')

        if isinstance(n_environments_per_batch, int):
            self.n_environments_per_batch = n_environments_per_batch
        else:
            raise TypeError('n_environments_per_batch')

        if T is None:
            self.T = math.inf
        elif isinstance(T, int):
            self.T = T
        else:
            raise TypeError('T')

        self.s = 0
        self.b = 0

    def __iter__(self):
        self.s = 0
        self.b = -1
        return self

    def __next__(self):

        self.s += self.n_environments_per_batch
        self.b += 1

        if self.s < self.n_environments_per_epoch:

            batch_i = Batch(n_environments_per_batch=self.n_environments_per_batch, T=self.T)

            return self.b, self.s, batch_i
        else:
            raise StopIteration


class Batch:

    def __init__(self, n_environments_per_batch, T=None):

        if isinstance(n_environments_per_batch, int):
            self.n_environments_per_batch = n_environments_per_batch
        else:
            raise TypeError('n_environments_per_batch')

        if T is None:
            self.T = math.inf
        elif isinstance(T, int):
            self.T = T
        else:
            raise TypeError('T')

        self.i = 0

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):

        self.i += 1

        if self.i < self.n_environments_per_batch:
            environment_i = Environment(T=self.T)

            return self.i, environment_i
        else:
            raise StopIteration


class Environment:

    def __init__(self, get_observation, T=None):

        self.get_observation = get_observation

        if T is None:
            self.T = math.inf
        elif isinstance(T, int):
            self.T = T
        else:
            raise TypeError('T')

        self.t = 0

    def __iter__(self):
        self.t = -1
        return self

    def __next__(self):

        self.t += 1

        if self.t < self.T:

            # todo: observation function
            observation_t = self.get_observation()

            return self.t
        else:
            raise StopIteration
