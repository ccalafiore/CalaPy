
import math


class EpochsIterator:

    def __init__(self, U=10, E=None):

        """

        :type U: int | float | None
        :type E: int | float | None
        """

        if U is None:
            self.U = math.inf
        elif isinstance(U, int):
            self.U = U
        elif isinstance(U, float):
            if U == math.inf:
                self.U = U
            else:
                raise ValueError
        else:
            raise TypeError('U')

        if E is None:
            self.E = math.inf
        elif isinstance(E, int):
            self.E = E
        elif isinstance(E, float):
            if E == math.inf:
                self.E = E
            else:
                raise ValueError
        else:
            raise TypeError('E')

        self.u = 0
        self.e = 0
        self.are_unsuccessful_epochs_counted = False

    def __iter__(self):
        self.e = -1
        self.u = 0
        self.are_unsuccessful_epochs_counted = True
        return self

    def __next__(self):

        if self.are_unsuccessful_epochs_counted:
            self.e += 1

            if (self.e < self.E) and (self.u < self.U):
                self.are_unsuccessful_epochs_counted = False
                return self.e, self.u
            else:
                raise StopIteration
        else:
            raise EnvironmentError(
                'epochs.count_unsuccessful_epochs() needs to be called one time at end of each epoch')

    def count_unsuccessful_epochs(self, is_successful_epoch):

        """

        :type is_successful_epoch: bool
        """

        if self.are_unsuccessful_epochs_counted:
            raise EnvironmentError(
                'epochs.count_unsuccessful_epochs() needs to be called only one time at end of each epoch')
        else:
            if is_successful_epoch:
                self.u = 0
            else:
                self.u += 1
            self.are_unsuccessful_epochs_counted = True

        return self.u
