
import math


class Epochs:

    def __init__(self, I, E):

        if I is None:
            self.I = math.inf
        elif isinstance(I, int):
            self.I = I
        else:
            raise TypeError('I')

        if E is None:
            self.E = math.inf
        elif isinstance(E, int):
            self.E = E
        else:
            raise TypeError('E')

        self.i = 0
        self.e = 0

    def __iter__(self):
        self.e = -1
        self.i = 0
        self.are_unsuccessful_epochs_counted = True
        return self

    def __next__(self):

        if self.are_unsuccessful_epochs_counted:
            self.e += 1

            if (self.e < self.E) and (self.i < self.I):
                self.are_unsuccessful_epochs_counted = False
                return self.e, self.i
            else:
                raise StopIteration
        else:
            raise EnvironmentError(
                'epochs.count_unsuccessful_epochs() needs to be called one time at end of each epoch')

    def count_unsuccessful_epochs(self, is_successful_epoch):

        if self.are_unsuccessful_epochs_counted:
            raise EnvironmentError(
                'epochs.count_unsuccessful_epochs() needs to be called only one time at end of each epoch')
        else:
            self.are_unsuccessful_epochs_counted = True
            if is_successful_epoch:
                self.i = 0
            else:
                self.i += 1

        return self.i
