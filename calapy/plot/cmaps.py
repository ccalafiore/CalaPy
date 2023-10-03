import numpy as np
import math
from copy import deepcopy
from . import colors
# from .. import maths as cp_maths


class ColorMap:

    def __init__(self, name, n):



        if isinstance(name, str):

            self.name = name

            if self.name.lower() == 'fire':
                self.colors = [colors.dark_red, colors.red, colors.orange, colors.yellow, colors.light_yellow]
            elif name.lower() == 'fire_r':
                self.colors = [colors.light_yellow, colors.yellow, colors.orange, colors.red, colors.dark_red]
        else:
            self.name = 'custom'

            self.colors = name



        self.C = len(self.colors)

        if isinstance(n, list):
            self.N = n
        elif isinstance(n, tuple):
            self.N = list(n)
        elif isinstance(n, np.ndarray):
            self.N = n.tolist()
            if not isinstance(self.N, list):
                self.N = [self.N]
        else:
            self.N = [n]

        self.M = self.C - 1
        M_tmp = len(self.N)
        if M_tmp == 1:
            self.S = self.N[0]
            self.N = [math.floor(self.S / self.M)] * self.M
            S_tmp = sum(self.N)
            m = self.M
            while S_tmp != self.S:
                m -= 1
                self.N[m] += 1
                S_tmp += 1

        elif M_tmp == 4:
            self.S = sum(self.N)
        else:
            raise ValueError('n')

        endpoints = [False] * self.M
        endpoints[-1] = True
        arrays = [-1] * self.M  # type: list
        for m in range(self.M):
            arrays[m] = np.linspace(
                self.colors[m], self.colors[m + 1], num=self.N[m], endpoint=endpoints[m],
                retstep=False, dtype='f', axis=0)
            self.N[m] = arrays[m].shape[0]

        self.array = np.concatenate(arrays, axis=0)
        self.N = tuple(self.N)
        self.S, self.D = self.array.shape

    def reverse(self, copy=True):
        if copy:
            cmap_r = deepcopy(self)
            cmap_r.array = cmap_r.array[slice(-1, -(cmap_r.S + 1), -1), slice(0, cmap_r.D, 1)]
            cmap_r.N = cmap_r.N[slice(-1, -(cmap_r.M + 1), -1)]

            cmap_r.colors = cmap_r.colors[slice(-1, -(cmap_r.C + 1), -1)]
            if cmap_r.name[slice(-2, len(cmap_r.name), 1)] == '_r':
                cmap_r.name = cmap_r.name[slice(0, -2, 1)]
            else:
                cmap_r.name += '_r'

            return cmap_r
        else:
            self.array = self.array[slice(-1, -(self.S + 1), -1), slice(0, self.D, 1)]
            self.N = self.N[slice(-1, -(self.M + 1), -1)]

            self.colors = self.colors[slice(-1, -(self.C + 1), -1)]
            if self.name[slice(-2, len(self.name), 1)] == '_r':
                self.name = self.name[slice(0, -2, 1)]
            else:
                self.name += '_r'



