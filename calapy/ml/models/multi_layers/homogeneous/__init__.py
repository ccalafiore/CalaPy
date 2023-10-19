

from .fcn import *
from .gru import *

__all__ = [
    'SequentialFCLs', 'ParallelFCLs', 'SequentialFCLsParallelFCLs',
    'SequentialGRUs', 'ParallelGRUs', 'SequentialGRUsParallelGRUs',
    'SequentialLSTMs',
]


class SequentialLSTMs:

    def __init__(self):
        raise NotImplementedError()
