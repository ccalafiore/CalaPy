

from .fcns import *
from .rnns import *

__all__ = [
    'FCNN', 'IndFCNNs', 'FCNNsWithSharedShallowerLayers',
    'RNN', 'IndRNNs', 'RNNsWithSharedShallowerLayers',
    'LSTMNNs'
]


class LSTMNNs:

    def __init__(self):
        raise NotImplementedError()
