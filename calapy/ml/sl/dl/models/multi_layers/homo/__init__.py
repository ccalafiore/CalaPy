

"""The module for homogeneous neural networks (NNs).

The homogeneous NNs can have homogeneous layers (i.e. all fully-connected, all recurrent, or all convolutional, and so
on).
"""

from .fcns import *
from .rnns import *

__all__ = [
    'FCNN', 'IndFCNNs', 'FCNNsWithSharedLayersAndPrivateLayers',
    'RNN', 'IndRNNs', 'RNNsWithSharedLayersAndPrivateLayers', 'SharedRNNAndIndRNNsAndIndFCNNs',
    'LSTMNNs'
]


class LSTMNNs:

    def __init__(self):
        raise NotImplementedError()
