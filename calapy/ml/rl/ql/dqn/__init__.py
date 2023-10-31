

# import importlib


# submodules = ['models', 'output_methods']
#
# others = []
# __all__ = submodules + others
#
# for sub_module_m in submodules:
#     importlib.import_module(name='.' + sub_module_m, package=__package__)

from .output_methods import *

__all__ = ['DQNMethods', 'TimedDQNMethods']
