

# import importlib


# submodules = ['models', 'output_methods']
#
# others = []
# __all__ = submodules + others
#
# for sub_module_m in submodules:
#     importlib.import_module(name='.' + sub_module_m, package=__package__)

from .module import *

__all__ = ['DQNMethods', 'TimedDQNMethods']
