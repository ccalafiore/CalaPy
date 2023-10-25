

# import importlib
#
#
# submodules = ['dqc', 'dqn']
#
# others = []
# __all__ = submodules + others
#
# for sub_module_m in submodules:
#     importlib.import_module(name='.' + sub_module_m, package=__package__)

from .dqn import *

__all__ = ['DQNMethods', 'TimedDQNMethods']
