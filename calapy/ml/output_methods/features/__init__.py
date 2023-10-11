

import importlib

submodules = ['general', 'classifiers', 'DQNs', 'DQCs']

others = []
__all__ = submodules + others

for sub_module_m in submodules:
    importlib.import_module(name='.' + sub_module_m, package=__package__)
