
from matplotlib import pyplot as plt
from .save import *
from .parameters import my_dpi
import importlib

# todo: fix the size unit of the figures


submodules = ['heatmaps', 'points', 'bars', 'cmaps', 'colors']
others = ['save_figures', 'my_dpi', 'plt']
__all__ = submodules + others

for sub_module_m in submodules:
    importlib.import_module(name='.' + sub_module_m, package=__package__)
