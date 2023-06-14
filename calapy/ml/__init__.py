

import torch
import torchvision


__all__ = ['torch', 'torchvision', 'datasets', 'models', 'output_methods', 'devices', 'preprocess', 'test', 'train']


def initiate(names_submodules=None):
    template_import_string = "from . import "
    if isinstance(names_submodules, str):
        import_string_m = template_import_string + names_submodules
        exec(import_string_m)
    elif isinstance(names_submodules, (list, tuple)):
        for name_m in names_submodules:
            if isinstance(name_m, str):
                import_string_m = template_import_string + name_m
            else:
                import_string_m = template_import_string + str(name_m)
            exec(import_string_m)
    elif names_submodules is None:
        # exec('from . import *')
        for name_m in __all__:
            if isinstance(name_m, str):
                import_string_m = template_import_string + name_m
            else:
                import_string_m = template_import_string + str(name_m)
            exec(import_string_m)
    else:
        raise TypeError(names_submodules)


