

import torch
import typing
import numpy as np
from ...maths import prod as cp_prod

__all__ = ['ModelMethods']


class ModelMethods(torch.nn.Module):

    def __init__(self):

        superclass = ModelMethods
        subclass = type(self)
        if superclass == subclass:
            self.superclasses_initiated = []

        if torch.nn.Module not in self.superclasses_initiated:
            torch.nn.Module.__init__(self=self)
            if torch.nn.Module not in self.superclasses_initiated:
                self.superclasses_initiated.append(torch.nn.Module)

        self.dtype = None
        self.device = None

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def freeze(self):
        # Now set requires_grad to false
        for param_model in self.parameters():
            param_model.requires_grad = False

    def unfreeze(self):
        # Now set requires_grad to false
        for param_model in self.parameters():
            param_model.requires_grad = True

    def init_device(self, device: typing.Union[torch.device, str, None] = None):

        if device is None:
            self.device = device
        else:
            self.device = torch.device(device)
        return self.device

    def set_device(self):

        if self.device is not None:
            self.to(device=self.device)
        self.device = self.get_device()

        return self.device

    def get_device(self):
        no_param = True
        for param_model in self.parameters(recurse=True):
            self.device = param_model.device
            no_param = False
            break
        if no_param:
            self.device = None
        return self.device

    def get_dtype(self):
        no_param = True
        for param_model in self.parameters(recurse=True):
            self.dtype = param_model.dtype
            no_param = False
            break
        if no_param:
            self.dtype = None

        return self.dtype
