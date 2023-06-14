

import torch
import typing
import numpy as np
from ...maths import prod as cp_prod

__all__ = ['ModelMethods']


class ModelMethods(torch.nn.Module):

    def __init__(self, device: typing.Union[torch.device, str, None] = None):

        name_superclass = ModelMethods.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        if torch.nn.Module.__name__ not in self.superclasses_initiated:
            torch.nn.Module.__init__(self=self)
            self.superclasses_initiated.append(torch.nn.Module.__name__)

        self.device = self.init_device(device=device)

        self.superclasses_initiated.append(name_superclass)

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

    def get_device(self: torch.nn.Module):
        no_param = True
        for param_model in self.parameters():
            no_param = False
            self.device = param_model.device
            break
        if no_param:
            self.device = None
        return self.device

