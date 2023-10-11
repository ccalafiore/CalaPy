import torch
import typing

__all__ = ['define_device', 'set_device']


def define_device(device: typing.Union[torch.device, str, None] = None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError


def set_device(tensor, device):

    # if isinstance(tensor, torch.Tensor):
    if isinstance(tensor, (torch.nn.Module, torch.Tensor)):
        if device is None:
            return tensor
        elif isinstance(device, (str, torch.device)):
            tensor = tensor.to(device)
            return tensor
        else:
            raise TypeError('device')
    else:
        raise TypeError('tensor')
