# websites:
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/hub/pytorch_vision_resnet/
# https://discuss.pytorch.org/t/normalize-each-input-image-in-a-batch-independently-and-inverse-normalize-the-output/23739
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import torchvision

if __name__ == "__main__":
    from calapy.ml.sl.dl import devices as cp_device
else:
    from ...dl import devices as cp_device


class ResNetNoLastLayer(torchvision.models.resnet.ResNet):

    def __init__(self, name_resnet, device=None):

        if name_resnet == 'resnet18':
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
        elif name_resnet == 'resnet34':
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3])
        elif name_resnet == 'resnet50':
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        elif name_resnet == 'resnet101':
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
        elif name_resnet == 'resnet152':
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3])

        elif name_resnet == 'resnext50_32x4d':
            kwargs = {'groups': 32, 'width_per_group': 4}
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)

        elif name_resnet == 'resnext101_32x8d':
            kwargs = {'groups': 32, 'width_per_group': 8}
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], **kwargs)

        elif name_resnet == 'wide_resnet50_2':
            kwargs = {'width_per_group': 64 * 2}
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)

        elif name_resnet == 'wide_resnet101_2':
            kwargs = {'width_per_group': 64 * 2}
            super(ResNetNoLastLayer, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], **kwargs)
        else:
            ValueError('name_resnet')

        delattr(self, 'fc')

        self.device = cp_device.define_device(device)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_resnet(name_resnet, last_layer=True, K=None, softmax=False, pretrained=False, device=None):

    if last_layer:
        if isinstance(pretrained, str):
            resnet = load_model(name_resnet, pretrained=False, device=None)
        else:
            resnet = load_model(name_resnet, pretrained=pretrained, device=None)

        if (K is None) or (K == resnet.fc.out_features):
            if softmax:
                resnet.fc = torch.nn.Sequential(resnet.fc, torch.nn.Softmax())
        else:
            num_ftrs = resnet.fc.in_features
            if softmax:
                resnet.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, K), torch.nn.Softmax())
            else:
                # Here the size of each output sample is set to K.
                resnet.fc = torch.nn.Linear(num_ftrs, K)
        if isinstance(pretrained, str):
            # TODO: what if the model was saved with cuda and is being loaded to cpu?
            state_dict = torch.load(pretrained)
            resnet.load_state_dict(state_dict)
    else:
        resnet = ResNetNoLastLayer(name_resnet)

        if pretrained == False:
            pass
        else:
            if isinstance(pretrained, str):
                state_dict = torch.load(pretrained)

            elif pretrained:
                urls_resnets = {
                    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
                    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
                    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
                    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
                }
                state_dict = torch.hub.load_state_dict_from_url(urls_resnets[name_resnet])
            else:
                raise TypeError('pretrained')

            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            resnet.load_state_dict(state_dict)

    device_resnet = cp_device.define_device(device)
    resnet.to(device_resnet)

    return resnet


def load_model(name_model, pretrained=False, device=None):
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)

    # model = torchvision.models.resnet18()
    if isinstance(name_model, str):
        template_load_string = f"model = torchvision.models.{name_model:s}(pretrained=pretrained)"
    else:
        template_load_string = f"model = torchvision.models.{str(name_model):s}(pretrained=pretrained)"

    dict_globals = {'__builtins__': None}
    if isinstance(pretrained, str):
        dict_locals = {'torchvision': torchvision, 'pretrained': False}  # type: dict
    else:
        dict_locals = {'torchvision': torchvision, 'pretrained': pretrained}  # type: dict
    exec(template_load_string, dict_globals, dict_locals)
    model = dict_locals['model']

    if isinstance(pretrained, str):
        model.load_state_dict(torch.load(pretrained))

    device_model = cp_device.define_device(device)
    model.to(device_model)

    return model
