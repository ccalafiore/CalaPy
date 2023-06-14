import torchvision.models as models
import torch


def new_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


# define a resnet instance
resent = models.resnet18()

# add new_forward function to the resnet instance as a class method
bound_method = new_forward.__get__(resent, resent.__class__)
setattr(resent, 'forward', bound_method)

# you can also remove the 2 layers resent.avgpool and resent.fc because you are not using them in the new forward method
delattr(resent, 'avgpool')
delattr(resent, 'fc')

# call the new forward method
inputs = torch.rand(1, 3, 224, 224)
outputs = resent(inputs)

print('type(resent) = ', type(resent))
print('type(resent.forward) = ', type(resent.forward))
print('outputs.shape = ', outputs.shape)
