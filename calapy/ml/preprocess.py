# websites:
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/hub/pytorch_vision_resnet/
# https://discuss.pytorch.org/t/normalize-each-input-image-in-a-batch-independently-and-inverse-normalize-the-output/23739
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from . import torch
from .. import txt as cp_txt
from .. import clock as cp_clock
from .. import directory as cp_directory
import os


def extract_features(model, loader, directory_dataset_features=None):
    cp_timer = cp_clock.Timer()

    n_decimals_for_printing = 6

    if model.training:
        model.eval()  # Set model to evaluate mode

    # Now set requires_grad to false
    for param in model.parameters():
        param.requires_grad = False

    torch.set_grad_enabled(False)

    if directory_dataset_features is None:
        directory_dataset_features = 'features'

    os.makedirs(directory_dataset_features, exist_ok=True)

    # Iterate over data.
    for data_eb in loader:

        samples_eb, relative_directories_eb = data_eb

        # forward
        outputs_eb = model(samples_eb)
        # outputs_eb = outputs_eb.numpy()

        # _, a = torch.max(outputs_eb, 1)
        # a = a.cpu().numpy()

        # 1. produce the directory_features
        relative_directories_features_eb = cp_directory.replace_extensions(relative_directories_eb, 'csv')
        absolute_directories_features_eb = cp_directory.conditions_to_directories(
            [[directory_dataset_features], relative_directories_features_eb], order_outputs='v')

        # 2. save features (make funtion array_to_csv_files(array, directories, axes='frc'))
        outputs_eb = torch.unsqueeze(outputs_eb, 1).cpu().numpy()

        cp_txt.array_to_csv_files(
            outputs_eb, 1, 2, [absolute_directories_features_eb], headers=None)

    time_extraction = cp_timer.get_delta_time()

    print('Test completed in {d} days {h} hours {m} minutes {s} seconds'.format(
        d=time_extraction.days, h=time_extraction.hours,
        m=time_extraction.minutes, s=time_extraction.seconds))
