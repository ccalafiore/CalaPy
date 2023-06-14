# websites:
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/hub/pytorch_vision_resnet/
# https://discuss.pytorch.org/t/normalize-each-input-image-in-a-batch-independently-and-inverse-normalize-the-output/23739
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from . import torch
import numpy as np
import typing
import math
import os
import copy
from .. import txt as cp_txt
from .. import clock as cp_clock
from .. import strings as cp_strings


def passive_feature_sequence_classifiers(
        model, loader, optimizer, scheduler, I=10, E=None, directory_outputs=None):
    cp_timer = cp_clock.Timer()

    if model.training:
        model.eval()
    model.freeze()
    torch.set_grad_enabled(False)

    for key_loader_k in loader.keys():
        if key_loader_k == 'training' or key_loader_k == 'validation':
            pass
        else:
            raise ValueError('Unknown keys in loader')

    headers = [
        'Epoch', 'Unsuccessful_Epochs',

        'Training_Unweighted_Class_Prediction_Loss',
        'Training_Weighted_Class_Prediction_Loss',
        'Training_Accuracy',

        'Training_Unweighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Training_Weighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Training_Accuracy_In_Last_Time_Point',

        'Training_Unweighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Training_Weighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Training_Accuracy_In_Each_Time_Point',

        'Validation_Unweighted_Class_Prediction_Loss',
        'Lowest_Validation_Unweighted_Class_Prediction_Loss',
        'Is_Lower_Validation_Unweighted_Class_Prediction_Loss',

        'Validation_Weighted_Class_Prediction_Loss',
        'Lowest_Validation_Weighted_Class_Prediction_Loss',
        'Is_Lower_Validation_Weighted_Class_Prediction_Loss',

        'Validation_Accuracy',
        'Highest_Validation_Accuracy',
        'Is_Higher_Accuracy',

        'Validation_Unweighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Validation_Weighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Validation_Accuracy_In_Last_Time_Point',

        'Validation_Unweighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Validation_Weighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Validation_Accuracy_In_Each_Time_Point']

    n_columns = len(headers)
    new_line_stats = [None for i in range(0, n_columns, 1)]  # type: list

    stats = {
        'headers': {headers[k]: k for k in range(n_columns)},
        'n_columns': n_columns,
        'lines': []}

    if directory_outputs is None:
        directory_outputs = 'outputs'
    os.makedirs(directory_outputs, exist_ok=True)

    directory_model_at_last_epoch = os.path.join(directory_outputs, 'model_at_last_epoch.pth')

    directory_model_with_lowest_unweighted_class_prediction_loss = os.path.join(
        directory_outputs, 'model_with_lowest_unweighted_class_prediction_loss.pth')

    directory_model_with_lowest_weighted_class_prediction_loss = os.path.join(
        directory_outputs, 'model_with_lowest_weighted_class_prediction_loss.pth')

    directory_model_with_highest_accuracy = os.path.join(directory_outputs, 'model_with_highest_accuracy.pth')

    directory_stats = os.path.join(directory_outputs, 'stats.csv')

    separators_times = '  '

    n_decimals_for_printing = 6
    n_dashes = 150
    dashes = '-' * n_dashes
    print(dashes)

    lowest_unweighted_class_prediction_loss = math.inf
    lowest_unweighted_class_prediction_loss_str = str(lowest_unweighted_class_prediction_loss)

    lowest_weighted_class_prediction_loss = math.inf
    lowest_weighted_class_prediction_loss_str = str(lowest_weighted_class_prediction_loss)

    highest_accuracy = -math.inf
    highest_accuracy_str = str(highest_accuracy)

    if E is None:
        E = math.inf

    if I is None:
        I = math.inf

    i = 0
    e = 0

    while (e < E) and (i < I):

        print('Epoch {e} ...'.format(e=e))

        stats['lines'].append(new_line_stats.copy())
        stats['lines'][e][stats['headers']['Epoch']] = e

        # Each epoch has a training and a validation phase
        # training phase

        running_n_corrects_e = 0
        running_n_classifications_e = 0
        running_unweighted_class_prediction_loss_e = 0.0
        running_weighted_class_prediction_loss_e = 0.0

        running_n_corrects_T_e = 0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_n_classifications_T_e = 0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_unweighted_class_prediction_losses_T_e = 0.0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_weighted_class_prediction_losses_T_e = 0.0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]

        b = 0
        # Iterate over data.
        for batch_eb, labels_eb in loader['training']:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history
            torch.set_grad_enabled(True)
            model.unfreeze()
            model.train()

            predictions_classes_eb = model(batch_eb)

            # compute class prediction loss
            class_prediction_losses_eb = model.compute_class_prediction_losses(
                predictions_classes=predictions_classes_eb, labels=labels_eb)

            weighted_class_prediction_loss_eb = model.reduce_losses(
                losses=class_prediction_losses_eb, weigh=True, type_models=0, axes_not_included=None)

            # backward + optimize
            weighted_class_prediction_loss_eb.backward()
            optimizer.step()

            model.eval()
            model.freeze()
            torch.set_grad_enabled(False)

            unweighted_class_prediction_loss_eb = model.reduce_losses(
                losses=class_prediction_losses_eb, weigh=False, type_models=0, axes_not_included=None)

            # compute accuracy
            classifications_eb = model.compute_classifications(predictions_classes=predictions_classes_eb)
            correct_classifications_eb = model.compute_correct_classifications(
                classifications=classifications_eb, labels=labels_eb)
            n_corrects_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb, axes_not_included=None, keepdim=False)
            n_classifications_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=None)

            running_n_corrects_e += n_corrects_eb
            running_n_classifications_e += n_classifications_eb
            running_unweighted_class_prediction_loss_e += (
                        unweighted_class_prediction_loss_eb.item() * n_classifications_eb)
            running_weighted_class_prediction_loss_e += (
                        weighted_class_prediction_loss_eb.item() * n_classifications_eb)

            # compute accuracy for each time point
            n_corrects_T_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb,
                axes_not_included=model.axis_time_losses, keepdim=False)
            n_classifications_T_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=model.axis_time_losses)

            # compute class prediction losses for each time point
            unweighted_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                weighted=False, loss_weights_classifiers=None, format_weights=False)

            weighted_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                weighted=True, loss_weights_classifiers=None, format_weights=False)

            running_n_corrects_T_e += n_corrects_T_eb
            running_n_classifications_T_e += n_classifications_T_eb
            running_unweighted_class_prediction_losses_T_e += (
                        unweighted_class_prediction_losses_T_eb * n_classifications_T_eb)
            running_weighted_class_prediction_losses_T_e += (
                        weighted_class_prediction_losses_T_eb * n_classifications_T_eb)

            b += 1

        # scheduler.step()

        unweighted_class_prediction_loss_e = running_unweighted_class_prediction_loss_e / running_n_classifications_e
        weighted_class_prediction_loss_e = running_weighted_class_prediction_loss_e / running_n_classifications_e
        accuracy_e = running_n_corrects_e / running_n_classifications_e

        unweighted_class_prediction_losses_T_e = (
                    running_unweighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        weighted_class_prediction_losses_T_e = (
                    running_weighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        accuracy_T_e = (running_n_corrects_T_e / running_n_classifications_T_e)

        last_unweighted_class_prediction_loss_e = unweighted_class_prediction_losses_T_e[-1].item()
        last_weighted_class_prediction_loss_e = weighted_class_prediction_losses_T_e[-1].item()
        last_accuracy_e = accuracy_T_e[-1].item()

        stats['lines'][e][stats['headers']['Training_Unweighted_Class_Prediction_Loss']] = (
            unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Weighted_Class_Prediction_Loss']] = (
            weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Accuracy']] = accuracy_e

        stats['lines'][e][stats['headers']['Training_Unweighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Weighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Accuracy_In_Last_Time_Point']] = last_accuracy_e

        stats['lines'][e][stats['headers']['Training_Unweighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in unweighted_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Training_Weighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in weighted_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Training_Accuracy_In_Each_Time_Point']] = separators_times.join(
            [str(t) for t in accuracy_T_e.tolist()])

        unweighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            unweighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        weighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            weighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        accuracy_str_e = cp_strings.format_float_to_str(accuracy_e, n_decimals=n_decimals_for_printing)

        print(
            'Epoch: {e:d}. Training. Unweighted Classification Loss: {class_prediction_loss:s}. Accuracy: {accuracy:s}.'.format(
                e=e, class_prediction_loss=unweighted_class_prediction_loss_str_e, accuracy=accuracy_str_e))

        # validation phase

        running_n_corrects_e = 0
        running_n_classifications_e = 0
        running_unweighted_class_prediction_loss_e = 0.0
        running_weighted_class_prediction_loss_e = 0.0

        running_n_corrects_T_e = 0
        running_n_classifications_T_e = 0
        running_unweighted_class_prediction_losses_T_e = 0.0
        running_weighted_class_prediction_losses_T_e = 0.0

        b = 0
        # Iterate over data.
        for batch_eb, labels_eb in loader['validation']:
            predictions_classes_eb = model(batch_eb)

            # compute accuracy
            classifications_eb = model.compute_classifications(
                predictions_classes=predictions_classes_eb, axis_classes=None)
            correct_classifications_eb = model.compute_correct_classifications(
                classifications=classifications_eb, labels=labels_eb)
            n_corrects_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb, axes_not_included=None, keepdim=False)
            n_classifications_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=None)

            # compute class prediction loss
            class_prediction_losses_eb = model.compute_class_prediction_losses(
                predictions_classes=predictions_classes_eb, labels=labels_eb)

            unweighted_class_prediction_loss_eb = model.reduce_losses(
                losses=class_prediction_losses_eb, weigh=False, type_models=0, axes_not_included=None)
            weighted_class_prediction_loss_eb = model.reduce_losses(
                losses=class_prediction_losses_eb, weigh=True, type_models=0, axes_not_included=None)

            running_n_corrects_e += n_corrects_eb
            running_n_classifications_e += n_classifications_eb
            running_unweighted_class_prediction_loss_e += unweighted_class_prediction_loss_eb.item() * n_classifications_eb
            running_weighted_class_prediction_loss_e += weighted_class_prediction_loss_eb.item() * n_classifications_eb

            # compute accuracy for each time point
            n_corrects_T_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb,
                axes_not_included=model.axis_time_losses, keepdim=False)
            n_classifications_T_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=model.axis_time_losses)

            # compute class prediction loss for each time point
            unweighted_class_prediction_losses_T_eb = model.reduce_losses(
                losses=class_prediction_losses_eb, weigh=False, type_models=0,
                axes_not_included=model.axis_time_losses)
            weighted_class_prediction_losses_T_eb = model.reduce_losses(
                losses=class_prediction_losses_eb, weigh=True, type_models=0,
                axes_not_included=model.axis_time_losses)

            running_n_corrects_T_e += n_corrects_T_eb
            running_n_classifications_T_e += n_classifications_T_eb
            running_unweighted_class_prediction_losses_T_e += unweighted_class_prediction_losses_T_eb * n_classifications_T_eb
            running_weighted_class_prediction_losses_T_e += weighted_class_prediction_losses_T_eb * n_classifications_T_eb

            b += 1

        unweighted_class_prediction_loss_e = running_unweighted_class_prediction_loss_e / running_n_classifications_e
        weighted_class_prediction_loss_e = running_weighted_class_prediction_loss_e / running_n_classifications_e
        accuracy_e = running_n_corrects_e / running_n_classifications_e

        unweighted_class_prediction_losses_T_e = (
                running_unweighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        weighted_class_prediction_losses_T_e = (
                running_weighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        accuracy_T_e = running_n_corrects_T_e / running_n_classifications_T_e

        last_unweighted_class_prediction_loss_e = unweighted_class_prediction_losses_T_e[-1].item()
        last_weighted_class_prediction_loss_e = weighted_class_prediction_losses_T_e[-1].item()
        last_accuracy_e = accuracy_T_e[-1].item()

        stats['lines'][e][stats['headers']['Validation_Unweighted_Class_Prediction_Loss']] = (
            unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Weighted_Class_Prediction_Loss']] = (
            weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Accuracy']] = accuracy_e

        stats['lines'][e][stats['headers']['Validation_Unweighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Weighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Accuracy_In_Last_Time_Point']] = last_accuracy_e

        stats['lines'][e][stats['headers']['Validation_Unweighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in unweighted_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Validation_Weighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in weighted_class_prediction_losses_T_e.tolist()]))

        stats['lines'][e][stats['headers']['Validation_Accuracy_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in accuracy_T_e.tolist()]))

        model_dict = copy.deepcopy(model.state_dict())
        if os.path.isfile(directory_model_at_last_epoch):
            os.remove(directory_model_at_last_epoch)
        torch.save(model_dict, directory_model_at_last_epoch)

        is_successful_epoch = False

        if unweighted_class_prediction_loss_e < lowest_unweighted_class_prediction_loss:

            lowest_unweighted_class_prediction_loss = unweighted_class_prediction_loss_e
            lowest_unweighted_class_prediction_loss_str = cp_strings.format_float_to_str(
                lowest_unweighted_class_prediction_loss, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Lower_Validation_Unweighted_Class_Prediction_Loss']] = 1
            is_successful_epoch = True

            if os.path.isfile(directory_model_with_lowest_unweighted_class_prediction_loss):
                os.remove(directory_model_with_lowest_unweighted_class_prediction_loss)
            torch.save(model_dict, directory_model_with_lowest_unweighted_class_prediction_loss)
        else:
            stats['lines'][e][stats['headers']['Is_Lower_Validation_Unweighted_Class_Prediction_Loss']] = 0

        stats['lines'][e][stats['headers']['Lowest_Validation_Unweighted_Class_Prediction_Loss']] = (
            lowest_unweighted_class_prediction_loss)

        if weighted_class_prediction_loss_e < lowest_weighted_class_prediction_loss:

            lowest_weighted_class_prediction_loss = weighted_class_prediction_loss_e
            lowest_weighted_class_prediction_loss_str = cp_strings.format_float_to_str(
                lowest_weighted_class_prediction_loss, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Lower_Validation_Weighted_Class_Prediction_Loss']] = 1
            is_successful_epoch = True

            if os.path.isfile(directory_model_with_lowest_weighted_class_prediction_loss):
                os.remove(directory_model_with_lowest_weighted_class_prediction_loss)
            torch.save(model_dict, directory_model_with_lowest_weighted_class_prediction_loss)
        else:
            stats['lines'][e][stats['headers']['Is_Lower_Validation_Weighted_Class_Prediction_Loss']] = 0

        stats['lines'][e][stats['headers']['Lowest_Validation_Weighted_Class_Prediction_Loss']] = (
            lowest_weighted_class_prediction_loss)

        if accuracy_e > highest_accuracy:
            highest_accuracy = accuracy_e
            highest_accuracy_str = cp_strings.format_float_to_str(
                highest_accuracy, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Higher_Accuracy']] = 1
            # is_successful_epoch = True

            if os.path.isfile(directory_model_with_highest_accuracy):
                os.remove(directory_model_with_highest_accuracy)
            torch.save(model_dict, directory_model_with_highest_accuracy)
        else:
            stats['lines'][e][stats['headers']['Is_Higher_Accuracy']] = 0

        stats['lines'][e][stats['headers']['Highest_Validation_Accuracy']] = highest_accuracy

        if is_successful_epoch:
            i = 0
        else:
            i += 1
        stats['lines'][e][stats['headers']['Unsuccessful_Epochs']] = i

        if os.path.isfile(directory_stats):
            os.remove(directory_stats)

        cp_txt.lines_to_csv_file(stats['lines'], directory_stats, stats['headers'])

        unweighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            unweighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        weighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            weighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        accuracy_str_e = cp_strings.format_float_to_str(accuracy_e, n_decimals=n_decimals_for_printing)

        print(
            'Epoch: {e:d}. Validation. Unweighted Classification Loss: {class_prediction_loss:s}. Accuracy: {accuracy:s}.'.format(
                e=e, class_prediction_loss=unweighted_class_prediction_loss_str_e, accuracy=accuracy_str_e))

        print('Epoch {e:d} - Unsuccessful Epochs {i:d}.'.format(e=e, i=i))

        print(dashes)

        e += 1

    print()

    E = e

    time_training = cp_timer.get_delta_time()

    print('Training completed in {d} days {h} hours {m} minutes {s} seconds'.format(
        d=time_training.days, h=time_training.hours,
        m=time_training.minutes, s=time_training.seconds))
    print('Number of Epochs: {E:d}'.format(E=E))
    print('Lowest Unweighted Classification Loss: {:s}'.format(lowest_unweighted_class_prediction_loss_str))
    print('Lowest Weighted Classification Loss: {:s}'.format(lowest_weighted_class_prediction_loss_str))
    print('Highest Accuracy: {:s}'.format(highest_accuracy_str))

    return None


def proactive_feature_sequence_classifiers(
        model, loader, delta_preprocessor, optimizer, scheduler, I=10, E=None, T=None,
        epsilon_start=.9, epsilon_end=.2, epsilon_step=-.1, directory_outputs=None):
    cp_timer = cp_clock.Timer()

    if model.training:
        model.eval()
    model.freeze()
    torch.set_grad_enabled(False)

    for key_loader_k in loader.keys():
        if key_loader_k == 'training' or key_loader_k == 'validation':
            pass
        else:
            raise ValueError('Unknown keys in loader')

    headers = [
        'Epoch', 'Unsuccessful_Epochs',

        'Training_Unweighted_Loss',
        'Training_Weighted_Loss',

        'Training_Unweighted_Value_Action_Loss',
        'Training_Weighted_Value_Action_Loss',

        'Training_Unweighted_Class_Prediction_Loss',
        'Training_Weighted_Class_Prediction_Loss',
        'Training_Accuracy',

        'Training_Unweighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Training_Weighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Training_Accuracy_In_Last_Time_Point',

        'Training_Unweighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Training_Weighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Training_Accuracy_In_Each_Time_Point',

        'Validation_Unweighted_Loss',
        'Validation_Weighted_Loss',

        'Validation_Unweighted_Value_Action_Loss',
        'Validation_Weighted_Value_Action_Loss',

        'Validation_Unweighted_Class_Prediction_Loss',
        'Lowest_Validation_Unweighted_Class_Prediction_Loss',
        'Is_Lower_Validation_Unweighted_Class_Prediction_Loss',

        'Validation_Weighted_Class_Prediction_Loss',
        'Lowest_Validation_Weighted_Class_Prediction_Loss',
        'Is_Lower_Validation_Weighted_Class_Prediction_Loss',

        'Validation_Accuracy',
        'Highest_Validation_Accuracy',
        'Is_Higher_Accuracy',

        'Validation_Unweighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Validation_Weighted_Class_Prediction_Loss_In_Last_Time_Point',
        'Validation_Accuracy_In_Last_Time_Point',

        'Validation_Unweighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Validation_Weighted_Class_Prediction_Losses_In_Each_Time_Point',
        'Validation_Accuracy_In_Each_Time_Point']

    n_columns = len(headers)
    new_line_stats = [None for i in range(0, n_columns, 1)]  # type: list

    stats = {
        'headers': {headers[k]: k for k in range(n_columns)},
        'n_columns': n_columns,
        'lines': []}

    if directory_outputs is None:
        directory_outputs = 'outputs'
    os.makedirs(directory_outputs, exist_ok=True)

    directory_model_at_last_epoch = os.path.join(directory_outputs, 'model_at_last_epoch.pth')

    directory_model_with_lowest_unweighted_class_prediction_loss = os.path.join(
        directory_outputs, 'model_with_lowest_unweighted_class_prediction_loss.pth')

    directory_model_with_lowest_weighted_class_prediction_loss = os.path.join(
        directory_outputs, 'model_with_lowest_weighted_class_prediction_loss.pth')

    directory_model_with_highest_accuracy = os.path.join(directory_outputs, 'model_with_highest_accuracy.pth')

    directory_stats = os.path.join(directory_outputs, 'stats.csv')

    separators_times = '  '

    n_decimals_for_printing = 6
    n_dashes = 150
    dashes = '-' * n_dashes
    print(dashes)

    replay_memory = ReplayMemory(
        axis_time_features=model.axis_time_inputs, axis_time_actions=model.axis_time_losses,
        axis_models_actions=model.axis_models_losses)

    lowest_unweighted_class_prediction_loss = math.inf
    lowest_unweighted_class_prediction_loss_str = str(lowest_unweighted_class_prediction_loss)

    lowest_weighted_class_prediction_loss = math.inf
    lowest_weighted_class_prediction_loss_str = str(lowest_weighted_class_prediction_loss)

    highest_accuracy = -math.inf
    highest_accuracy_str = str(highest_accuracy)

    if E is None:
        E = math.inf

    if I is None:
        I = math.inf

    if T is None:
        T = math.inf

    epsilon = epsilon_start
    if epsilon < epsilon_end:
        epsilon = epsilon_end

    epsilon_validation = 0

    i = 0
    e = 0

    while (e < E) and (i < I):

        print('Epoch {e} ...'.format(e=e))

        stats['lines'].append(new_line_stats.copy())
        stats['lines'][e][stats['headers']['Epoch']] = e

        # Each epoch has a training and a validation phase
        # training phase

        running_unweighted_loss_e = 0.0
        running_weighted_loss_e = 0.0

        running_n_selected_actions_e = 0
        running_unweighted_value_action_loss_e = 0.0
        running_weighted_value_action_loss_e = 0.0

        running_n_corrects_e = 0
        running_n_classifications_e = 0
        running_unweighted_class_prediction_loss_e = 0.0
        running_weighted_class_prediction_loss_e = 0.0

        running_n_corrects_T_e = 0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_n_classifications_T_e = 0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_unweighted_class_prediction_losses_T_e = 0.0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_weighted_class_prediction_losses_T_e = 0.0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]

        b = 0
        # Iterate over data.
        for environments_eb in loader['training']:

            replay_memory.clear()

            hc_ebt = None, None

            t = 0
            for state_ebt, labels_ebt in environments_eb:

                outs_ebt, hc_ebt = model(x=state_ebt, hc=hc_ebt)

                values_actions_ebt, predictions_classes_ebt = model.split(outs_ebt)

                action_ebt = model.sample_action(values_actions=values_actions_ebt, epsilon=epsilon)

                rewards_ebt = None

                replay_memory.put(
                    states=state_ebt, states_labels=labels_ebt, actions=action_ebt,
                    next_states=None, rewards=rewards_ebt)

                if t > 0:
                    class_prediction_losses_ebt = model.compute_class_prediction_losses(
                        predictions_classes=predictions_classes_ebt, labels=labels_ebt)

                    replay_memory.rewards[t - 1] = model.get_previous_rewards(
                        class_prediction_losses=class_prediction_losses_ebt)

                delta_ebt = model.compute_deltas(action_ebt)

                if delta_preprocessor is not None:
                    delta_ebt = delta_preprocessor(delta_ebt)

                environments_eb.step(delta_ebt)

                t += 1

                if t >= T:
                    break

            replay_memory.actions[-1] = None
            # replay_memory.actions.pop()
            # replay_memory.rewards.pop()

            samples_eb = replay_memory.sample()
            states_eb = samples_eb['states']
            states_labels_eb = samples_eb['states_labels']
            actions_eb = samples_eb['actions']
            next_states_eb = samples_eb['next_states']
            rewards_eb = samples_eb['rewards']
            # non_final_eb = samples_eb['non_final']

            next_outs_eb, next_hc_eb = model(x=next_states_eb)

            # todo: set rewards_eb to -next_predictions_classes_eb
            next_values_actions_eb, next_predictions_classes_eb = model.split(next_outs_eb)

            expected_values_actions_eb = model.compute_expected_values_actions(
                next_values_actions=next_values_actions_eb, rewards=rewards_eb)

            optimizer.zero_grad()

            # forward
            # track history
            torch.set_grad_enabled(True)
            model.unfreeze()
            model.train()

            outs_eb, hc_eb = model(x=states_eb)
            values_actions_eb, predictions_classes_eb = model.split(outs_eb)

            values_actions_eb = model.remove_last_values_actions(values_actions=values_actions_eb)

            values_selected_actions_eb = model.gather_values_selected_actions(
                values_actions=values_actions_eb, actions=actions_eb)

            value_action_losses_eb = model.compute_value_action_losses(
                values_selected_actions=values_selected_actions_eb, expected_values_actions=expected_values_actions_eb)

            class_prediction_losses_eb = model.compute_class_prediction_losses(
                predictions_classes=predictions_classes_eb, labels=states_labels_eb)

            weighted_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                weighted=True, loss_weights_actors=None, format_weights=False)

            weighted_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                weighted=True, loss_weights_classifiers=None, format_weights=False)

            weighted_loss_eb = model.compute_multitask_losses(
                value_action_loss=weighted_value_action_loss_eb,
                class_prediction_loss=weighted_class_prediction_loss_eb, weighted=True)

            weighted_loss_eb.backward()
            optimizer.step()

            model.eval()
            model.freeze()
            torch.set_grad_enabled(False)

            unweighted_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                weighted=False, loss_weights_actors=None, format_weights=False)

            unweighted_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                weighted=False, loss_weights_classifiers=None, format_weights=False)

            unweighted_loss_eb = model.compute_multitask_losses(
                value_action_loss=unweighted_value_action_loss_eb,
                class_prediction_loss=unweighted_class_prediction_loss_eb, weighted=False)

            n_selected_actions_eb = model.compute_n_selected_actions(
                selected_actions=actions_eb, axes_not_included=None)

            # compute accuracy
            classifications_eb = model.compute_classifications(predictions_classes=predictions_classes_eb)
            correct_classifications_eb = model.compute_correct_classifications(
                classifications=classifications_eb, labels=states_labels_eb)
            n_corrects_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb, axes_not_included=None, keepdim=False)
            n_classifications_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=None)
            n_actions_and_classifications_eb = n_selected_actions_eb + n_classifications_eb

            running_unweighted_loss_e += (unweighted_loss_eb.item() * n_actions_and_classifications_eb)
            running_weighted_loss_e += (weighted_loss_eb.item() * n_actions_and_classifications_eb)

            running_n_selected_actions_e += n_selected_actions_eb
            running_unweighted_value_action_loss_e += (unweighted_value_action_loss_eb.item() * n_selected_actions_eb)
            running_weighted_value_action_loss_e += (weighted_value_action_loss_eb.item() * n_selected_actions_eb)

            running_n_corrects_e += n_corrects_eb
            running_n_classifications_e += n_classifications_eb
            running_unweighted_class_prediction_loss_e += (
                        unweighted_class_prediction_loss_eb.item() * n_classifications_eb)
            running_weighted_class_prediction_loss_e += (
                        weighted_class_prediction_loss_eb.item() * n_classifications_eb)

            # compute accuracy for each time point
            n_corrects_T_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb,
                axes_not_included=model.axis_time_losses, keepdim=False)
            n_classifications_T_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=model.axis_time_losses)

            # compute class prediction losses for each time point
            unweighted_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                weighted=False, loss_weights_classifiers=None, format_weights=False)

            weighted_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                weighted=True, loss_weights_classifiers=None, format_weights=False)

            running_n_corrects_T_e += n_corrects_T_eb
            running_n_classifications_T_e += n_classifications_T_eb
            running_unweighted_class_prediction_losses_T_e += (
                        unweighted_class_prediction_losses_T_eb * n_classifications_T_eb)
            running_weighted_class_prediction_losses_T_e += (
                        weighted_class_prediction_losses_T_eb * n_classifications_T_eb)

            b += 1

        # scheduler.step()

        running_n_actions_and_classifications_e = running_n_selected_actions_e + running_n_classifications_e
        unweighted_loss_e = running_unweighted_loss_e / running_n_actions_and_classifications_e
        weighted_loss_e = running_weighted_loss_e / running_n_actions_and_classifications_e

        unweighted_value_action_loss_e = running_unweighted_value_action_loss_e / running_n_selected_actions_e
        weighted_value_action_loss_e = running_weighted_value_action_loss_e / running_n_selected_actions_e

        unweighted_class_prediction_loss_e = running_unweighted_class_prediction_loss_e / running_n_classifications_e
        weighted_class_prediction_loss_e = running_weighted_class_prediction_loss_e / running_n_classifications_e
        accuracy_e = running_n_corrects_e / running_n_classifications_e

        unweighted_class_prediction_losses_T_e = (
                    running_unweighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        weighted_class_prediction_losses_T_e = (
                    running_weighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        accuracy_T_e = (running_n_corrects_T_e / running_n_classifications_T_e)

        last_unweighted_class_prediction_loss_e = unweighted_class_prediction_losses_T_e[-1].item()
        last_weighted_class_prediction_loss_e = weighted_class_prediction_losses_T_e[-1].item()
        last_accuracy_e = accuracy_T_e[-1].item()

        stats['lines'][e][stats['headers']['Training_Unweighted_Loss']] = unweighted_loss_e
        stats['lines'][e][stats['headers']['Training_Weighted_Loss']] = weighted_loss_e

        stats['lines'][e][stats['headers']['Training_Unweighted_Value_Action_Loss']] = unweighted_value_action_loss_e
        stats['lines'][e][stats['headers']['Training_Weighted_Value_Action_Loss']] = weighted_value_action_loss_e

        stats['lines'][e][stats['headers']['Training_Unweighted_Class_Prediction_Loss']] = (
            unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Weighted_Class_Prediction_Loss']] = (
            weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Accuracy']] = accuracy_e

        stats['lines'][e][stats['headers']['Training_Unweighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Weighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Accuracy_In_Last_Time_Point']] = last_accuracy_e

        stats['lines'][e][stats['headers']['Training_Unweighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in unweighted_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Training_Weighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in weighted_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Training_Accuracy_In_Each_Time_Point']] = separators_times.join(
            [str(t) for t in accuracy_T_e.tolist()])

        unweighted_loss_str_e = cp_strings.format_float_to_str(
            unweighted_loss_e, n_decimals=n_decimals_for_printing)
        weighted_loss_str_e = cp_strings.format_float_to_str(weighted_loss_e, n_decimals=n_decimals_for_printing)

        unweighted_value_action_loss_str_e = cp_strings.format_float_to_str(
            unweighted_value_action_loss_e, n_decimals=n_decimals_for_printing)

        weighted_value_action_loss_str_e = cp_strings.format_float_to_str(
            weighted_value_action_loss_e, n_decimals=n_decimals_for_printing)

        unweighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            unweighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        weighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            weighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        accuracy_str_e = cp_strings.format_float_to_str(accuracy_e, n_decimals=n_decimals_for_printing)

        print(
            'Epoch: {e:d}. Training. Unweighted Value Action Loss: {action_loss:s}. Unweighted Classification Loss: {class_prediction_loss:s}. Accuracy: {accuracy:s}.'.format(
                e=e, action_loss=unweighted_value_action_loss_str_e,
                class_prediction_loss=unweighted_class_prediction_loss_str_e, accuracy=accuracy_str_e))

        epsilon = epsilon + epsilon_step
        if epsilon < epsilon_end:
            epsilon = epsilon_end

        # validation phase

        running_unweighted_loss_e = 0.0
        running_weighted_loss_e = 0.0

        running_n_selected_actions_e = 0
        running_unweighted_value_action_loss_e = 0.0
        running_weighted_value_action_loss_e = 0.0

        running_n_corrects_e = 0
        running_n_classifications_e = 0
        running_unweighted_class_prediction_loss_e = 0.0
        running_weighted_class_prediction_loss_e = 0.0

        running_n_corrects_T_e = 0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_n_classifications_T_e = 0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_unweighted_class_prediction_losses_T_e = 0.0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]
        running_weighted_class_prediction_losses_T_e = 0.0  # type: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor]

        b = 0
        # Iterate over data.
        for environments_eb in loader['validation']:

            replay_memory.clear()

            hc_ebt = None, None

            t = 0
            for state_ebt, labels_ebt in environments_eb:

                outs_ebt, hc_ebt = model(x=state_ebt, hc=hc_ebt)

                values_actions_ebt, predictions_classes_ebt = model.split(outs_ebt)

                action_ebt = model.sample_action(values_actions=values_actions_ebt, epsilon=epsilon_validation)

                rewards_ebt = None

                replay_memory.put(
                    states=state_ebt, states_labels=labels_ebt, actions=action_ebt,
                    next_states=None, rewards=rewards_ebt)

                if t > 0:
                    class_prediction_losses_ebt = model.compute_class_prediction_losses(
                        predictions_classes=predictions_classes_ebt, labels=labels_ebt)

                    replay_memory.rewards[t - 1] = model.get_previous_rewards(
                        class_prediction_losses=class_prediction_losses_ebt)

                delta_ebt = model.compute_deltas(action_ebt)

                if delta_preprocessor is not None:
                    delta_ebt = delta_preprocessor(delta_ebt)

                environments_eb.step(delta_ebt)

                t += 1

                if t >= T:
                    break

            replay_memory.actions[-1] = None
            # replay_memory.actions.pop()
            # replay_memory.rewards.pop()

            samples_eb = replay_memory.sample()
            states_eb = samples_eb['states']
            states_labels_eb = samples_eb['states_labels']
            actions_eb = samples_eb['actions']
            next_states_eb = samples_eb['next_states']
            rewards_eb = samples_eb['rewards']
            # non_final_eb = samples_eb['non_final']

            next_outs_eb, next_hc_eb = model(x=next_states_eb)
            next_values_actions_eb, next_predictions_classes_eb = model.split(next_outs_eb)

            expected_values_actions_eb = model.compute_expected_values_actions(
                next_values_actions=next_values_actions_eb, rewards=rewards_eb)

            # forward

            outs_eb, hc_eb = model(x=states_eb)
            values_actions_eb, predictions_classes_eb = model.split(outs_eb)

            values_actions_eb = model.remove_last_values_actions(values_actions=values_actions_eb)

            values_selected_actions_eb = model.gather_values_selected_actions(
                values_actions=values_actions_eb, actions=actions_eb)

            value_action_losses_eb = model.compute_value_action_losses(
                values_selected_actions=values_selected_actions_eb, expected_values_actions=expected_values_actions_eb)

            class_prediction_losses_eb = model.compute_class_prediction_losses(
                predictions_classes=predictions_classes_eb, labels=states_labels_eb)

            weighted_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                weighted=True, loss_weights_actors=None, format_weights=False)

            weighted_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                weighted=True, loss_weights_classifiers=None, format_weights=False)

            weighted_loss_eb = model.compute_multitask_losses(
                value_action_loss=weighted_value_action_loss_eb,
                class_prediction_loss=weighted_class_prediction_loss_eb, weighted=True)

            unweighted_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                weighted=False, loss_weights_actors=None, format_weights=False)

            unweighted_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                weighted=False, loss_weights_classifiers=None, format_weights=False)

            unweighted_loss_eb = model.compute_multitask_losses(
                value_action_loss=unweighted_value_action_loss_eb,
                class_prediction_loss=unweighted_class_prediction_loss_eb, weighted=False)

            n_selected_actions_eb = model.compute_n_selected_actions(
                selected_actions=actions_eb, axes_not_included=None)

            # compute accuracy
            classifications_eb = model.compute_classifications(predictions_classes=predictions_classes_eb)
            correct_classifications_eb = model.compute_correct_classifications(
                classifications=classifications_eb, labels=states_labels_eb)
            n_corrects_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb, axes_not_included=None, keepdim=False)
            n_classifications_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=None)
            n_actions_and_classifications_eb = n_selected_actions_eb + n_classifications_eb

            running_unweighted_loss_e += (unweighted_loss_eb.item() * n_actions_and_classifications_eb)
            running_weighted_loss_e += (weighted_loss_eb.item() * n_actions_and_classifications_eb)

            running_n_selected_actions_e += n_selected_actions_eb
            running_unweighted_value_action_loss_e += (unweighted_value_action_loss_eb.item() * n_selected_actions_eb)
            running_weighted_value_action_loss_e += (weighted_value_action_loss_eb.item() * n_selected_actions_eb)

            running_n_corrects_e += n_corrects_eb
            running_n_classifications_e += n_classifications_eb
            running_unweighted_class_prediction_loss_e += (
                        unweighted_class_prediction_loss_eb.item() * n_classifications_eb)
            running_weighted_class_prediction_loss_e += (
                        weighted_class_prediction_loss_eb.item() * n_classifications_eb)

            # compute accuracy for each time point
            n_corrects_T_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb,
                axes_not_included=model.axis_time_losses, keepdim=False)
            n_classifications_T_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=model.axis_time_losses)

            # compute class prediction losses for each time point
            unweighted_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                weighted=False, loss_weights_classifiers=None, format_weights=False)

            weighted_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                weighted=True, loss_weights_classifiers=None, format_weights=False)

            running_n_corrects_T_e += n_corrects_T_eb
            running_n_classifications_T_e += n_classifications_T_eb
            running_unweighted_class_prediction_losses_T_e += (
                        unweighted_class_prediction_losses_T_eb * n_classifications_T_eb)
            running_weighted_class_prediction_losses_T_e += (
                        weighted_class_prediction_losses_T_eb * n_classifications_T_eb)

            b += 1

        running_n_actions_and_classifications_e = running_n_selected_actions_e + running_n_classifications_e
        unweighted_loss_e = running_unweighted_loss_e / running_n_actions_and_classifications_e
        weighted_loss_e = running_weighted_loss_e / running_n_actions_and_classifications_e

        unweighted_value_action_loss_e = running_unweighted_value_action_loss_e / running_n_selected_actions_e
        weighted_value_action_loss_e = running_weighted_value_action_loss_e / running_n_selected_actions_e

        unweighted_class_prediction_loss_e = running_unweighted_class_prediction_loss_e / running_n_classifications_e
        weighted_class_prediction_loss_e = running_weighted_class_prediction_loss_e / running_n_classifications_e
        accuracy_e = running_n_corrects_e / running_n_classifications_e

        unweighted_class_prediction_losses_T_e = (
                    running_unweighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        weighted_class_prediction_losses_T_e = (
                    running_weighted_class_prediction_losses_T_e / running_n_classifications_T_e)
        accuracy_T_e = (running_n_corrects_T_e / running_n_classifications_T_e)

        last_unweighted_class_prediction_loss_e = unweighted_class_prediction_losses_T_e[-1].item()
        last_weighted_class_prediction_loss_e = weighted_class_prediction_losses_T_e[-1].item()
        last_accuracy_e = accuracy_T_e[-1].item()

        stats['lines'][e][stats['headers']['Validation_Unweighted_Loss']] = unweighted_loss_e
        stats['lines'][e][stats['headers']['Validation_Weighted_Loss']] = weighted_loss_e

        stats['lines'][e][stats['headers']['Validation_Unweighted_Value_Action_Loss']] = unweighted_value_action_loss_e
        stats['lines'][e][stats['headers']['Validation_Weighted_Value_Action_Loss']] = weighted_value_action_loss_e

        stats['lines'][e][stats['headers']['Validation_Unweighted_Class_Prediction_Loss']] = (
            unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Weighted_Class_Prediction_Loss']] = (
            weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Accuracy']] = accuracy_e

        stats['lines'][e][stats['headers']['Validation_Unweighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_unweighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Weighted_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_weighted_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Accuracy_In_Last_Time_Point']] = last_accuracy_e

        stats['lines'][e][stats['headers']['Validation_Unweighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in unweighted_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Validation_Weighted_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in weighted_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Validation_Accuracy_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in accuracy_T_e.tolist()]))

        model_dict = copy.deepcopy(model.state_dict())
        if os.path.isfile(directory_model_at_last_epoch):
            os.remove(directory_model_at_last_epoch)
        torch.save(model_dict, directory_model_at_last_epoch)

        is_successful_epoch = False

        if unweighted_class_prediction_loss_e < lowest_unweighted_class_prediction_loss:

            lowest_unweighted_class_prediction_loss = unweighted_class_prediction_loss_e
            lowest_unweighted_class_prediction_loss_str = cp_strings.format_float_to_str(
                lowest_unweighted_class_prediction_loss, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Lower_Validation_Unweighted_Class_Prediction_Loss']] = 1
            is_successful_epoch = True

            if os.path.isfile(directory_model_with_lowest_unweighted_class_prediction_loss):
                os.remove(directory_model_with_lowest_unweighted_class_prediction_loss)
            torch.save(model_dict, directory_model_with_lowest_unweighted_class_prediction_loss)
        else:
            stats['lines'][e][stats['headers']['Is_Lower_Validation_Unweighted_Class_Prediction_Loss']] = 0

        stats['lines'][e][stats['headers']['Lowest_Validation_Unweighted_Class_Prediction_Loss']] = (
            lowest_unweighted_class_prediction_loss)

        if weighted_class_prediction_loss_e < lowest_weighted_class_prediction_loss:

            lowest_weighted_class_prediction_loss = weighted_class_prediction_loss_e
            lowest_weighted_class_prediction_loss_str = cp_strings.format_float_to_str(
                lowest_weighted_class_prediction_loss, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Lower_Validation_Weighted_Class_Prediction_Loss']] = 1
            is_successful_epoch = True

            if os.path.isfile(directory_model_with_lowest_weighted_class_prediction_loss):
                os.remove(directory_model_with_lowest_weighted_class_prediction_loss)
            torch.save(model_dict, directory_model_with_lowest_weighted_class_prediction_loss)
        else:
            stats['lines'][e][stats['headers']['Is_Lower_Validation_Weighted_Class_Prediction_Loss']] = 0

        stats['lines'][e][stats['headers']['Lowest_Validation_Weighted_Class_Prediction_Loss']] = (
            lowest_weighted_class_prediction_loss)

        if accuracy_e > highest_accuracy:
            highest_accuracy = accuracy_e
            highest_accuracy_str = cp_strings.format_float_to_str(
                highest_accuracy, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Higher_Accuracy']] = 1
            # is_successful_epoch = True

            if os.path.isfile(directory_model_with_highest_accuracy):
                os.remove(directory_model_with_highest_accuracy)
            torch.save(model_dict, directory_model_with_highest_accuracy)
        else:
            stats['lines'][e][stats['headers']['Is_Higher_Accuracy']] = 0

        stats['lines'][e][stats['headers']['Highest_Validation_Accuracy']] = highest_accuracy

        if is_successful_epoch:
            i = 0
        else:
            i += 1
        stats['lines'][e][stats['headers']['Unsuccessful_Epochs']] = i

        if os.path.isfile(directory_stats):
            os.remove(directory_stats)

        cp_txt.lines_to_csv_file(stats['lines'], directory_stats, stats['headers'])

        unweighted_loss_str_e = cp_strings.format_float_to_str(
            unweighted_loss_e, n_decimals=n_decimals_for_printing)
        weighted_loss_str_e = cp_strings.format_float_to_str(weighted_loss_e, n_decimals=n_decimals_for_printing)

        unweighted_value_action_loss_str_e = cp_strings.format_float_to_str(
            unweighted_value_action_loss_e, n_decimals=n_decimals_for_printing)

        weighted_value_action_loss_str_e = cp_strings.format_float_to_str(
            weighted_value_action_loss_e, n_decimals=n_decimals_for_printing)

        unweighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            unweighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        weighted_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            weighted_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        accuracy_str_e = cp_strings.format_float_to_str(accuracy_e, n_decimals=n_decimals_for_printing)

        print(
            'Epoch: {e:d}. Validation. Unweighted Value Action Loss: {action_loss:s}. Unweighted Classification Loss: {class_prediction_loss:s}. Accuracy: {accuracy:s}.'.format(
                e=e, action_loss=unweighted_value_action_loss_str_e,
                class_prediction_loss=unweighted_class_prediction_loss_str_e, accuracy=accuracy_str_e))

        print('Epoch {e:d} - Unsuccessful Epochs {i:d}.'.format(e=e, i=i))

        print(dashes)

        e += 1

    print()

    E = e

    time_training = cp_timer.get_delta_time()

    print('Training completed in {d} days {h} hours {m} minutes {s} seconds'.format(
        d=time_training.days, h=time_training.hours,
        m=time_training.minutes, s=time_training.seconds))
    print('Number of Epochs: {E:d}'.format(E=E))
    print('Lowest Unweighted Classification Loss: {:s}'.format(lowest_unweighted_class_prediction_loss_str))
    print('Lowest Weighted Classification Loss: {:s}'.format(lowest_weighted_class_prediction_loss_str))
    print('Highest Accuracy: {:s}'.format(highest_accuracy_str))

    return None


class ReplayMemory:
    """A simple numpy replay buffer."""

    def __init__(self, axis_time_features: int, axis_time_actions: int, axis_models_actions: int):

        self.states = []
        self.states_labels = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        # self.non_final = []

        self.axis_time_features = axis_time_features
        self.axis_time_actions = axis_time_actions
        if axis_models_actions < self.axis_time_actions:
            self.axis_time_rewards = self.axis_time_actions - 1
        else:
            self.axis_time_rewards = self.axis_time_actions

    def put(self, states=None, states_labels=None, actions=None, next_states=None, rewards=None):  # , non_final):

        self.states.append(states)
        self.states_labels.append(states_labels)
        self.actions.append(actions)

        self.next_states.append(next_states)

        self.rewards.append(rewards)

        # if non_final is not None:
        # self.non_final.append(non_final)

    def clear(self):

        self.states = []
        self.states_labels = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        # self.non_final = []

        return None

    def sample(self):

        T = len(self.states)

        states = torch.cat([s for s in self.states if s is not None], dim=self.axis_time_features)

        states_labels = torch.cat(
            [self.states_labels[t] for t in range(0, T, 1) if self.states_labels[t] is not None],
            dim=self.axis_time_actions)

        actions = torch.cat(
            [self.actions[t] for t in range(0, T, 1) if self.actions[t] is not None], dim=self.axis_time_actions)

        next_states = [n for n in self.next_states if n is not None]
        if len(next_states) == 0:
            ind = tuple(
                [slice(0, states.shape[d], 1) if d != self.axis_time_features else slice(1, T, 1)
                 for d in range(0, states.ndim, 1)])
            next_states = states[ind]
        else:
            next_states = torch.cat(next_states, dim=self.axis_time_features)

        rewards = torch.cat([r for r in self.rewards if r is not None], dim=self.axis_time_rewards)

        # non_final = torch.cat(self.non_final, dim=self.axis_time)

        return dict(
            states=states, states_labels=states_labels, actions=actions,
            next_states=next_states, rewards=rewards)  # , non_final=non_final)

    def __len__(self) -> int:
        return len(self.states)
