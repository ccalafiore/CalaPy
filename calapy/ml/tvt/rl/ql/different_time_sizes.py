

import os
import copy
import math
import numpy as np
import torch
from ..... import clock as cp_clock
from ..... import strings as cp_strings
from ..... import txt as cp_txt
from ....tvt import utilities as cp_tvt_utilities
from ...rl import utilities as cp_rl_utilities


def train(
        model, environment, optimizer, U=10, E=None,
        tot_observations_per_epoch=None,
        epsilon_start=.95, epsilon_end=.05, epsilon_step=-.05, n_samples_per_epsilon_step=10000,  # todo: to the model
        directory_outputs=None):

    cp_timer = cp_clock.Timer()

    phases_names = ('training', 'validation')
    for key_environment_k in environment.keys():
        if key_environment_k in phases_names:
            pass
        else:
            raise ValueError('Unknown keys in environment')

    if tot_observations_per_epoch is None:
        tot_observations_per_epoch = {'training': 10000, 'validation': 10000}
    elif isinstance(tot_observations_per_epoch, dict):
        for key_k in tot_observations_per_epoch.keys():
            if key_k in phases_names:
                pass
            else:
                raise ValueError('Unknown keys in tot_observations_per_epoch')
    else:
        raise TypeError('tot_observations_per_epoch')

    if model.training:
        model.eval()
    model.freeze()
    torch.set_grad_enabled(False)

    headers = [
        'Start_Date', 'Start_Time' 'Duration', 'Elapsed_Time',

        'Epoch', 'Unsuccessful_Epochs',

        'Training_Unscaled_Loss',
        'Validation_Unscaled_Loss',
        'Lowest_Validation_Unscaled_Loss',
        'Is_Lower_Validation_Unscaled_Loss',

        'Training_Scaled_Loss',
        'Validation_Scaled_Loss',
        'Lowest_Validation_Scaled_Loss',
        'Is_Lower_Validation_Scaled_Loss'
    ]

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

    directory_model_with_lowest_unscaled_loss = os.path.join(directory_outputs, 'model_with_lowest_unscaled_loss.pth')

    directory_model_with_lowest_scaled_loss = os.path.join(directory_outputs, 'model_with_lowest_scaled_loss.pth')

    directory_model_with_highest_reward = os.path.join(directory_outputs, 'model_with_highest_reward.pth')

    directory_stats = os.path.join(directory_outputs, 'stats.csv')

    n_decimals_for_printing = 6
    n_dashes = 150
    dashes = '-' * n_dashes
    print(dashes)

    replay_memory = ReplayMemory()

    lowest_unscaled_loss = math.inf
    lowest_unscaled_loss_str = str(lowest_unscaled_loss)

    lowest_scaled_loss = math.inf
    lowest_scaled_loss_str = str(lowest_scaled_loss)

    highest_reward = -math.inf
    highest_reward_str = str(highest_reward)

    epsilon = epsilon_start  # todo to the model
    if epsilon < epsilon_end:
        epsilon = epsilon_end

    epsilon_validation = 0

    epochs = cp_tvt_utilities.Epochs(U=U, E=E)

    for e, u in epochs:

        print('Epoch {e} ...'.format(e=e))

        stats['lines'].append(new_line_stats.copy())
        stats['lines'][e][stats['headers']['Epoch']] = e

        # Each Training Epoch has a training and a validation phase
        # training phase

        running_unscaled_loss_e = 0.0
        running_scaled_loss_e = 0.0

        # running_n_selected_actions_e = 0  # todo

        env_iterator = cp_rl_utilities.EnvironmentsIterator(
            tot_observations_per_epoch=tot_observations_per_epoch['training'])



        for environment_eb in env_iterator:

            hc_ebit = None, None  # todo

            t = 0
            for observations_ebit in environment_eb:

                state_ebit = observations_ebit, hc_ebit

                outs_ebit, hc_ebit = model(x=state_ebit)

                action_ebit = model.sample_action(values_actions=values_actions_ebit, epsilon=epsilon)

                rewards_ebt = None

                replay_memory.put(state=state_ebt, action=action_ebt, next_state=None, reward=rewards_ebt)

                delta_ebt = model.compute_deltas(action_ebt)

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

            scaled_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                scaled=True, loss_scales_actors=None, format_scales=False)

            scaled_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                scaled=True, loss_scales_classifiers=None, format_scales=False)

            scaled_loss_eb = model.compute_multitask_losses(
                value_action_loss=scaled_value_action_loss_eb,
                class_prediction_loss=scaled_class_prediction_loss_eb, scaled=True)

            scaled_loss_eb.backward()
            optimizer.step()

            model.eval()
            model.freeze()
            torch.set_grad_enabled(False)

            unscaled_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                scaled=False, loss_scales_actors=None, format_scales=False)

            unscaled_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                scaled=False, loss_scales_classifiers=None, format_scales=False)

            unscaled_loss_eb = model.compute_multitask_losses(
                value_action_loss=unscaled_value_action_loss_eb,
                class_prediction_loss=unscaled_class_prediction_loss_eb, scaled=False)

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

            running_unscaled_loss_e += (unscaled_loss_eb.item() * n_actions_and_classifications_eb)
            running_scaled_loss_e += (scaled_loss_eb.item() * n_actions_and_classifications_eb)

            running_n_selected_actions_e += n_selected_actions_eb
            running_unscaled_value_action_loss_e += (unscaled_value_action_loss_eb.item() * n_selected_actions_eb)
            running_scaled_value_action_loss_e += (scaled_value_action_loss_eb.item() * n_selected_actions_eb)

            running_n_corrects_e += n_corrects_eb
            running_n_classifications_e += n_classifications_eb
            running_unscaled_class_prediction_loss_e += (
                        unscaled_class_prediction_loss_eb.item() * n_classifications_eb)
            running_scaled_class_prediction_loss_e += (
                        scaled_class_prediction_loss_eb.item() * n_classifications_eb)

            # compute accuracy for each time point
            n_corrects_T_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb,
                axes_not_included=model.axis_time_losses, keepdim=False)
            n_classifications_T_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=model.axis_time_losses)

            # compute class prediction losses for each time point
            unscaled_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                scaled=False, loss_scales_classifiers=None, format_scales=False)

            scaled_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                scaled=True, loss_scales_classifiers=None, format_scales=False)

            running_n_corrects_T_e += n_corrects_T_eb
            running_n_classifications_T_e += n_classifications_T_eb
            running_unscaled_class_prediction_losses_T_e += (
                        unscaled_class_prediction_losses_T_eb * n_classifications_T_eb)
            running_scaled_class_prediction_losses_T_e += (
                        scaled_class_prediction_losses_T_eb * n_classifications_T_eb)


        replay_memory.clear()

        # scheduler.step()

        running_n_actions_and_classifications_e = running_n_selected_actions_e + running_n_classifications_e
        unscaled_loss_e = running_unscaled_loss_e / running_n_actions_and_classifications_e
        scaled_loss_e = running_scaled_loss_e / running_n_actions_and_classifications_e

        unscaled_value_action_loss_e = running_unscaled_value_action_loss_e / running_n_selected_actions_e
        scaled_value_action_loss_e = running_scaled_value_action_loss_e / running_n_selected_actions_e

        unscaled_class_prediction_loss_e = running_unscaled_class_prediction_loss_e / running_n_classifications_e
        scaled_class_prediction_loss_e = running_scaled_class_prediction_loss_e / running_n_classifications_e
        accuracy_e = running_n_corrects_e / running_n_classifications_e

        unscaled_class_prediction_losses_T_e = (
                    running_unscaled_class_prediction_losses_T_e / running_n_classifications_T_e)
        scaled_class_prediction_losses_T_e = (
                    running_scaled_class_prediction_losses_T_e / running_n_classifications_T_e)
        accuracy_T_e = (running_n_corrects_T_e / running_n_classifications_T_e)

        last_unscaled_class_prediction_loss_e = unscaled_class_prediction_losses_T_e[-1].item()
        last_scaled_class_prediction_loss_e = scaled_class_prediction_losses_T_e[-1].item()
        last_accuracy_e = accuracy_T_e[-1].item()

        stats['lines'][e][stats['headers']['Training_Unscaled_Loss']] = unscaled_loss_e
        stats['lines'][e][stats['headers']['Training_Scaled_Loss']] = scaled_loss_e

        stats['lines'][e][stats['headers']['Training_Unscaled_Value_Action_Loss']] = unscaled_value_action_loss_e
        stats['lines'][e][stats['headers']['Training_Scaled_Value_Action_Loss']] = scaled_value_action_loss_e

        stats['lines'][e][stats['headers']['Training_Unscaled_Class_Prediction_Loss']] = (
            unscaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Scaled_Class_Prediction_Loss']] = (
            scaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Accuracy']] = accuracy_e

        stats['lines'][e][stats['headers']['Training_Unscaled_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_unscaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Scaled_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_scaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Training_Accuracy_In_Last_Time_Point']] = last_accuracy_e

        stats['lines'][e][stats['headers']['Training_Unscaled_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in unscaled_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Training_Scaled_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in scaled_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Training_Accuracy_In_Each_Time_Point']] = separators_times.join(
            [str(t) for t in accuracy_T_e.tolist()])

        unscaled_loss_str_e = cp_strings.format_float_to_str(
            unscaled_loss_e, n_decimals=n_decimals_for_printing)
        scaled_loss_str_e = cp_strings.format_float_to_str(scaled_loss_e, n_decimals=n_decimals_for_printing)

        unscaled_value_action_loss_str_e = cp_strings.format_float_to_str(
            unscaled_value_action_loss_e, n_decimals=n_decimals_for_printing)

        scaled_value_action_loss_str_e = cp_strings.format_float_to_str(
            scaled_value_action_loss_e, n_decimals=n_decimals_for_printing)

        unscaled_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            unscaled_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        scaled_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            scaled_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        accuracy_str_e = cp_strings.format_float_to_str(accuracy_e, n_decimals=n_decimals_for_printing)

        print(
            'Epoch: {e:d}. Training. Unscaled Value Action Loss: {action_loss:s}. Unscaled Classification Loss: {class_prediction_loss:s}. Accuracy: {accuracy:s}.'.format(
                e=e, action_loss=unscaled_value_action_loss_str_e,
                class_prediction_loss=unscaled_class_prediction_loss_str_e, accuracy=accuracy_str_e))

        epsilon = epsilon + epsilon_step
        if epsilon < epsilon_end:
            epsilon = epsilon_end

        # validation phase

        running_unscaled_loss_e = 0.0
        running_scaled_loss_e = 0.0

        running_n_selected_actions_e = 0
        running_unscaled_value_action_loss_e = 0.0
        running_scaled_value_action_loss_e = 0.0

        running_n_corrects_e = 0
        running_n_classifications_e = 0
        running_unscaled_class_prediction_loss_e = 0.0
        running_scaled_class_prediction_loss_e = 0.0

        running_n_corrects_T_e = 0  # type: int | float | list | tuple | np.ndarray | torch.Tensor
        running_n_classifications_T_e = 0  # type: int | float | list | tuple | np.ndarray | torch.Tensor
        running_unscaled_class_prediction_losses_T_e = 0.0  # type: int | float | list | tuple | np.ndarray | torch.Tensor
        running_scaled_class_prediction_losses_T_e = 0.0  # type: int | float | list | tuple | np.ndarray | torch.Tensor

        b = 0
        # Iterate over data.
        for environments_eb in environment['validation']:

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

            scaled_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                scaled=True, loss_scales_actors=None, format_scales=False)

            scaled_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                scaled=True, loss_scales_classifiers=None, format_scales=False)

            scaled_loss_eb = model.compute_multitask_losses(
                value_action_loss=scaled_value_action_loss_eb,
                class_prediction_loss=scaled_class_prediction_loss_eb, scaled=True)

            unscaled_value_action_loss_eb = model.reduce_value_action_losses(
                value_action_losses=value_action_losses_eb, axes_not_included=None,
                scaled=False, loss_scales_actors=None, format_scales=False)

            unscaled_class_prediction_loss_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=None,
                scaled=False, loss_scales_classifiers=None, format_scales=False)

            unscaled_loss_eb = model.compute_multitask_losses(
                value_action_loss=unscaled_value_action_loss_eb,
                class_prediction_loss=unscaled_class_prediction_loss_eb, scaled=False)

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

            running_unscaled_loss_e += (unscaled_loss_eb.item() * n_actions_and_classifications_eb)
            running_scaled_loss_e += (scaled_loss_eb.item() * n_actions_and_classifications_eb)

            running_n_selected_actions_e += n_selected_actions_eb
            running_unscaled_value_action_loss_e += (unscaled_value_action_loss_eb.item() * n_selected_actions_eb)
            running_scaled_value_action_loss_e += (scaled_value_action_loss_eb.item() * n_selected_actions_eb)

            running_n_corrects_e += n_corrects_eb
            running_n_classifications_e += n_classifications_eb
            running_unscaled_class_prediction_loss_e += (
                        unscaled_class_prediction_loss_eb.item() * n_classifications_eb)
            running_scaled_class_prediction_loss_e += (
                        scaled_class_prediction_loss_eb.item() * n_classifications_eb)

            # compute accuracy for each time point
            n_corrects_T_eb = model.compute_n_corrects(
                correct_classifications=correct_classifications_eb,
                axes_not_included=model.axis_time_losses, keepdim=False)
            n_classifications_T_eb = model.compute_n_classifications(
                classifications=classifications_eb, axes_not_included=model.axis_time_losses)

            # compute class prediction losses for each time point
            unscaled_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                scaled=False, loss_scales_classifiers=None, format_scales=False)

            scaled_class_prediction_losses_T_eb = model.reduce_class_prediction_losses(
                class_prediction_losses=class_prediction_losses_eb, axes_not_included=model.axis_time_losses,
                scaled=True, loss_scales_classifiers=None, format_scales=False)

            running_n_corrects_T_e += n_corrects_T_eb
            running_n_classifications_T_e += n_classifications_T_eb
            running_unscaled_class_prediction_losses_T_e += (
                        unscaled_class_prediction_losses_T_eb * n_classifications_T_eb)
            running_scaled_class_prediction_losses_T_e += (
                        scaled_class_prediction_losses_T_eb * n_classifications_T_eb)

            b += 1

        replay_memory.clear()

        running_n_actions_and_classifications_e = running_n_selected_actions_e + running_n_classifications_e
        unscaled_loss_e = running_unscaled_loss_e / running_n_actions_and_classifications_e
        scaled_loss_e = running_scaled_loss_e / running_n_actions_and_classifications_e

        unscaled_value_action_loss_e = running_unscaled_value_action_loss_e / running_n_selected_actions_e
        scaled_value_action_loss_e = running_scaled_value_action_loss_e / running_n_selected_actions_e

        unscaled_class_prediction_loss_e = running_unscaled_class_prediction_loss_e / running_n_classifications_e
        scaled_class_prediction_loss_e = running_scaled_class_prediction_loss_e / running_n_classifications_e
        accuracy_e = running_n_corrects_e / running_n_classifications_e

        unscaled_class_prediction_losses_T_e = (
                    running_unscaled_class_prediction_losses_T_e / running_n_classifications_T_e)
        scaled_class_prediction_losses_T_e = (
                    running_scaled_class_prediction_losses_T_e / running_n_classifications_T_e)
        accuracy_T_e = (running_n_corrects_T_e / running_n_classifications_T_e)

        last_unscaled_class_prediction_loss_e = unscaled_class_prediction_losses_T_e[-1].item()
        last_scaled_class_prediction_loss_e = scaled_class_prediction_losses_T_e[-1].item()
        last_accuracy_e = accuracy_T_e[-1].item()

        stats['lines'][e][stats['headers']['Validation_Unscaled_Loss']] = unscaled_loss_e
        stats['lines'][e][stats['headers']['Validation_Scaled_Loss']] = scaled_loss_e

        stats['lines'][e][stats['headers']['Validation_Unscaled_Value_Action_Loss']] = unscaled_value_action_loss_e
        stats['lines'][e][stats['headers']['Validation_Scaled_Value_Action_Loss']] = scaled_value_action_loss_e

        stats['lines'][e][stats['headers']['Validation_Unscaled_Class_Prediction_Loss']] = (
            unscaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Scaled_Class_Prediction_Loss']] = (
            scaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Accuracy']] = accuracy_e

        stats['lines'][e][stats['headers']['Validation_Unscaled_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_unscaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Scaled_Class_Prediction_Loss_In_Last_Time_Point']] = (
            last_scaled_class_prediction_loss_e)
        stats['lines'][e][stats['headers']['Validation_Accuracy_In_Last_Time_Point']] = last_accuracy_e

        stats['lines'][e][stats['headers']['Validation_Unscaled_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in unscaled_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Validation_Scaled_Class_Prediction_Losses_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in scaled_class_prediction_losses_T_e.tolist()]))
        stats['lines'][e][stats['headers']['Validation_Accuracy_In_Each_Time_Point']] = (
            separators_times.join([str(t) for t in accuracy_T_e.tolist()]))

        model_dict = copy.deepcopy(model.state_dict())
        if os.path.isfile(directory_model_at_last_epoch):
            os.remove(directory_model_at_last_epoch)
        torch.save(model_dict, directory_model_at_last_epoch)

        is_successful_epoch = False

        if unscaled_class_prediction_loss_e < lowest_unscaled_class_prediction_loss:

            lowest_unscaled_class_prediction_loss = unscaled_class_prediction_loss_e
            lowest_unscaled_class_prediction_loss_str = cp_strings.format_float_to_str(
                lowest_unscaled_class_prediction_loss, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Lower_Validation_Unscaled_Class_Prediction_Loss']] = 1
            is_successful_epoch = True

            if os.path.isfile(directory_model_with_lowest_unscaled_class_prediction_loss):
                os.remove(directory_model_with_lowest_unscaled_class_prediction_loss)
            torch.save(model_dict, directory_model_with_lowest_unscaled_class_prediction_loss)
        else:
            stats['lines'][e][stats['headers']['Is_Lower_Validation_Unscaled_Class_Prediction_Loss']] = 0

        stats['lines'][e][stats['headers']['Lowest_Validation_Unscaled_Class_Prediction_Loss']] = (
            lowest_unscaled_class_prediction_loss)

        if scaled_class_prediction_loss_e < lowest_scaled_class_prediction_loss:

            lowest_scaled_class_prediction_loss = scaled_class_prediction_loss_e
            lowest_scaled_class_prediction_loss_str = cp_strings.format_float_to_str(
                lowest_scaled_class_prediction_loss, n_decimals=n_decimals_for_printing)

            stats['lines'][e][stats['headers']['Is_Lower_Validation_Scaled_Class_Prediction_Loss']] = 1
            is_successful_epoch = True

            if os.path.isfile(directory_model_with_lowest_scaled_class_prediction_loss):
                os.remove(directory_model_with_lowest_scaled_class_prediction_loss)
            torch.save(model_dict, directory_model_with_lowest_scaled_class_prediction_loss)
        else:
            stats['lines'][e][stats['headers']['Is_Lower_Validation_Scaled_Class_Prediction_Loss']] = 0

        stats['lines'][e][stats['headers']['Lowest_Validation_Scaled_Class_Prediction_Loss']] = (
            lowest_scaled_class_prediction_loss)

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

        unscaled_loss_str_e = cp_strings.format_float_to_str(
            unscaled_loss_e, n_decimals=n_decimals_for_printing)
        scaled_loss_str_e = cp_strings.format_float_to_str(scaled_loss_e, n_decimals=n_decimals_for_printing)

        unscaled_value_action_loss_str_e = cp_strings.format_float_to_str(
            unscaled_value_action_loss_e, n_decimals=n_decimals_for_printing)

        scaled_value_action_loss_str_e = cp_strings.format_float_to_str(
            scaled_value_action_loss_e, n_decimals=n_decimals_for_printing)

        unscaled_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            unscaled_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        scaled_class_prediction_loss_str_e = cp_strings.format_float_to_str(
            scaled_class_prediction_loss_e, n_decimals=n_decimals_for_printing)
        accuracy_str_e = cp_strings.format_float_to_str(accuracy_e, n_decimals=n_decimals_for_printing)

        print(
            'Epoch: {e:d}. Validation. Unscaled Value Action Loss: {action_loss:s}. Unscaled Classification Loss: {class_prediction_loss:s}. Accuracy: {accuracy:s}.'.format(
                e=e, action_loss=unscaled_value_action_loss_str_e,
                class_prediction_loss=unscaled_class_prediction_loss_str_e, accuracy=accuracy_str_e))

        print('Epoch {e:d} - Unsuccessful Epochs {i:d}.'.format(e=e, i=i))

        print(dashes)

    print()

    n_completed_epochs = e + 1

    time_training = cp_timer.get_delta_time_total()

    print('Training completed in {d} days {h} hours {m} minutes {s} seconds'.format(
        d=time_training.days, h=time_training.hours,
        m=time_training.minutes, s=time_training.seconds))
    print('Number of Epochs: {E:d}'.format(E=E))
    print('Lowest Unscaled Classification Loss: {:s}'.format(lowest_unscaled_class_prediction_loss_str))
    print('Lowest Scaled Classification Loss: {:s}'.format(lowest_scaled_class_prediction_loss_str))
    print('Highest Accuracy: {:s}'.format(highest_accuracy_str))

    return None


class ReplayMemory:
    """A simple replay buffer."""

    def __init__(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def append(self, state=None, action=None, reward=None, next_state=None):

        self.states.append(state)

        self.actions.append(action)

        self.rewards.append(reward)

        self.next_states.append(next_state)

    def clear(self):
        self.__init__()
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

