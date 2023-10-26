

import os
import copy
import math
import torch
import numpy as np
from ....sl.dl.output_methods.general import *
from ..... import clock as cp_clock
from ..... import strings as cp_strings
from ..... import txt as cp_txt
from .....ml import utilities as cp_ml_utilities
from ....rl import utilities as cp_rl_utilities


__all__ = ['DQNMethods', 'TimedDQNMethods']


class DQNMethods(OutputMethods):

    def __init__(
            self, axis_features_outs, axis_models_losses,
            possible_actions, action_selection_type='active', same_indexes_actions=None, is_recurrent=False,
            gamma=0.999, reward_bias=0.0, loss_scales_actors=None):

        """

        :type axis_features_outs: int
        :type axis_models_losses: int
        :type possible_actions: list[list[int | float] | tuple[int | float]] |
                                tuple[list[int | float] | tuple[int | float]]
        :type action_selection_type: str
        :type same_indexes_actions: int | list | tuple | np.ndarray | torch.Tensor | None
        :type is_recurrent: bool
        :type gamma: int | float
        :type reward_bias: int | float
        :type loss_scales_actors: list[int | float] | tuple[int | float] |
                                  np.ndarray[int | float] | torch.Tensor[int | float] | float | int | None
        """

        superclass = DQNMethods
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if isinstance(possible_actions, list):
            self.possible_actions = possible_actions
        elif isinstance(possible_actions, tuple):
            self.possible_actions = list(possible_actions)
        elif isinstance(possible_actions, np.ndarray):
            self.possible_actions = possible_actions.tolist()
        else:
            raise TypeError('n_possible_actions')

        self.n_agents = self.A = len(self.possible_actions)
        self.loss_scales_actors = _set_loss_scales(M=self.A, loss_scales=loss_scales_actors)

        self.n_possible_actions = [-1 for a in range(0, self.A, 1)]  # type: list

        for a in range(0, self.A, 1):
            if isinstance(self.possible_actions[a], (list, tuple)):
                self.possible_actions[a] = torch.tensor(self.possible_actions[a])
            elif isinstance(self.possible_actions[a], np.ndarray):
                self.possible_actions[a] = torch.from_numpy(self.possible_actions[a])
            elif isinstance(self.possible_actions[a], torch.Tensor):
                pass
            else:
                raise TypeError('n_possible_actions')

            self.n_possible_actions[a] = len(self.possible_actions[a])

        self.possible_actions = tuple(self.possible_actions)
        self.n_possible_actions = tuple(self.n_possible_actions)

        if TimedOutputMethods not in self.superclasses_initiated:
            OutputMethods.__init__(
                self=self, axis_features_outs=axis_features_outs,
                axis_models_losses=axis_models_losses, M=self.A, loss_scales=self.loss_scales_actors)
            if TimedOutputMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(TimedOutputMethods)

        if isinstance(action_selection_type, str):
            if action_selection_type.lower() in ['active', 'random', 'same']:
                self.action_selection_type = action_selection_type.lower()
            else:
                raise ValueError('action_selection_type')
        else:
            raise TypeError('action_selection_type')

        if self.action_selection_type == 'same':
            if isinstance(same_indexes_actions, int):
                self.same_indexes_actions = [same_indexes_actions]  # type: list
            elif isinstance(same_indexes_actions, list):
                self.same_indexes_actions = same_indexes_actions
            elif isinstance(same_indexes_actions, tuple):
                self.same_indexes_actions = list(same_indexes_actions)
            elif isinstance(same_indexes_actions, (np.ndarray, torch.Tensor)):
                self.same_indexes_actions = same_indexes_actions.tolist()
            else:
                raise TypeError('same_indexes_actions')
        else:
            self.same_indexes_actions = same_indexes_actions


        if is_recurrent is None:
            self.is_recurrent = False
        elif isinstance(is_recurrent, bool):
            self.is_recurrent = is_recurrent
        else:
            raise TypeError('is_recurrent')

        self.gamma = gamma

        self.reward_bias = reward_bias

        self.criterion_values_actions = torch.nn.SmoothL1Loss(reduction='none')
        self.criterion_values_actions_reduction = torch.nn.SmoothL1Loss(reduction='mean')

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def sample_action(self, values_actions, epsilon=.1):

        """

        :type values_actions: list | torch.Tensor
        :type epsilon: float
        """

        # todo: forward only n non-random actions

        if isinstance(values_actions, torch.Tensor):
            device = values_actions.device
        else:
            device = values_actions[0].device

        A = len(values_actions)
        shape_actions = self.compute_shape_losses(values_actions)

        indexes_actions = [
            slice(0, shape_actions[a], 1) if a != self.axis_models_losses else None
            for a in range(0, values_actions[0].ndim, 1)]  # type: list

        shape_actions_a = [
            shape_actions[a] for a in range(0, values_actions[0].ndim, 1) if a != self.axis_models_losses]

        actions = torch.empty(shape_actions, dtype=torch.int64, device=device, requires_grad=False)

        if self.action_selection_type == 'active':

            mask_randoms = torch.rand(
                shape_actions_a, out=None, dtype=None, layout=torch.strided,
                device=device, requires_grad=False) < epsilon

            n_randoms = mask_randoms.sum(dtype=None).item()

            mask_greedy = torch.logical_not(mask_randoms, out=None)

            for a in range(0, A, 1):

                indexes_actions[self.axis_models_losses] = a
                tuple_indexes_actions = tuple(indexes_actions)

                random_action_a = torch.randint(
                    low=0, high=self.n_possible_actions[a], size=(n_randoms,),
                    generator=None, dtype=torch.int64, device=device, requires_grad=False)

                actions[tuple_indexes_actions][mask_randoms] = random_action_a

                actions[tuple_indexes_actions][mask_greedy] = (
                    # values_actions[a].max(dim=self.axis_features_outs, keepdim=True)[1][mask_greedy])
                    values_actions[a].max(dim=self.axis_features_outs, keepdim=False)[1][mask_greedy])

        elif self.action_selection_type == 'random':

            for a in range(0, A, 1):

                indexes_actions[self.axis_models_losses] = a
                tuple_indexes_actions = tuple(indexes_actions)

                actions[tuple_indexes_actions] = torch.randint(
                    low=0, high=self.n_possible_actions[a], size=shape_actions_a,
                    generator=None, dtype=torch.int64, device=device, requires_grad=False)

        elif self.action_selection_type == 'same':

            for a in range(0, A, 1):

                indexes_actions[self.axis_models_losses] = a
                tuple_indexes_actions = tuple(indexes_actions)

                actions[tuple_indexes_actions] = torch.full(
                    size=shape_actions_a, fill_value=self.same_indexes_actions[a],
                    dtype=torch.int64, device=device, requires_grad=False)
        else:
            raise ValueError('self.action_selection_type')

        return actions

    def gather_values_selected_actions(self, values_actions, actions):

        A = len(values_actions)
        shape_actions = actions.shape

        device = values_actions[0].device

        values_selected_actions = torch.empty(shape_actions, dtype=torch.float32, device=device, requires_grad=False)
        indexes_actions = [
            slice(0, values_selected_actions.shape[a], 1)
            for a in range(0, values_selected_actions.ndim, 1)]  # type: list

        for a in range(0, A, 1):

            indexes_actions[self.axis_models_losses] = a
            tuple_indexes_actions = tuple(indexes_actions)

            values_selected_actions[tuple_indexes_actions] = values_actions[a].gather(
                self.axis_features_outs, actions[tuple_indexes_actions].unsqueeze(
                    dim=self.axis_features_outs)).squeeze(dim=self.axis_features_outs)

        # values_selected_actions = [values_actions[a].gather(self.axis_features_outs, actions[a]).squeeze(
        #     dim=self.axis_features_outs) for a in range(0, self.A, 1)]
        # values_selected_actions = [values_actions[a].gather(self.axis_features_outs, actions[a].unsqueeze(
        #     dim=self.axis_features_outs)).squeeze(dim=self.axis_features_outs) for a in range(0, self.A, 1)]

        return values_selected_actions

    def compute_expected_values_actions(self, next_values_actions, rewards):

        A = len(next_values_actions)
        shape_actions = self.compute_shape_losses(next_values_actions)

        device = next_values_actions[0].device

        expected_values_actions = torch.empty(shape_actions, dtype=torch.float32, device=device, requires_grad=False)
        indexes_actions = [
            slice(0, expected_values_actions.shape[a], 1)
            for a in range(0, expected_values_actions.ndim, 1)]  # type: list

        biased_rewards = rewards + self.reward_bias

        for a in range(0, A, 1):

            indexes_actions[self.axis_models_losses] = a
            tuple_indexes_actions = tuple(indexes_actions)

            max_next_values_actions_a = next_values_actions[a].max(
                dim=self.axis_features_outs, keepdim=False)[0].detach()

            expected_values_actions[tuple_indexes_actions] = biased_rewards + (self.gamma * max_next_values_actions_a)

        return expected_values_actions.detach()

    def compute_value_action_losses(self, values_selected_actions, expected_values_actions):

        value_action_losses = self.criterion_values_actions(values_selected_actions, expected_values_actions.detach())

        return value_action_losses

    def reduce_value_action_losses(
            self, value_action_losses, axes_not_included=None,
            scaled=False, loss_scales_actors=None, format_scales=True):

        """


        :type value_action_losses: torch.Tensor | np.ndarray
        :type axes_not_included: int | list | tuple | np.ndarray | torch.Tensor | None
        :type scaled: bool
        :type loss_scales_actors: list | tuple | np.ndarray | torch.Tensor | None
        :type format_scales: bool

        :rtype:
        """

        if scaled and (loss_scales_actors is None):
            loss_scales_actors = self.loss_scales_actors
            format_scales = False

        reduced_value_action_losses = self.reduce_losses(
            losses=value_action_losses, axes_not_included=axes_not_included,
            scaled=scaled, loss_scales=loss_scales_actors, format_scales=format_scales)

        return reduced_value_action_losses

    def compute_n_selected_actions(self, selected_actions, axes_not_included=None):

        """

        :type selected_actions: np.ndarray | torch.Tensor
        :type axes_not_included: int | list | tuple | np.ndarray | torch.Tensor | None

        :rtype:
        """

        n_selected_actions = self.compute_n_losses(losses=selected_actions, axes_not_included=axes_not_included)

        return n_selected_actions

    def compute_deltas(self, actions: torch.Tensor, to_numpy: bool = True):

        indexes_actions = [
            slice(0, actions.shape[a], 1) if a != self.axis_models_losses else None
            for a in range(0, actions.ndim, 1)]  # type: list

        deltas = copy.deepcopy(actions)

        for a in range(0, self.A, 1):
            indexes_actions[self.axis_models_losses] = a

            tup_indexes_actions = tuple(indexes_actions)

            deltas[tup_indexes_actions] = self.possible_actions[a][actions[tup_indexes_actions]]

        if to_numpy:
            if deltas.is_cuda:
                deltas = deltas.cpu().numpy()
            else:
                deltas = deltas.numpy()

        return deltas

    class ReplayMemory:
        """A simple replay buffer."""

        def __init__(self, capacity, batch_size, is_recurrent):

            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []

            if isinstance(capacity, int):
                self.capacity = capacity
            else:
                raise TypeError('capacity')

            if isinstance(batch_size, int):
                self.batch_size = batch_size
            else:
                raise TypeError('batch_size')

            if is_recurrent is None:
                self.is_recurrent = False
            elif isinstance(is_recurrent, bool):
                self.is_recurrent = is_recurrent
            else:
                raise TypeError('is_recurrent')

            self.current_len = 0


        def add(self, states=None, actions=None, rewards=None, next_states=None):


            n_new_states = len(states)

            for a in range(0, n_new_states, 1):

                if states[a] is not None:
                    self.states.append(states[a])

                    self.actions.append(actions[a])

                    self.rewards.append(rewards[a])

                    if next_states[a] is None:

                        if self.is_recurrent:
                            self.next_states.append([
                                torch.zeros(
                                    size=states[a][0].shape, device=self.states[a][0].device,
                                    dtype=states[a][0].dtype, requires_grad=False),
                                next_states[a][1]])
                        else:
                            self.next_states.append(torch.zeros(
                                size=states[a][0].shape, device=self.states[a][0].device,
                                dtype=states[a][0].dtype, requires_grad=False))
                    else:
                        self.next_states.append(next_states[a])

            self.current_len = len(self.states)

            self.remove_extras()

        def remove_extras(self):

            n_extras = self.current_len - self.capacity

            if n_extras > 0:
                self.states = self.states[slice(n_extras, self.current_len, 1)]

                self.actions = self.actions[slice(n_extras, self.current_len, 1)]

                self.rewards = self.rewards[slice(n_extras, self.current_len, 1)]

                self.next_states = self.next_states[slice(n_extras, self.current_len, 1)]

                self.current_len = len(self.states)

            return None

        def clear(self):
            self.__init__(capacity=self.capacity, batch_size=self.batch_size, is_recurrent=self.is_recurrent)
            return None

        def sample(self):

            if self.batch_size > self.capacity:
                raise ValueError('self.batch_size > self.capacity')
            else:
                indexes = np.random.choice(a=np.arange(0, self.current_len, 1), size=[self.batch_size], replace=False)

            states = []
            actions = []
            rewards = []
            next_states = []
            for i in indexes:
                states.append(self.states.pop(i))
                actions.append(self.actions.pop(i))
                rewards.append(self.rewards.pop(i))
                next_states.append(self.next_states.pop(i))

            self.current_len = len(self.states)

            if self.is_recurrent:
                states = [
                    torch.cat([states[i][0] for i in range(0, self.batch_size, 1)], dim=0),
                    torch.cat([states[i][1] for i in range(0, self.batch_size, 1)], dim=0)]

                next_states = [
                    torch.cat([next_states[i][0] for i in range(0, self.batch_size, 1)], dim=0),
                    torch.cat([next_states[i][1] for i in range(0, self.batch_size, 1)], dim=0)]
                # todo: if lstm
            else:
                states = torch.cat(states, dim=0)
                next_states = torch.cat(next_states, dim=0)

            actions = torch.cat(actions, dim=0)
            rewards = torch.cat(rewards, dim=0)

            return dict(
                states=states, actions=actions,
                next_states=next_states, rewards=rewards)

        def __len__(self) -> int:
            return len(self.states)

    def train(
            self, model, environment, optimizer, U=10, E=None, min_n_episodes_for_optim=10, min_n_samples_for_optim=1000,
            n_batches_per_train_phase=100, batch_size_of_train=100, max_time_steps_per_train_episode=None,
            n_batches_per_val_phase=1000, batch_size_of_val=10, max_time_steps_per_val_episode=None,
            epsilon_start=.95, epsilon_end=.05, epsilon_step=-.05,
            directory_outputs=None):

        cp_timer = cp_clock.Timer()

        phases_names = ('training', 'validation')
        for key_environment_k in environment.keys():
            if key_environment_k in phases_names:
                pass
            else:
                raise ValueError('Unknown keys in environment')

        n_batches_per_phase = {'training': n_batches_per_train_phase, 'validation': n_batches_per_val_phase}
        if n_batches_per_phase['training'] is None:
            n_batches_per_phase['training'] = 100
        elif isinstance(n_batches_per_phase['training'], int):
            pass
        else:
            raise TypeError('n_batches_per_train_phase')
        if n_batches_per_phase['validation'] is None:
            n_batches_per_phase['validation'] = 100
        elif isinstance(n_batches_per_phase['validation'], int):
            pass
        else:
            raise TypeError('n_batches_per_val_phase')

        batch_size = {'training': batch_size_of_train, 'validation': batch_size_of_val}
        if batch_size['training'] is None:
            batch_size['training'] = 1000
        elif isinstance(batch_size['training'], int):
            pass
        else:
            raise TypeError('batch_size_of_train')
        if batch_size['validation'] is None:
            batch_size['validation'] = 10
        elif isinstance(batch_size['validation'], int):
            pass
        else:
            raise TypeError('batch_size_of_val')

        tot_observations_per_phase = {
            phases_name_p: n_batches_per_phase[phases_name_p] * batch_size[phases_name_p]
            for phases_name_p in phases_names}

        max_time_steps_per_episode = {
            'training': max_time_steps_per_train_episode, 'validation': max_time_steps_per_val_episode}
        if max_time_steps_per_episode['training'] is None:
            max_time_steps_per_episode['training'] = math.inf
        elif isinstance(max_time_steps_per_episode['training'], int):
            pass
        else:
            raise TypeError('max_time_steps_per_train_episode')
        if max_time_steps_per_episode['validation'] is None:
            max_time_steps_per_episode['validation'] = 10
        elif isinstance(max_time_steps_per_episode['validation'], int):
            pass
        else:
            raise TypeError('max_time_steps_per_val_episode')

        model.freeze()
        torch.set_grad_enabled(False)

        headers = [
            'Epoch', 'Unsuccessful_Epochs',
            'Start_Date', 'Start_Time' 'Epoch_Duration', 'Elapsed_Time',
            'Training_Loss', 'Validation_Loss', 'Lowest_Validation_Loss', 'Is_Lower_Validation_Loss',
            'Training_Reward', 'Validation_Reward', 'Highest_Validation_Reward', 'Is_Highest_Validation_Reward'
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

        directory_model_with_lowest_loss = os.path.join(directory_outputs, 'model_with_lowest_loss.pth')

        directory_model_with_lowest_loss = os.path.join(directory_outputs, 'model_with_lowest_loss.pth')

        directory_model_with_highest_reward = os.path.join(directory_outputs, 'model_with_highest_reward.pth')

        directory_stats = os.path.join(directory_outputs, 'stats.csv')

        n_decimals_for_printing = 6
        n_dashes = 150
        dashes = '-' * n_dashes
        print(dashes)

        replay_memory = self.ReplayMemory(
            capacity=100000, batch_size=batch_size['training'], is_recurrent=self.is_recurrent)

        lowest_loss = math.inf
        lowest_loss_str = str(lowest_loss)

        highest_reward = -math.inf
        highest_reward_str = str(highest_reward)

        epsilon = epsilon_start  # todo to the model
        if epsilon < epsilon_end:
            epsilon = epsilon_end

        epsilon_validation = 0

        epochs = cp_ml_utilities.EpochsIterator(U=U, E=E)

        for e, u in epochs:

            print('Epoch {e} ...'.format(e=e))

            stats['lines'].append(new_line_stats.copy())
            stats['lines'][e][stats['headers']['Epoch']] = e

            # Each Training Epoch has a training and a validation phase
            # training phase


            model.train()

            running_loss_e = 0.0
            running_n_selected_actions_e = 1

            env_iterator = cp_rl_utilities.EnvironmentsIterator(
                tot_observations_per_epoch=tot_observations_per_phase['training'])

            b = 0

            for i in env_iterator:

                hc_eit = [None for a in range(0, environment['training'].n_platforms, 1)]  # type: list
                state_eit = [None for a in range(0, environment['training'].n_platforms, 1)]  # type: list
                action_eit = [None for a in range(0, environment['training'].n_platforms, 1)]  # type: list
                delta_ebt = [None for a in range(0, environment['training'].n_platforms, 1)]  # type: list
                # hc_eit = None
                # state_eit = None

                observation_eit = environment['training'].reset()

                obs_iterator = cp_rl_utilities.ObservationsIterator(T=max_time_steps_per_episode['training'])

                for t in obs_iterator:

                    for a in range(0, len(observation_eit), 1):

                        if observation_eit[a] is None:
                            hc_eit[a] = None
                            state_eit[a] = None
                            action_eit[a] = None
                            delta_ebt[a] = None
                        else:
                            if self.is_recurrent:
                                if hc_eit[a] is None:
                                    hc_eit[a] = model.init_h(
                                        batch_shape=model.get_batch_shape(input_shape=observation_eit[a].shape))
                                if state_eit[a] is None:
                                    state_eit[a] = copy.deepcopy([observation_eit[a], hc_eit[a]])
                                values_actions_eit, hc_eit[a] = model(x=state_eit[a][0], h=state_eit[a][1])
                            else:
                                if state_eit[a] is None:
                                    state_eit[a] = copy.deepcopy(observation_eit[a])
                                values_actions_eit = model(x=state_eit[a])

                            action_eit[a] = self.sample_action(values_actions=values_actions_eit, epsilon=epsilon)
                            delta_ebt[a] = model.compute_deltas(action_eit[a])

                    next_observation_eit, reward_eit, obs_iterator.not_over = environment['training'].step(
                        actions=delta_ebt)

                    if next_observation_eit is None:
                        next_state_eit = None
                    else:
                        if self.is_recurrent:
                            next_state_eit = copy.deepcopy([next_observation_eit, hc_eit])
                        else:
                            next_state_eit = copy.deepcopy(next_observation_eit)

                    replay_memory.add(
                        states=state_eit, actions=action_eit, next_states=next_state_eit, rewards=reward_eit)

                    observation_eit = next_observation_eit
                    state_eit = copy.deepcopy(next_state_eit)

                if i >= min_n_episodes_for_optim:
                    while (replay_memory.current_len >= min_n_samples_for_optim) and (replay_memory.current_len >= batch_size['training']):

                        samples_eb = replay_memory.sample()
                        states_eb = samples_eb['states']
                        actions_eb = samples_eb['actions']
                        next_states_eb = samples_eb['next_states']
                        rewards_eb = samples_eb['rewards']

                        if self.is_recurrent:
                            next_values_actions_eb, next_hc_eb = model(x=next_states_eb)
                        else:
                            next_values_actions_eb = model(x=next_states_eb)

                        expected_values_actions_eb = self.compute_expected_values_actions(
                            next_values_actions=next_values_actions_eb, rewards=rewards_eb)

                        optimizer.zero_grad()

                        # forward
                        # track history
                        torch.set_grad_enabled(True)
                        model.unfreeze()

                        if self.is_recurrent:
                            values_actions_eb, hc_eb = model(x=states_eb)
                        else:
                            values_actions_eb = model(x=states_eb)

                        values_selected_actions_eb = self.gather_values_selected_actions(
                            values_actions=values_actions_eb, actions=actions_eb)

                        value_action_losses_eb = self.compute_value_action_losses(
                            values_selected_actions=values_selected_actions_eb,
                            expected_values_actions=expected_values_actions_eb)

                        scaled_value_action_loss_eb = self.reduce_value_action_losses(
                            value_action_losses=value_action_losses_eb, axes_not_included=None,
                            scaled=True, loss_scales_actors=None, format_scales=False)

                        scaled_value_action_loss_eb.backward()
                        optimizer.step()

                        model.freeze()
                        torch.set_grad_enabled(False)

                        n_selected_actions_eb = self.compute_n_selected_actions(
                            selected_actions=actions_eb, axes_not_included=None)

                        running_n_selected_actions_e += n_selected_actions_eb
                        running_loss_e += (scaled_value_action_loss_eb.item() * n_selected_actions_eb)


                        env_iterator + batch_size['training']
                        b += 1

            loss_e = running_loss_e / running_n_selected_actions_e

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

            model.eval()

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
                    values_selected_actions=values_selected_actions_eb,
                    expected_values_actions=expected_values_actions_eb)

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


class TimedDQNMethods(DQNMethods, TimedOutputMethods):

    def __init__(
            self,
            axis_time_outs, axis_batch_outs, axis_features_outs, axis_models_losses,
            possible_actions, action_selection_type='active', same_indexes_actions=None,
            gamma=0.999, reward_bias=0.0, loss_scales_actors=None):

        """

        :type axis_batch_outs: int
        :type axis_features_outs: int
        :type axis_models_losses: int
        :type possible_actions: list[list[int | float] | tuple[int | float]] |
                                tuple[list[int | float] | tuple[int | float]]
        :type action_selection_type: str
        :type same_indexes_actions: int | list | tuple | np.ndarray | torch.Tensor | None
        :type gamma: int | float
        :type reward_bias: int | float
        :type loss_scales_actors: list[int | float] | tuple[int | float] |
                                  np.ndarray[int | float] | torch.Tensor[int | float] | float | int | None
        """

        superclass = TimedDQNMethods
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if DQNMethods not in self.superclasses_initiated:
            DQNMethods.__init__(
                self=self, axis_features_outs=axis_features_outs, axis_models_losses=axis_models_losses,
                possible_actions=possible_actions, action_selection_type=action_selection_type,
                same_indexes_actions=same_indexes_actions,
                gamma=gamma, reward_bias=reward_bias, loss_scales_actors=loss_scales_actors)
            if DQNMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(DQNMethods)

        if TimedOutputMethods not in self.superclasses_initiated:
            TimedOutputMethods.__init__(
                self=self, axis_time_outs=axis_time_outs, axis_batch_outs=axis_batch_outs,
                axis_features_outs=axis_features_outs, axis_models_losses=axis_models_losses,
                M=self.A, loss_scales=self.loss_scales_actors)
            if TimedOutputMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(TimedOutputMethods)

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def remove_last_values_actions(self, values_actions: list):

        if self.axis_time_outs is None:
            raise ValueError('self.axis_time_outs')
        else:
            A = len(values_actions)
            values_actions_out = [None for a in range(0, A, 1)]

            for a in range(A):
                tuple_indexes_a = tuple(
                    [slice(0, values_actions[a].shape[d], 1)
                     if d != self.axis_time_outs
                     else slice(0, values_actions[a].shape[d] - 1, 1)
                     for d in range(0, values_actions[a].ndim, 1)])

                values_actions_out[a] = values_actions[a][tuple_indexes_a]

        return values_actions_out


