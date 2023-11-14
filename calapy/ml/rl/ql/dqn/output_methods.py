

import os
import copy
import math
import torch
import numpy as np
from ..... import clock as cp_clock
from ..... import strings as cp_strings
from ..... import txt as cp_txt
from ....rl import utilities as cp_rl_utilities
from .....ml import utilities as cp_ml_utilities
from ....sl.dl.output_methods.general import *


__all__ = ['DQNMethods', 'TimedDQNMethods']


class DQNMethods(OutputMethods):

    def __init__(
            self, model, axis_features_outs, axis_models_losses,
            possible_actions, action_selection_type='active', same_indexes_actions=None,
            gamma=0.999, reward_bias=0.0, loss_scales_actors=None, is_recurrent=False):

        """

        :type axis_features_outs: int
        :type axis_models_losses: int
        :type possible_actions: list[list[int | float] | tuple[int | float]] |
                                tuple[list[int | float] | tuple[int | float]]
        :type action_selection_type: str
        :type same_indexes_actions: int | list | tuple | np.ndarray | torch.Tensor | None
        :type is_recurrent: bool | None
        :type gamma: int | float | None
        :type reward_bias: int | float | None
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

        self.model = model

        self.device = model.device
        self.dtype = model.dtype

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
                self.possible_actions[a] = torch.tensor(self.possible_actions[a], device=self.device)
            elif isinstance(self.possible_actions[a], np.ndarray):
                self.possible_actions[a] = torch.from_numpy(self.possible_actions[a]).to(device=self.device)
            elif isinstance(self.possible_actions[a], torch.Tensor):
                self.possible_actions[a].to(device=self.device)
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

        if gamma is None:
            self.gamma = 0.999
        elif isinstance(gamma, float):
            self.gamma = gamma
        elif isinstance(gamma, int):
            self.gamma = float(gamma)
        else:
            raise TypeError('gamma')

        if reward_bias is None:
            self.reward_bias = 0.0
        elif isinstance(reward_bias, float):
            self.reward_bias = reward_bias
        elif isinstance(reward_bias, int):
            self.reward_bias = float(reward_bias)
        else:
            raise TypeError('reward_bias')

        if reward_bias is None:
            self.reward_bias = 0.0
        elif isinstance(reward_bias, float):
            self.reward_bias = reward_bias
        elif isinstance(reward_bias, int):
            self.reward_bias = float(reward_bias)
        else:
            raise TypeError('reward_bias')

        self.criterion_values_actions = torch.nn.SmoothL1Loss(reduction='none')
        self.criterion_values_actions_reduction = torch.nn.SmoothL1Loss(reduction='mean')

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def q_values_to_actions(self, values_actions):

        """

        :type values_actions: list | torch.Tensor
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

        actions = torch.empty(shape_actions, dtype=torch.int64, device=device, requires_grad=False)

        for a in range(0, A, 1):

            indexes_actions[self.axis_models_losses] = a
            tuple_indexes_actions = tuple(indexes_actions)

            # actions[tuple_indexes_actions] = values_actions[a].max(dim=self.axis_features_outs, keepdim=True)[1]
            actions[tuple_indexes_actions] = values_actions[a].max(dim=self.axis_features_outs, keepdim=False)[1]

        return actions

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

        # deltas = copy.deepcopy(actions)
        # for a in range(0, self.A, 1):
        #     indexes_actions[self.axis_models_losses] = a
        #     tup_indexes_actions = tuple(indexes_actions)
        #     deltas[tup_indexes_actions] = self.possible_actions[a][actions[tup_indexes_actions]]
        # if to_numpy:
        #     if deltas.is_cuda:
        #         deltas = deltas.cpu().numpy()
        #     else:
        #         deltas = deltas.numpy()

        deltas = [None for a in range(0, self.A, 1)]  # type: list[torch.Tensor] | list[None]

        for a in range(0, self.A, 1):
            indexes_actions[self.axis_models_losses] = a
            tup_indexes_actions = tuple(indexes_actions)
            deltas[a] = self.possible_actions[a][actions[tup_indexes_actions]]

            if to_numpy:
                if deltas[a].is_cuda:
                    deltas[a] = deltas[a].cpu().numpy()
                else:
                    deltas[a] = deltas[a].numpy()

        return deltas

    def train_from_ind_agents(
            self, environment, optimizer,
            replay_memory_add_as, state_batch_axis, action_batch_axis, reward_batch_axis,
            U=10, E=None,
            n_batches_per_train_phase=100, batch_size_of_train=100, T_train=None,
            epsilon_start=.95, epsilon_end=.01, epsilon_step=-.05,  capacity=100000,
            min_n_episodes_for_optim=2, min_n_samples_for_optim=1000,
            n_episodes_per_val_phase=1000, T_val=None, directory_outputs=None):

        cp_timer = cp_clock.Timer()

        phases_names = ('training', 'validation')
        for key_environment_k in environment.keys():
            if key_environment_k in phases_names:
                pass
            else:
                raise ValueError('Unknown keys in environment')

        if n_batches_per_train_phase is None:
            n_batches_per_train_phase = 100
        elif isinstance(n_batches_per_train_phase, int):
            pass
        else:
            raise TypeError('n_batches_per_train_phase')

        if batch_size_of_train is None:
            batch_size_of_train = 100
        elif isinstance(batch_size_of_train, int):
            pass
        else:
            raise TypeError('batch_size_of_train')

        tot_observations_per_train_phase = n_batches_per_train_phase * batch_size_of_train

        T = {'training': T_train, 'validation': T_val}
        if T['training'] is None:
            T['training'] = math.inf
        elif isinstance(T['training'], int):
            pass
        elif isinstance(T['training'], float):
            if T['training'] == math.inf:
                pass
            else:
                raise ValueError('T_train')
        else:
            raise TypeError('T_train')

        if T['validation'] is None:
            T['validation'] = math.inf
        elif isinstance(T['validation'], int):
            pass
        elif isinstance(T['validation'], float):
            if T['validation'] == math.inf:
                pass
            else:
                raise ValueError('T_val')
        else:
            raise TypeError('T_val')

        self.model.freeze()
        torch.set_grad_enabled(False)

        headers = [
            'Epoch', 'Unsuccessful_Epochs',
            # 'Start_Date', 'Start_Time' 'Epoch_Duration', 'Elapsed_Time',
            'Training_Loss', 'Training_Reward_Per_Observation',

            # 'Validation_Loss',
            'Lowest_Validation_Loss', 'Is_Lower_Validation_Loss',
            # 'Validation_Reward',
            'Highest_Validation_Reward', 'Is_Highest_Validation_Reward'
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
        directory_model_with_highest_reward = os.path.join(directory_outputs, 'model_with_highest_reward.pth')

        directory_stats = os.path.join(directory_outputs, 'stats.csv')

        n_decimals_for_printing = 6
        n_dashes = 150
        dashes = '-' * n_dashes
        print(dashes)

        replay_memory = ReplayMemory(
            capacity=capacity, batch_size=batch_size_of_train, add_as=replay_memory_add_as,
            state_batch_axis=state_batch_axis, action_batch_axis=action_batch_axis, reward_batch_axis=reward_batch_axis,
            is_recurrent=self.is_recurrent, model=self.model)

        lowest_loss = math.inf
        lowest_loss_str = str(lowest_loss)

        highest_reward = -math.inf
        highest_reward_str = str(highest_reward)

        epsilon = epsilon_start  # todo to the model
        if epsilon < epsilon_end:
            epsilon = epsilon_end
        epsilon_validation = 0

        epochs = cp_ml_utilities.EpochsIterator(U=U, E=E)
        e = 0
        u = 0

        for e, u in epochs:

            print('Epoch {e} ...'.format(e=e))

            stats['lines'].append(new_line_stats.copy())
            stats['lines'][e][stats['headers']['Epoch']] = e

            # Each Training Epoch has a training and a validation phase
            # training phase

            self.model.train()

            running_n_selected_actions_e = 0
            running_loss_e = 0.0

            running_n_rewards_e = 0
            running_rewards_e = 0.0

            s = 0
            episodes_iterator = cp_rl_utilities.TrainEpisodesIterator(
                tot_observations_per_epoch=tot_observations_per_train_phase)

            for i, j in episodes_iterator:

                observation_eit = environment['training'].reset()

                hc_eit = [None for a in range(0, environment['training'].n_agents, 1)]  # type: list
                state_eit = [None for a in range(0, environment['training'].n_agents, 1)]  # type: list
                action_eit = [None for a in range(0, environment['training'].n_agents, 1)]  # type: list
                delta_ebt = [None for a in range(0, environment['training'].n_agents, 1)]  # type: list
                # hc_eit = None
                # state_eit = None

                obs_iterator = cp_rl_utilities.ObservationsIterator(T=T['training'])

                for t in obs_iterator:

                    for a in range(0, environment['training'].n_agents, 1):

                        if observation_eit[a] is None:
                            hc_eit[a] = None
                            state_eit[a] = None
                            action_eit[a] = None
                            delta_ebt[a] = None
                        else:
                            if self.is_recurrent:
                                if hc_eit[a] is None:
                                    hc_eit[a] = self.model.init_h(
                                        batch_shape=self.model.get_batch_shape(input_shape=observation_eit[a].shape))
                                if state_eit[a] is None:
                                    state_eit[a] = copy.deepcopy([observation_eit[a], hc_eit[a]])
                                values_actions_eit, hc_eit[a] = self.model(x=state_eit[a][0], h=state_eit[a][1])
                            else:
                                if state_eit[a] is None:
                                    state_eit[a] = copy.deepcopy(observation_eit[a])
                                values_actions_eit = self.model(x=state_eit[a])

                            action_eit[a] = self.sample_action(values_actions=values_actions_eit, epsilon=epsilon)
                            delta_ebt[a] = self.compute_deltas(action_eit[a])

                    next_observation_eit, reward_eit, obs_iterator.not_over = environment['training'].step(
                        deltas=delta_ebt)

                    if next_observation_eit is None:
                        next_state_eit = None
                    else:
                        if self.is_recurrent:
                            next_state_eit = [
                                [next_observation_eit[a], hc_eit[a]] for a in range(0, environment['training'].n_agents, 1)]
                        else:
                            next_state_eit = next_observation_eit

                    for a in range(0, environment['training'].n_agents, 1):
                        if observation_eit[a] is not None:
                            replay_memory.add(
                                states=state_eit[a], actions=action_eit[a], next_states=next_state_eit[a], rewards=reward_eit[a])

                    observation_eit = copy.deepcopy(next_observation_eit)
                    state_eit = copy.deepcopy(next_state_eit)

                if j >= min_n_episodes_for_optim:
                    while ((replay_memory.current_len >= min_n_samples_for_optim) and
                           (replay_memory.current_len >= batch_size_of_train) and episodes_iterator.not_over):

                        samples_eb = replay_memory.sample()
                        states_eb = samples_eb['states']
                        actions_eb = samples_eb['actions']
                        next_states_eb = samples_eb['next_states']
                        rewards_eb = samples_eb['rewards']

                        if self.is_recurrent:
                            next_values_actions_eb, next_hc_eb = self.model(x=next_states_eb[0], h=next_states_eb[1])
                        else:
                            next_values_actions_eb = self.model(x=next_states_eb)

                        expected_values_actions_eb = self.compute_expected_values_actions(
                            next_values_actions=next_values_actions_eb, rewards=rewards_eb)

                        optimizer.zero_grad()

                        # forward
                        # track history
                        torch.set_grad_enabled(True)
                        self.model.unfreeze()

                        if self.is_recurrent:
                            values_actions_eb, hc_eb = self.model(x=states_eb[0], h=states_eb[1])
                        else:
                            values_actions_eb = self.model(x=states_eb)

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

                        self.model.freeze()
                        torch.set_grad_enabled(False)

                        n_selected_actions_eb = self.compute_n_selected_actions(
                            selected_actions=actions_eb, axes_not_included=None)

                        n_rewards_eb = self.compute_n_selected_actions(
                            selected_actions=actions_eb, axes_not_included=None)

                        running_n_selected_actions_e += n_selected_actions_eb
                        running_loss_e += (scaled_value_action_loss_eb.item() * n_selected_actions_eb)

                        running_n_rewards_e += n_rewards_eb
                        running_rewards_e += rewards_eb.sum(dim=None, keepdim=False, dtype=None).item()

                        s, b = episodes_iterator.count_observations(n_new_observations=replay_memory.batch_size)
                        j = 0

            loss_e = running_loss_e / running_n_selected_actions_e
            reward_e = running_rewards_e / running_n_rewards_e

            stats['lines'][e][stats['headers']['Training_Loss']] = loss_e
            stats['lines'][e][stats['headers']['Training_Reward_Per_Observation']] = reward_e

            loss_str_e = cp_strings.format_float_to_str(loss_e, n_decimals=n_decimals_for_printing)
            reward_str_e = cp_strings.format_float_to_str(reward_e, n_decimals=n_decimals_for_printing)

            print('Epoch: {e:d}. Training. Value Action Loss: {loss:s}. Reward per Observation {reward:s}'.format(
                e=e, loss=loss_str_e, reward=reward_str_e))

            epsilon = epsilon + epsilon_step
            if epsilon < epsilon_end:
                epsilon = epsilon_end






            # validation phase

            self.model.eval()

            running_n_selected_actions_e = 0
            running_loss_e = 0.0

            running_n_rewards_e = 0
            running_rewards_e = 0.0

            s = 0
            episodes_iterator = cp_rl_utilities.ValEpisodesIterator(n_episodes=n_episodes_per_val_phase)

            for i in episodes_iterator:

                observation_eit = environment['validation'].reset()

                hc_eit = [None for a in range(0, environment['validation'].n_agents, 1)]  # type: list
                state_eit = [None for a in range(0, environment['validation'].n_agents, 1)]  # type: list
                action_eit = [None for a in range(0, environment['validation'].n_agents, 1)]  # type: list
                delta_ebt = [None for a in range(0, environment['validation'].n_agents, 1)]  # type: list

                cum_rewards = []  # type: list
                time_lengths = []  # type: list

                cum_rewards_i = [None for a in range(0, environment['validation'].n_agents, 1)]  # type: list
                time_lengths_i = [None for a in range(0, environment['validation'].n_agents, 1)]  # type: list

                obs_iterator = cp_rl_utilities.ObservationsIterator(T=T['validation'])

                for t in obs_iterator:

                    for a in range(0, environment['validation'].n_agents, 1):

                        if observation_eit[a] is None:
                            hc_eit[a] = None
                            state_eit[a] = None
                            action_eit[a] = None
                            delta_ebt[a] = None

                            time_lengths_i[a] = t
                        else:
                            if self.is_recurrent:
                                if hc_eit[a] is None:
                                    hc_eit[a] = self.model.init_h(
                                        batch_shape=self.model.get_batch_shape(input_shape=observation_eit[a].shape))
                                if state_eit[a] is None:
                                    state_eit[a] = copy.deepcopy([observation_eit[a], hc_eit[a]])
                                values_actions_eit, hc_eit[a] = self.model(x=state_eit[a][0], h=state_eit[a][1])
                            else:
                                if state_eit[a] is None:
                                    state_eit[a] = copy.deepcopy(observation_eit[a])
                                values_actions_eit = self.model(x=state_eit[a])

                            action_eit[a] = self.sample_action(values_actions=values_actions_eit, epsilon=epsilon)
                            delta_ebt[a] = self.compute_deltas(action_eit[a])

                    next_observation_eit, reward_eit, obs_iterator.not_over = environment['validation'].step(
                        deltas=delta_ebt)

                    if next_observation_eit is None:
                        next_state_eit = None
                    else:
                        if self.is_recurrent:
                            next_state_eit = [
                                [next_observation_eit[a], hc_eit[a]] for a in
                                range(0, environment['validation'].n_agents, 1)]
                        else:
                            next_state_eit = next_observation_eit

                    expected_values_actions_eb = self.compute_expected_values_actions(
                        next_values_actions=next_values_actions_eb, rewards=rewards_eb)


                    values_selected_actions_eb = self.gather_values_selected_actions(
                        values_actions=values_actions_eb, actions=actions_eb)


                    observation_eit = next_observation_eit
                    state_eit = copy.deepcopy(next_state_eit)




                    value_action_losses_eb = self.compute_value_action_losses(
                        values_selected_actions=values_selected_actions_eb,
                        expected_values_actions=expected_values_actions_eb)

                    scaled_value_action_loss_eb = self.reduce_value_action_losses(
                        value_action_losses=value_action_losses_eb, axes_not_included=None,
                        scaled=True, loss_scales_actors=None, format_scales=False)


                    n_selected_actions_eb = self.compute_n_selected_actions(
                        selected_actions=actions_eb, axes_not_included=None)

                    n_rewards_eb = self.compute_n_selected_actions(
                        selected_actions=actions_eb, axes_not_included=None)

                    running_n_selected_actions_e += n_selected_actions_eb
                    running_loss_e += (scaled_value_action_loss_eb.item() * n_selected_actions_eb)

                    running_n_rewards_e += n_rewards_eb
                    running_rewards_e += rewards_eb.sum(dim=None, keepdim=False, dtype=None).item()

                    s, b = episodes_iterator.count_observations(n_new_observations=replay_memory.batch_size)
                    j = 0

                cum_rewards += cum_rewards_i
                time_lengths += time_lengths_i




            loss_e = running_loss_e / running_n_selected_actions_e
            reward_e = running_rewards_e / running_n_rewards_e

            stats['lines'][e][stats['headers']['Training_Loss']] = loss_e
            stats['lines'][e][stats['headers']['Training_Reward_Per_Observation']] = reward_e

            loss_str_e = cp_strings.format_float_to_str(loss_e, n_decimals=n_decimals_for_printing)
            reward_str_e = cp_strings.format_float_to_str(reward_e, n_decimals=n_decimals_for_printing)

            print('Epoch: {e:d}. Training. Value Action Loss: {loss:s}. Reward per Observation {reward:s}'.format(
                e=e, loss=loss_str_e, reward=reward_str_e))










            model_dict = copy.deepcopy(self.model.state_dict())
            if os.path.isfile(directory_model_at_last_epoch):
                os.remove(directory_model_at_last_epoch)
            torch.save(model_dict, directory_model_at_last_epoch)

            is_successful_epoch = False

            if loss_e < lowest_loss:

                lowest_loss = loss_e
                lowest_loss_str = cp_strings.format_float_to_str(loss_e, n_decimals=n_decimals_for_printing)

                stats['lines'][e][stats['headers']['Is_Lower_Validation_Loss']] = 1
                is_successful_epoch = True

                if os.path.isfile(directory_model_with_lowest_loss):
                    os.remove(directory_model_with_lowest_loss)
                torch.save(model_dict, directory_model_with_lowest_loss)
            else:
                stats['lines'][e][stats['headers']['Is_Lower_Validation_Loss']] = 0

            stats['lines'][e][stats['headers']['Lowest_Validation_Loss']] = lowest_loss

            if reward_e > highest_reward:
                highest_reward = reward_e
                highest_reward_str = cp_strings.format_float_to_str(highest_reward, n_decimals=n_decimals_for_printing)

                stats['lines'][e][stats['headers']['Is_Highest_Validation_Reward']] = 1
                is_successful_epoch = True

                if os.path.isfile(directory_model_with_highest_reward):
                    os.remove(directory_model_with_highest_reward)
                torch.save(model_dict, directory_model_with_highest_reward)
            else:
                stats['lines'][e][stats['headers']['Is_Highest_Validation_Reward']] = 0

            stats['lines'][e][stats['headers']['Highest_Validation_Reward']] = highest_reward

            epochs.count_unsuccessful_epochs(is_successful_epoch=is_successful_epoch)
            stats['lines'][e][stats['headers']['Unsuccessful_Epochs']] = epochs.u

            if os.path.isfile(directory_stats):
                os.remove(directory_stats)
            cp_txt.lines_to_csv_file(stats['lines'], directory_stats, stats['headers'])

            print('Epoch: {e:d}. Validation. Value Action Loss: {loss:s}. Reward per Observation {reward:s}.'.format(
                e=e, loss=loss_str_e, reward=reward_str_e))

            print('Epoch {e:d} - Unsuccessful Epochs {u:d}.'.format(e=e, u=epochs.u))

            print(dashes)

        print()

        n_completed_epochs = E = e + 1

        time_training = cp_timer.get_delta_time_total()

        print('Training completed in {d} days {h} hours {m} minutes {s} seconds'.format(
            d=time_training.days, h=time_training.hours,
            m=time_training.minutes, s=time_training.seconds))
        print('Number of Epochs: {E:d}'.format(E=E))
        print('Lowest Loss: {:s}'.format(lowest_loss_str))
        print('Highest Reward: {:s}'.format(highest_reward_str))

        return self.model

    # def _multi_agent_obs_to_deltas(self, observation_eit, ):
    #
    #     n_agents = len(observation_eit)
    #
    #     for a in range(0, n_agents, 1):
    #
    #         if observation_eit[a] is None:
    #             hc_eit[a] = None
    #             state_eit[a] = None
    #             action_eit[a] = None
    #             delta_ebt[a] = None
    #         else:
    #             if self.is_recurrent:
    #                 if hc_eit[a] is None:
    #                     hc_eit[a] = self.model.init_h(
    #                         batch_shape=self.model.get_batch_shape(input_shape=observation_eit[a].shape))
    #                 if state_eit[a] is None:
    #                     state_eit[a] = copy.deepcopy([observation_eit[a], hc_eit[a]])
    #                 values_actions_eit, hc_eit[a] = self.model(x=state_eit[a][0], h=state_eit[a][1])
    #             else:
    #                 if state_eit[a] is None:
    #                     state_eit[a] = copy.deepcopy(observation_eit[a])
    #                 values_actions_eit = self.model(x=state_eit[a])
    #
    #             action_eit[a] = self.q_values_to_actions(values_actions=values_actions_eit)
    #
    #             delta_ebt[a] = self.compute_deltas(action_eit[a])


class ReplayMemory:
    """A simple replay buffer."""

    def __init__(
            self, capacity, batch_size, add_as,
            state_batch_axis, action_batch_axis, reward_batch_axis,
            is_recurrent=False, model=None):

        """


        :type capacity: int
        :type batch_size: int
        :param add_as: "s" for single observation, "l" for list or tuple of observation, "t" for tensor or array of
                       some batched observations
        :type state_batch_axis: int
        :type action_batch_axis: int
        :type reward_batch_axis: int

        :type add_as: str
        :type is_recurrent: bool
        :type model: model | None
        """

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

        if isinstance(state_batch_axis, int):
            self.state_batch_axis = state_batch_axis
        else:
            raise TypeError('state_batch_axis')

        if isinstance(action_batch_axis, int):
            self.action_batch_axis = action_batch_axis
        else:
            raise TypeError('action_batch_axis')

        if isinstance(reward_batch_axis, int):
            self.reward_batch_axis = reward_batch_axis
        else:
            raise TypeError('reward_batch_axis')

        if isinstance(add_as, str):
            self.add_as = add_as.lower()
            if self.add_as == 's':
                self.add = self.add_element
            elif self.add_as == 'l':
                self.add = self.add_list
            elif self.add_as == 't':
                self.add = self.add_batch
            else:
                raise ValueError('add_as')
        else:
            raise TypeError('add_as')

        if is_recurrent is None:
            self.is_recurrent = False
        elif isinstance(is_recurrent, bool):
            self.is_recurrent = is_recurrent
        else:
            raise TypeError('is_recurrent')

        if self.is_recurrent:
            if model is None:
                raise ValueError('model')
            else:
                self.model = model
        else:
            self.model = None

        self.current_len = 0

    def add(self, states, actions, rewards, next_states):
        return None

    def add_element(self, states, actions, rewards, next_states):

        self.states.append(states)

        self.actions.append(actions)

        self.rewards.append(rewards)

        self.next_states.append(next_states)

        self.current_len = len(self.states)
        self.remove_extras()

        return None

    def add_list(self, states, actions, rewards, next_states):

        self.states += states

        self.actions += actions

        self.rewards += rewards

        self.next_states += next_states

        self.current_len = len(self.states)
        self.remove_extras()

        return None

    def add_batch(self, states, actions, rewards, next_states):

        self.add_list(
            states=self.unbatch_states(states=states), actions=self.unbatch_actions(actions=actions),
            rewards=self.unbatch_rewards(rewards=rewards), next_states=self.unbatch_states(states=next_states))

        return None

    def unbatch_states(self, states):

        if self.is_recurrent:
            n_new_states = states[0].shape[self.state_batch_axis]
            idx = [slice(0, states[0].shape[a], 1) for a in range(0, states[0].ndim, 1)]

            new_states = np.empty(shape=[n_new_states, 2], dtype='O')
            # new_idx = [slice(0, new_states.shape[a], 1) for a in range(0, new_states.ndim, 1)]

            j = 0
            new_idx = [0, j]
            for i in range(0, n_new_states, 1):

                idx[self.state_batch_axis] = slice(i, i + 1, 1)

                new_idx[0] = i

                new_states[tuple(new_idx)] = states[j][tuple(idx)]

            j = 1
            new_idx = [slice(0, n_new_states, 1), j]
            new_states[tuple(new_idx)] = self.model.unbatch_h(h=states[j], axes=self.state_batch_axis, keepdims=True)

            return new_states
        else:
            return self._unbatch(samples=states, batch_axis=self.state_batch_axis)

    def unbatch_actions(self, actions):
        return self._unbatch(samples=actions, batch_axis=self.action_batch_axis)

    def unbatch_rewards(self, rewards):
        return self._unbatch(samples=rewards, batch_axis=self.reward_batch_axis)

    def _unbatch(self, samples, batch_axis):

        n_new_samples = samples.shape[batch_axis]
        new_samples = [None for i in range(0, n_new_samples, 1)]
        idx = [slice(0, samples.shape[a], 1) for a in range(0, samples.ndim, 1)]
        for i in range(0, n_new_samples, 1):
            idx[batch_axis] = slice(i, i + 1, 1)
            new_samples[i] = samples[tuple(idx)]
        return samples

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
        self.__init__(
            capacity=self.capacity, batch_size=self.batch_size, state_batch_axis=self.state_batch_axis,
            action_batch_axis=self.action_batch_axis, reward_batch_axis=self.reward_batch_axis,
            add_as=self.add_as, is_recurrent=self.is_recurrent, model=self.model)
        return None

    def sample(self):

        if self.batch_size > self.current_len:
            raise ValueError('self.batch_size > self.current_len')

        states = []
        actions = []
        rewards = []
        next_states = []
        for i in range(0, self.batch_size, 1):

            index = np.random.randint(low=0, high=self.current_len, size=None, dtype='i').tolist()

            states.append(self.states.pop(index))
            actions.append(self.actions.pop(index))
            rewards.append(self.rewards.pop(index))
            next_states.append(self.next_states.pop(index))

            if self.is_recurrent:
                if next_states[i][0] is None:
                    next_states[i][0] = torch.zeros(
                        size=states[i][0].shape, device=states[i][0].device,
                        dtype=states[i][0].dtype, requires_grad=False)
            else:
                if next_states[i] is None:
                    next_states[i] = torch.zeros(
                        size=states[i].shape, device=states[i].device, dtype=states[i].dtype, requires_grad=False)

            self.current_len = len(self.states)

        if self.is_recurrent:
            states = [
                torch.cat([states[i][0] for i in range(0, self.batch_size, 1)], dim=self.state_batch_axis),
                self.model.concatenate_hs([states[i][1] for i in range(0, self.batch_size, 1)], axis=self.state_batch_axis)]
            # device = states[0].device
            # dtype = states[0].dtype

            next_states = [
                torch.cat([next_states[i][0] for i in range(0, self.batch_size, 1)], dim=self.state_batch_axis),
                self.model.concatenate_hs([next_states[i][1] for i in range(0, self.batch_size, 1)], axis=self.state_batch_axis)]
            # todo: if lstm
        else:
            states = torch.cat(states, dim=self.state_batch_axis)
            next_states = torch.cat(next_states, dim=self.state_batch_axis)
            # device = states.device
            # dtype = states.dtype

        actions = torch.cat(actions, dim=self.action_batch_axis)

        rewards = torch.cat(rewards, dim=self.reward_batch_axis)
        # rewards = torch.tensor(data=rewards, device=device, dtype=dtype, requires_grad=False)

        return dict(states=states, actions=actions, rewards=rewards, next_states=next_states)

    def __len__(self) -> int:
        return len(self.states)


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


