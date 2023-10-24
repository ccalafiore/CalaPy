

import copy
import torch
from ....sl.dl.output_methods.general import *
import numpy as np
import typing

__all__ = ['DQNMethods', 'TimedDQNMethods']


class DQNMethods(OutputMethods):

    def __init__(
            self, axis_features_outs, axis_models_losses,
            possible_actions, action_selection_type='active', same_indexes_actions=None,
            gamma=0.999, reward_bias=0.0, loss_scales_actors=None):

        """

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
            self, value_action_losses: typing.Union[torch.Tensor, np.ndarray],
            axes_not_included: typing.Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
            scaled: bool = False,
            loss_scales_actors: typing.Union[list, tuple, np.ndarray, torch.Tensor] = None,
            format_scales: bool = True):

        if scaled and (loss_scales_actors is None):
            loss_scales_actors = self.loss_scales_actors
            format_scales = False

        reduced_value_action_losses = self.reduce_losses(
            losses=value_action_losses, axes_not_included=axes_not_included,
            scaled=scaled, loss_scales=loss_scales_actors, format_scales=format_scales)

        return reduced_value_action_losses

    def compute_n_selected_actions(
            self, selected_actions, axes_not_included: typing.Union[int, list, tuple, np.ndarray, torch.Tensor] = None):

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


