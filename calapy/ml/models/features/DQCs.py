

import torch
import typing
import numpy as np
from . import no_task
from ...output_methods.features import DQCs


class DQC(no_task.LSTMSequentialParallelFCLs, DQCs.DQCMethods):

    def __init__(
            self,
            tasks: typing.Union[str, list, tuple, np.ndarray],
            possible_actions: [list, tuple],
            axis_models_losses: int,
            n_features_inputs_lstm: int,
            n_features_outs_lstm: int,
            n_features_non_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            n_features_parallel_fc_layers: typing.Union[int, list, tuple, np.ndarray, torch.Tensor],
            bias_lstm: typing.Union[bool, int] = True,
            biases_non_parallel_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            biases_parallel_fc_layers: typing.Union[bool, int, list, tuple, np.ndarray, torch.Tensor] = True,
            n_layers_lstm: int = 1,
            dropout_lstm: typing.Union[int, float] = 0,
            bidirectional_lstm: bool = False,
            batch_first: bool = True,
            return_hc: bool = True,
            movement_type: str = 'proactive',
            same_actions: typing.Union[int, list, tuple, np.ndarray, torch.Tensor, None] = None,
            gamma: typing.Union[int, float] = .999,
            reward_bias: typing.Union[int, float] = .0,
            loss_weights: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor, None] = None,
            device: typing.Union[torch.device, str, None] = None):

        name_superclass = DQC.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        no_task.LSTMSequentialParallelFCLs.__init__(
            self=self,
            n_features_inputs_lstm=n_features_inputs_lstm,
            n_features_outs_lstm=n_features_outs_lstm,
            n_features_non_parallel_fc_layers=n_features_non_parallel_fc_layers,
            n_features_parallel_fc_layers=n_features_parallel_fc_layers,
            bias_lstm=bias_lstm,
            biases_non_parallel_layers=biases_non_parallel_layers,
            biases_parallel_fc_layers=biases_parallel_fc_layers,
            n_layers_lstm=n_layers_lstm,
            dropout_lstm=dropout_lstm,
            bidirectional_lstm=bidirectional_lstm,
            batch_first=batch_first,
            return_hc=return_hc,
            device=device)

        self.axis_time_inputs = self.lstm.axis_time_inputs
        self.axis_batch_inputs = self.lstm.axis_batch_inputs
        self.axis_features_inputs = self.lstm.axis_features_inputs

        axis_batch_outs = self.axis_batch_inputs
        axis_features_outs = self.axis_features_inputs

        DQCs.DQCMethods.__init__(
            self=self,
            tasks=tasks,
            possible_actions=possible_actions,
            axis_batch_outs=axis_batch_outs,
            axis_features_outs=axis_features_outs,
            axis_models_losses=axis_models_losses,
            movement_type=movement_type,
            same_actions=same_actions,
            gamma=gamma,
            reward_bias=reward_bias,
            loss_weights=loss_weights)

        n_possible_actions = [
            self.parallel_fc_layers.n_features_last_layers[m] for m in range(0, self.M, 1) if tasks[m].upper() == 'A']

        if len(self.n_possible_actions) != len(n_possible_actions):
            raise ValueError('tasks, possible_actions, n_features_parallel_fc_layers')

        if any([self.n_possible_actions[a] != n_possible_actions[a] for a in range(0, self.A, 1)]):
            raise ValueError('tasks, possible_actions, n_features_parallel_fc_layers')

        self.set_device()

        self.superclasses_initiated.append(name_superclass)

