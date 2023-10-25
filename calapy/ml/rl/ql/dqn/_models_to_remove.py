

import torch
import numpy as np
from ....sl.dl.models.multi_layers.homo.rnns import SharedRNNAndIndRNNsAndIndFCNNs
from . import output_methods as cp_output_methods


class RecurrentDQN(SharedRNNAndIndRNNsAndIndFCNNs, cp_output_methods.TimedDQNMethods):

    def __init__(
            self,
            possible_actions, axis_features_outs, axis_models_losses,
            type_name, n_features_shared_rnn_layers, n_features_private_rnn_layers, n_features_private_fc_layers,
            biases_shared_rnn_layers=True, biases_private_rnn_layers=True, biases_private_fc_layers=True,
            h_sigma=0.1, nonlinearity='tanh', axis_time_outs=None,
            action_selection_type='active', same_indexes_actions=None,
            gamma=0.999, reward_bias=0.0, loss_scales_actors=None,
            device=None, dtype=None):

        superclass = RecurrentDQN
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if SharedRNNAndIndRNNsAndIndFCNNs not in self.superclasses_initiated:
            SharedRNNAndIndRNNsAndIndFCNNs.__init__(
                self=self, type_name=type_name, n_features_shared_rnn_layers=n_features_shared_rnn_layers,
                n_features_private_rnn_layers=n_features_private_rnn_layers,
                n_features_private_fc_layers=n_features_private_fc_layers,
                biases_shared_rnn_layers=biases_shared_rnn_layers, biases_private_rnn_layers=biases_private_rnn_layers,
                biases_private_fc_layers=biases_private_fc_layers,
                axis_features=axis_features_outs, axis_time=axis_time_outs, h_sigma=h_sigma, nonlinearity=nonlinearity,
                device=device, dtype=dtype)

            if SharedRNNAndIndRNNsAndIndFCNNs not in self.superclasses_initiated:
                self.superclasses_initiated.append(SharedRNNAndIndRNNsAndIndFCNNs)

        self.axis_time_inputs = self.lstm.axis_time_inputs
        self.axis_batch_inputs = self.lstm.axis_batch_inputs
        self.axis_features_inputs = self.lstm.axis_features_inputs

        axis_batch_outs = self.axis_batch_inputs
        axis_features_outs = self.axis_features_inputs

        if cp_output_methods.TimedDQNMethods not in self.superclasses_initiated:
            cp_output_methods.TimedDQNMethods.__init__(
                self=self,
                possible_actions=possible_actions,
                axis_time_outs=axis_time_outs,
                axis_batch_outs=axis_batch_outs,
                axis_features_outs=axis_features_outs,
                axis_models_losses=axis_models_losses,
                action_selection_type=action_selection_type,
                same_indexes_actions=same_indexes_actions,
                gamma=gamma,
                reward_bias=reward_bias,
                loss_scales_actors=loss_scales_actors)
            if cp_output_methods.TimedDQNMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(cp_output_methods.TimedDQNMethods)

        n_possible_actions_from_models = [
            self.private_fc_layers.n_features_last_layers[m] for m in range(0, self.private_fc_layers.M, 1)]

        if len(self.n_possible_actions) != len(n_possible_actions_from_models):
            raise ValueError('tasks, possible_actions, n_features_parallel_fc_layers')

        if any([self.n_possible_actions[a] != n_possible_actions_from_models[a] for a in range(0, self.A, 1)]):
            raise ValueError('tasks, possible_actions, n_features_parallel_fc_layers')

        if superclass == type(self):
            self.set_device()
            self.get_dtype()

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)
