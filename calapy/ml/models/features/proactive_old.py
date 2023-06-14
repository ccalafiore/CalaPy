

from ... import torch as pt
from ....ml import devices as cp_device
from ..model_tools import ModelMethods as cc_ModelMethods
from ....maths import prod as cp_prod
import numpy as np
import math
import typing


class ProactiveFeatureSequenceClassifier(cc_ModelMethods):

    def __init__(
            self, n_features_inputs_lstm: int, n_features_outs_lstm: int,
            n_features_outs_actors: typing.Union[int, list, tuple, np.ndarray, pt.Tensor],
            n_features_outs_layers: typing.Union[int, list, tuple, np.ndarray, pt.Tensor],
            n_layers_lstm: int = 1, bias_lstm: typing.Union[bool, int] = True, batch_first_lstm: bool = False,
            dropout_lstm: typing.Union[int, float] = 0, bidirectional_lstm: bool = False,
            biases_actors: typing.Union[bool, int, list, tuple, np.ndarray, pt.Tensor] = True,
            biases_layers: typing.Union[bool, int, list, tuple, np.ndarray, pt.Tensor] = True,
            loss_weights_types: typing.Union[list, tuple, np.ndarray, pt.Tensor, None] = None,
            loss_weights_actors: typing.Union[int, float, list, tuple, np.ndarray, pt.Tensor, None] = None,
            loss_weights_classifiers: typing.Union[int, float, list, tuple, np.ndarray, pt.Tensor, None] = None,
            gamma=.999, movement_type='proactive', reward_bias=1,
            device: typing.Union[pt.device, str, None] = None) -> None:

        super(ProactiveFeatureSequenceClassifier, self).__init__()

        self.n_features_inputs_lstm = n_features_inputs_lstm
        self.n_features_outs_lstm = n_features_outs_lstm

        self.n_layers_lstm = n_layers_lstm

        if isinstance(bias_lstm, bool):
            self.bias_lstm = bias_lstm
        elif isinstance(bias_lstm, int):
            self.bias_lstm = bool(bias_lstm)
        else:
            raise TypeError('bias_lstm = ' + str(bias_lstm))

        self.batch_first_inputs = self.batch_first_outs = self.batch_first_lstm = batch_first_lstm

        self.n_axes_inputs = 3
        self.axes_inputs = np.arange(0, self.n_axes_inputs, 1, dtype='i')
        if self.batch_first_inputs:
            self.axis_batch_inputs = 0
            self.axis_time_inputs = 1
        else:
            self.axis_batch_inputs = 1
            self.axis_time_inputs = 0
        self.axis_features_inputs = 2

        self.n_axes_outs = 3
        self.axes_outs = np.arange(0, self.n_axes_outs, 1, dtype='i')
        self.axis_time_outs = self.axis_time_inputs
        self.axis_batch_outs = self.axis_batch_inputs
        self.axis_features_outs = self.axis_features_inputs

        self.n_axes_losses = 3
        self.axes_losses = np.arange(0, self.n_axes_losses, 1, dtype='i')

        self.axis_models_losses = 0
        self.axis_time_losses = self.axis_time_outs
        if self.axis_time_outs > self.axis_features_outs:
            self.axis_time_losses -= 1
        if self.axis_time_losses >= self.axis_models_losses:
            self.axis_time_losses += 1

        self.axis_batch_losses = self.axis_batch_outs
        if self.axis_batch_outs > self.axis_features_outs:
            self.axis_batch_losses -= 1
        if self.axis_batch_losses >= self.axis_models_losses:
            self.axis_batch_losses += 1

        self.axis_batch_losses_trials = 0
        self.axis_models_losses_trials = 1
        self.axis_time_losses_trials = 2
        self.n_axes_losses_trials = self.n_axes_losses
        self.axes_losses_trials = np.arange(0, self.n_axes_losses_trials, 1, dtype='i')
        self.axes_losses_trials_in_losses = [self.axis_batch_losses, self.axis_models_losses, self.axis_time_losses]
        self.destination_axes_losses_trials = [
            a for a in range(0, self.n_axes_losses_trials, 1) if a != self.axes_losses_trials_in_losses[a]]
        self.source_axes_losses_trials = [
            self.axes_losses_trials_in_losses[a] for a in self.destination_axes_losses_trials]
        self.n_moves_axes_losses_trials = len(self.source_axes_losses_trials)
        self.move_axes_losses_trials = self.n_moves_axes_losses_trials > 0

        self.axis_batch_outs_trials = 0
        self.axis_features_outs_trials = 1
        self.axis_time_outs_trials = 2
        self.n_axes_outs_trials = self.n_axes_outs
        self.axes_outs_trials = np.arange(0, self.n_axes_outs_trials, 1, dtype='i')
        self.axes_outs_trials_in_outs = [self.axis_batch_outs, self.axis_features_outs, self.axis_time_outs]
        self.destination_axes_outs_trials = [
            a for a in range(0, self.n_axes_outs_trials, 1) if a != self.axes_outs_trials_in_outs[a]]
        self.source_axes_outs_trials = [
            self.axes_outs_trials_in_outs[a] for a in self.destination_axes_outs_trials]
        self.n_moves_axes_outs_trials = len(self.source_axes_outs_trials)
        self.move_axes_outs_trials = self.n_moves_axes_outs_trials > 0

        self.dropout_lstm = dropout_lstm

        self.bidirectional_lstm = bidirectional_lstm
        if self.bidirectional_lstm:
            self.num_directions_lstm = 2
        else:
            self.num_directions_lstm = 1

        self.out_features_lstm = self.n_features_outs_lstm * self.num_directions_lstm

        self.device = cp_device.define_device(device)

        # lstm tutorials at:
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

        self.lstm = pt.nn.LSTM(
            self.n_features_inputs_lstm, self.n_features_outs_lstm,
            num_layers=self.n_layers_lstm, bias=self.bias_lstm,
            batch_first=self.batch_first_lstm, dropout=self.dropout_lstm,
            bidirectional=self.bidirectional_lstm, device=self.device)
        
        self.lstm.to(self.device)

        if isinstance(n_features_outs_actors, int):
            self.n_features_outs_actors = [n_features_outs_actors]
        elif isinstance(n_features_outs_actors, list):
            self.n_features_outs_actors = n_features_outs_actors
        elif isinstance(n_features_outs_actors, tuple):
            self.n_features_outs_actors = list(n_features_outs_actors)
        elif isinstance(n_features_outs_actors, (np.ndarray, pt.Tensor)):
            self.n_features_outs_actors = n_features_outs_actors.tolist()
        else:
            raise TypeError('n_features_outs_actors')

        self.n_dimensions_actions = self.A = len(self.n_features_outs_actors)

        if isinstance(n_features_outs_layers, int):
            self.n_features_outs_layers = [n_features_outs_layers]
        elif isinstance(n_features_outs_layers, list):
            self.n_features_outs_layers = n_features_outs_layers
        elif isinstance(n_features_outs_layers, tuple):
            self.n_features_outs_layers = list(n_features_outs_layers)
        elif isinstance(n_features_outs_layers, (np.ndarray, pt.Tensor)):
            self.n_features_outs_layers = n_features_outs_layers.tolist()
        else:
            raise TypeError('n_features_outs_layers')

        self.n_dimensions_classes = self.C = len(self.n_features_outs_layers)

        self.n_outputs = self.O = self.A + self.C

        if isinstance(biases_actors, bool):
            self.biases_actors = [biases_actors]
        elif isinstance(biases_actors, int):
            self.biases_actors = [bool(biases_actors)]
        elif isinstance(biases_actors, list):
            self.biases_actors = biases_actors
        elif isinstance(biases_actors, tuple):
            self.biases_actors = list(biases_actors)
        elif isinstance(biases_actors, (np.ndarray, pt.Tensor)):
            self.biases_actors = biases_actors.tolist()
        else:
            raise TypeError('biases_actors')

        if len(self.biases_actors) != self.A:
            if len(self.biases_actors) == 1:
                self.biases_actors = [self.biases_actors[0] for a in range(self.A)]
            else:
                raise ValueError('biases_actors = ' + str(biases_actors))

        if isinstance(biases_layers, bool):
            self.biases_layers = [biases_layers]
        elif isinstance(biases_layers, int):
            self.biases_layers = [bool(biases_layers)]
        elif isinstance(biases_layers, list):
            self.biases_layers = biases_layers
        elif isinstance(biases_layers, tuple):
            self.biases_layers = list(biases_layers)
        elif isinstance(biases_layers, (np.ndarray, pt.Tensor)):
            self.biases_layers = biases_layers.tolist()
        else:
            raise TypeError('biases_layers')

        if len(self.biases_layers) != self.C:
            if len(self.biases_layers) == 1:
                self.biases_layers = [self.biases_layers[0] for c in range(self.C)]
            else:
                raise ValueError('biases_layers = ' + str(biases_layers))

        self.n_out_types = 2

        if loss_weights_types is None:
            loss_weight_type_i = 1.0 / self.n_out_types
            self.loss_weights_types = [loss_weight_type_i for i in range(self.n_out_types)]
        elif isinstance(loss_weights_types, list):
            self.loss_weights_types = loss_weights_types
        elif isinstance(loss_weights_types, tuple):
            self.loss_weights_types = list(loss_weights_types)
        elif isinstance(loss_weights_types, (np.ndarray, pt.Tensor)):
            self.loss_weights_types = loss_weights_types.tolist()
        else:
            raise TypeError('loss_weights_types')

        if len(self.loss_weights_types) != self.n_out_types:
            raise ValueError('loss_weights_types = ' + str(loss_weights_types))

        sum_loss_weights_types = sum(self.loss_weights_types)
        self.loss_weights_types = [(self.loss_weights_types[i] / sum_loss_weights_types) for i in range(self.n_out_types)]
        self.sum_loss_weights_types = sum(self.loss_weights_types)

        if loss_weights_actors is None:
            loss_weight_actor_a = 1.0 / self.A
            self.loss_weights_actors = [loss_weight_actor_a for a in range(self.A)]
        elif isinstance(loss_weights_actors, int):
            self.loss_weights_actors = [float(loss_weights_actors)]
        elif isinstance(loss_weights_actors, float):
            self.loss_weights_actors = [loss_weights_actors]
        elif isinstance(loss_weights_actors, list):
            self.loss_weights_actors = loss_weights_actors
        elif isinstance(loss_weights_actors, tuple):
            self.loss_weights_actors = list(loss_weights_actors)
        elif isinstance(loss_weights_actors, (np.ndarray, pt.Tensor)):
            self.loss_weights_actors = loss_weights_actors.tolist()
        else:
            raise TypeError('loss_weights_actors')

        if len(self.loss_weights_actors) != self.A:
            if len(self.loss_weights_actors) == 1:
                self.loss_weights_actors = [self.loss_weights_actors[0] for a in range(self.A)]
            else:
                raise ValueError('loss_weights_actors = ' + str(loss_weights_actors))

        sum_loss_weights_actors = sum(self.loss_weights_actors)
        self.loss_weights_actors = [(self.loss_weights_actors[a] / sum_loss_weights_actors) for a in range(self.A)]
        self.sum_loss_weights_actors = sum(self.loss_weights_actors)

        if loss_weights_classifiers is None:
            loss_weight_classifier_c = 1.0 / self.C
            self.loss_weights_classifiers = [loss_weight_classifier_c for c in range(self.C)]
        elif isinstance(loss_weights_classifiers, int):
            self.loss_weights_classifiers = [float(loss_weights_classifiers)]
        elif isinstance(loss_weights_classifiers, float):
            self.loss_weights_classifiers = [loss_weights_classifiers]
        elif isinstance(loss_weights_classifiers, list):
            self.loss_weights_classifiers = loss_weights_classifiers
        elif isinstance(loss_weights_classifiers, tuple):
            self.loss_weights_classifiers = list(loss_weights_classifiers)
        elif isinstance(loss_weights_classifiers, (np.ndarray, pt.Tensor)):
            self.loss_weights_classifiers = loss_weights_classifiers.tolist()
        else:
            raise TypeError('loss_weights_classifiers')

        if len(self.loss_weights_classifiers) != self.C:
            if len(self.loss_weights_classifiers) == 1:
                self.loss_weights_classifiers = [self.loss_weights_classifiers[0] for c in range(self.C)]
            else:
                raise ValueError('loss_weights_classifiers = ' + str(loss_weights_classifiers))

        sum_loss_weights_classifiers = sum(self.loss_weights_classifiers)
        self.loss_weights_classifiers = [
            (self.loss_weights_classifiers[c] / sum_loss_weights_classifiers) for c in range(self.C)]
        self.sum_loss_weights_classifiers = sum(self.loss_weights_classifiers)

        self.in_features_actors = [self.out_features_lstm for a in range(self.A)]
        self.out_features_actors = self.n_features_outs_actors

        self.actors = pt.nn.ModuleList([pt.nn.Linear(
            self.in_features_actors[a], self.out_features_actors[a],
            bias=self.biases_actors[a], device=self.device) for a in range(0, self.A, 1)])
        # self.actors = pt.nn.ModuleList([pt.nn.Linear(
        #     self.in_features_actors[a], self.out_features_actors[a],
        #     bias=self.biases_actors[a]) for a in range(0, self.A, 1)])
        # self.actors.to(self.device)

        self.n_features_inputs_models = [self.out_features_lstm for c in range(self.C)]
        self.n_features_outs_layers = self.n_features_outs_layers

        self.classifiers = pt.nn.ModuleList([pt.nn.Linear(
            self.n_features_inputs_models[c], self.n_features_outs_layers[c],
            bias=self.biases_layers[c], device=self.device) for c in range(0, self.C, 1)])
        # self.classifiers = pt.nn.ModuleList([pt.nn.Linear(
        #     self.n_features_inputs_models[c], self.n_features_outs_layers[c],
        #     bias=self.biases_layers[c]) for c in range(0, self.C, 1)])
        # self.classifiers.to(self.device)

        self.criterion_values_actions = pt.nn.SmoothL1Loss(reduction='none')
        self.criterion_values_actions_reduction = pt.nn.SmoothL1Loss(reduction='mean')

        self.criterion_predictions_classes = pt.nn.CrossEntropyLoss(reduction='none')
        self.criterion_predictions_classes_reduction = pt.nn.CrossEntropyLoss(reduction='mean')

        self.softmax = pt.nn.Softmax(dim=self.axis_features_outs)

        if isinstance(movement_type, str):
            if movement_type.lower() in ['proactive', 'random', 'passive', 'only_left', 'only_right']:
                self.movement_type = movement_type.lower()
            else:
                raise ValueError('movement_type')
        else:
            raise TypeError('movement_type')

        self.gamma = gamma

        self.reward_bias = reward_bias

        self.to(self.device)

    def forward(self, x, h=None, c=None):

        if h is None:
            batch_size = x.shape[self.axis_batch_inputs]
            if c is None:
                h, c = self.init_hidden_state(batch_size)
            else:
                h = self.init_h(batch_size)
        elif c is None:
            batch_size = x.shape[self.axis_batch_inputs]
            c = self.init_c(batch_size)

        x, (h, c) = self.lstm(x, (h, c))

        values_actions = [self.actors[a](x) for a in range(0, self.A, 1)]

        predictions_classes = [self.classifiers[c](x) for c in range(0, self.C, 1)]

        return values_actions, predictions_classes, (h, c)

    def init_h(self, batch_size):

        h = pt.zeros(
            [self.num_directions_lstm * self.n_layers_lstm, batch_size, self.n_features_outs_lstm],
            device=self.device)

        return h

    def init_c(self, batch_size):

        c = pt.zeros(
            [self.num_directions_lstm * self.n_layers_lstm, batch_size, self.n_features_outs_lstm],
            device=self.device)

        return c

    def init_hidden_state(self, batch_size):

        h = self.init_h(batch_size)
        c = self.init_c(batch_size)

        return h, c

    def compute_probabilities(self, predictions_classes):

        probabilities = [self.softmax(predictions_classes[c]) for c in range(0, self.C, 1)]

        return probabilities

    def sample_action(self, x_t: pt.Tensor, h=None, c=None, epsilon=.1):

        # self.eval()
        # pt.set_grad_enabled(False)
        values_actions, predictions_classes, (h, c) = self(x_t, h=h, c=c)

        # 1)
        # shape_actions = np.asarray(x_t.shape, dtype='i')
        # shape_actions = shape_actions[
        #     np.arange(0, len(shape_actions), 1, dtype='i') != self.axis_features_outs].tolist()
        # 2)
        # shape_actions = list(x_t.shape)
        # shape_actions[self.axis_features_outs] = 1
        # 3)
        shape_actions = [x_t.shape[a] for a in range(0, x_t.ndim, 1) if a != self.axis_features_outs]

        if self.movement_type == 'proactive':

            mask_randoms = pt.rand(
                shape_actions, out=None, dtype=None, layout=pt.strided,
                device=self.device, requires_grad=False) < epsilon

            n_randoms = mask_randoms.sum(dtype=None).item()

            mask_greedy = pt.logical_not(mask_randoms, out=None)

            actions = [None for a in range(0, self.A, 1)]  # type: list

            for a in range(0, self.A, 1):

                actions[a] = pt.empty(shape_actions, dtype=pt.int64, device=self.device, requires_grad=False)

                random_action_a = pt.randint(
                    low=0, high=self.actors[a].out_features, size=(n_randoms,),
                    generator=None, dtype=pt.int64, device=self.device, requires_grad=False)

                actions[a][mask_randoms] = random_action_a

                actions[a][mask_greedy] = (
                    # values_actions[a].max(dim=self.axis_features_outs, keepdim=True)[1][mask_greedy])
                    values_actions[a].max(dim=self.axis_features_outs, keepdim=False)[1][mask_greedy])

        elif self.movement_type == 'random':

            actions = [pt.randint(
                low=0, high=self.actors[a].out_features, size=shape_actions,
                generator=None, dtype=pt.int64, device=self.device, requires_grad=False)
                for a in range(0, self.A, 1)]

        elif self.movement_type == 'passive':

            actions = [pt.full(
                size=shape_actions, fill_value=math.floor(self.actors[a].out_features / 2),
                dtype=pt.int64, device=self.device, requires_grad=False)
                for a in range(0, self.A, 1)]

        elif self.movement_type == 'only_left':

            actions = [None for a in range(0, self.A, 1)]  # type: list

            for a in range(0, self.A, 1):

                if a == 0:
                    actions[a] = pt.zeros(
                        size=shape_actions, dtype=pt.int64, device=self.device, requires_grad=False)

                else:
                    actions[a] = pt.full(
                        size=shape_actions, fill_value=math.floor(self.actors[a].out_features / 2),
                        dtype=pt.int64, device=self.device, requires_grad=False)

        elif self.movement_type == 'only_right':

            actions = [None for a in range(0, self.A, 1)]  # type: list

            for a in range(0, self.A, 1):

                if a == 0:
                    actions[a] = pt.full(
                        size=shape_actions, fill_value=self.actors[a].out_features - 1,
                        dtype=pt.int64, device=self.device, requires_grad=False)

                else:
                    actions[a] = pt.full(
                        size=shape_actions, fill_value=math.floor(self.actors[a].out_features / 2),
                        dtype=pt.int64, device=self.device, requires_grad=False)
        else:
            raise ValueError('self.movement_type')

        return actions, predictions_classes, (h, c)

    def compute_unweighted_value_action_losses_ABT(self, values_selected_actions, expected_values_actions):

        unweighted_value_action_losses_ABT = [self.criterion_values_actions(
            values_selected_actions[a], expected_values_actions[a]) for a in range(0, self.A, 1)]

        return unweighted_value_action_losses_ABT

    def compute_unweighted_value_action_losses_BT(self, values_selected_actions, expected_values_actions):

        if self.A > 0:
            weight = (1 / self.A)

            unweighted_value_action_losses_ABT = self.compute_unweighted_value_action_losses_ABT(
                values_selected_actions, expected_values_actions)

            unweighted_value_action_losses_BT = (unweighted_value_action_losses_ABT[0] * weight)
            for a in range(1, self.A, 1):
                unweighted_value_action_losses_BT += (unweighted_value_action_losses_ABT[a] * weight)
        else:
            unweighted_value_action_losses_BT = 0

        return unweighted_value_action_losses_BT

    def compute_unweighted_value_action_losses_AT(self, values_selected_actions, expected_values_actions):

        unweighted_value_action_losses_ABT = self.compute_unweighted_value_action_losses_ABT(
            values_selected_actions, expected_values_actions)

        unweighted_value_action_losses_AT = [
            unweighted_value_action_losses_ABT[a].mean(dim=self.axis_batch_outs) for a in range(0, self.A, 1)]

        return unweighted_value_action_losses_AT

    def compute_unweighted_value_action_losses_T(self, values_selected_actions, expected_values_actions):
        if self.A > 0:
            weight = (1 / self.A)

            unweighted_value_action_losses_AT = self.compute_unweighted_value_action_losses_AT(
                values_selected_actions, expected_values_actions)

            unweighted_value_action_losses_T = (unweighted_value_action_losses_AT[0] * weight)
            for a in range(1, self.A, 1):
                unweighted_value_action_losses_T += (unweighted_value_action_losses_AT[a] * weight)
        else:
            unweighted_value_action_losses_T = 0

        return unweighted_value_action_losses_T

    def compute_unweighted_value_action_losses_A(self, values_selected_actions, expected_values_actions):

        unweighted_value_action_losses_A = [self.criterion_values_actions_reduction(
            values_selected_actions[a], expected_values_actions[a]) for a in range(0, self.A, 1)]

        return unweighted_value_action_losses_A

    def compute_unweighted_value_action_loss(self, values_selected_actions, expected_values_actions):
        if self.A > 0:
            weight = (1 / self.A)
            unweighted_value_action_losses_A = self.compute_unweighted_value_action_losses_A(
                values_selected_actions, expected_values_actions)

            unweighted_value_action_loss = unweighted_value_action_losses_A[0] * weight
            for a in range(1, self.A, 1):
                unweighted_value_action_loss += unweighted_value_action_losses_A[a] * weight
        else:
            unweighted_value_action_loss = 0

        return unweighted_value_action_loss

    def compute_weighted_value_action_losses_ABT(self, values_selected_actions, expected_values_actions):

        unweighted_value_action_losses_ABT = self.compute_unweighted_value_action_losses_ABT(
            values_selected_actions, expected_values_actions)

        weighted_value_action_losses_ABT = [
            (unweighted_value_action_losses_ABT[a] * self.loss_weights_actors[a]) for a in range(0, self.A, 1)]

        return weighted_value_action_losses_ABT

    def compute_weighted_value_action_losses_BT(self, values_selected_actions, expected_values_actions):

        if self.A > 0:
            unweighted_value_action_losses_ABT = self.compute_unweighted_value_action_losses_ABT(
                values_selected_actions, expected_values_actions)

            weighted_value_action_losses_BT = (unweighted_value_action_losses_ABT[0] * self.loss_weights_actors[0])
            for a in range(1, self.A, 1):
                weighted_value_action_losses_BT += (unweighted_value_action_losses_ABT[a] * self.loss_weights_actors[a])
        else:
            weighted_value_action_losses_BT = 0

        return weighted_value_action_losses_BT

    def compute_weighted_value_action_losses_AT(self, values_selected_actions, expected_values_actions):

        unweighted_value_action_losses_AT = self.compute_unweighted_value_action_losses_AT(
            values_selected_actions, expected_values_actions)

        weighted_value_action_losses_AT = [
            (unweighted_value_action_losses_AT[a] * self.loss_weights_actors[a]) for a in range(0, self.A, 1)]

        return weighted_value_action_losses_AT

    def compute_weighted_value_action_losses_T(self, values_selected_actions, expected_values_actions):

        if self.A > 0:

            unweighted_value_action_losses_AT = self.compute_unweighted_value_action_losses_AT(
                values_selected_actions, expected_values_actions)

            weighted_value_action_losses_T = (unweighted_value_action_losses_AT[0] * self.loss_weights_actors[0])
            for a in range(1, self.A, 1):
                weighted_value_action_losses_T += (unweighted_value_action_losses_AT[a] * self.loss_weights_actors[a])
        else:
            weighted_value_action_losses_T = 0

        return weighted_value_action_losses_T

    def compute_weighted_value_action_losses_A(self, values_selected_actions, expected_values_actions):

        unweighted_value_action_losses_A = self.compute_unweighted_value_action_losses_A(
            values_selected_actions, expected_values_actions)

        weighted_value_action_losses_A = [
            (unweighted_value_action_losses_A[a] * self.loss_weights_actors[a]) for a in range(0, self.A, 1)]

        return weighted_value_action_losses_A

    def compute_weighted_value_action_loss(self, values_selected_actions, expected_values_actions):
        if self.A > 0:

            unweighted_value_action_losses_A = self.compute_unweighted_value_action_losses_A(
                values_selected_actions, expected_values_actions)

            weighted_value_action_loss = (unweighted_value_action_losses_A[0] * self.loss_weights_actors[0])
            for a in range(1, self.A, 1):
                weighted_value_action_loss += (unweighted_value_action_losses_A[a] * self.loss_weights_actors[a])
        else:
            weighted_value_action_loss = 0

        return weighted_value_action_loss

    def compute_unweighted_class_prediction_losses_CBT(self, predictions_classes, labels, axis_features=None):

        if axis_features is None:
            axis_features = self.axis_features_outs

        if axis_features == 1:
            unweighted_class_prediction_losses_CBT = [
                self.criterion_predictions_classes(predictions_classes[c], labels[c]) for c in range(0, self.C, 1)]
        else:
            unweighted_class_prediction_losses_CBT = [self.criterion_predictions_classes(
                pt.movedim(predictions_classes[c], axis_features, 1), labels[c]) for c in range(0, self.C, 1)]

        return unweighted_class_prediction_losses_CBT

    def compute_unweighted_class_prediction_losses_BT(self, predictions_classes, labels, axis_features=None):

        if self.C > 0:
            weight = (1 / self.C)

            unweighted_class_prediction_losses_CBT = self.compute_unweighted_class_prediction_losses_CBT(
                predictions_classes, labels, axis_features=axis_features)

            unweighted_class_prediction_losses_BT = (unweighted_class_prediction_losses_CBT[0] * weight)
            for c in range(1, self.C, 1):
                unweighted_class_prediction_losses_BT += (unweighted_class_prediction_losses_CBT[c] * weight)
        else:
            unweighted_class_prediction_losses_BT = 0

        return unweighted_class_prediction_losses_BT

    def compute_unweighted_class_prediction_losses_CT(self, predictions_classes, labels, axis_features=None, axis_batch=None):

        unweighted_class_prediction_losses_CBT = self.compute_unweighted_class_prediction_losses_CBT(
            predictions_classes, labels, axis_features=axis_features)

        if axis_batch is None:
            axis_batch = self.axis_batch_outs

        unweighted_class_prediction_losses_CT = [
            unweighted_class_prediction_losses_CBT[c].mean(dim=axis_batch) for c in range(0, self.C, 1)]

        return unweighted_class_prediction_losses_CT

    def compute_unweighted_class_prediction_losses_T(self, predictions_classes, labels, axis_features=None, axis_batch=None):
        if self.C > 0:
            weight = (1 / self.C)

            unweighted_class_prediction_losses_CT = self.compute_unweighted_class_prediction_losses_CT(
                predictions_classes, labels, axis_features=axis_features, axis_batch=axis_batch)

            unweighted_class_prediction_losses_T = (unweighted_class_prediction_losses_CT[0] * weight)
            for c in range(1, self.C, 1):
                unweighted_class_prediction_losses_T += (unweighted_class_prediction_losses_CT[c] * weight)
        else:
            unweighted_class_prediction_losses_T = 0

        return unweighted_class_prediction_losses_T

    def compute_unweighted_class_prediction_losses_C(self, predictions_classes, labels, axis_features=None):

        if axis_features is None:
            axis_features = self.axis_features_outs

        if axis_features == 1:
            unweighted_class_prediction_losses_C = [
                self.criterion_predictions_classes_reduction(predictions_classes[c], labels[c]) for c in range(0, self.C, 1)]
        else:
            unweighted_class_prediction_losses_C = [self.criterion_predictions_classes_reduction(
                pt.movedim(predictions_classes[c], axis_features, 1), labels[c]) for c in range(0, self.C, 1)]

        return unweighted_class_prediction_losses_C

    def compute_unweighted_class_prediction_loss(self, predictions_classes, labels, axis_features=None):
        if self.C > 0:
            weight = (1 / self.C)
            unweighted_class_prediction_losses_C = self.compute_unweighted_class_prediction_losses_C(
                predictions_classes, labels, axis_features=axis_features)

            unweighted_class_prediction_loss = unweighted_class_prediction_losses_C[0] * weight
            for c in range(1, self.C, 1):
                unweighted_class_prediction_loss += unweighted_class_prediction_losses_C[c] * weight
        else:
            unweighted_class_prediction_loss = 0

        return unweighted_class_prediction_loss

    def compute_weighted_class_prediction_losses_CBT(self, predictions_classes, labels, axis_features=None):

        unweighted_class_prediction_losses_CBT = self.compute_unweighted_class_prediction_losses_CBT(
            predictions_classes, labels, axis_features=axis_features)

        weighted_class_prediction_losses_CBT = [
            (unweighted_class_prediction_losses_CBT[c] * self.loss_weights_classifiers[c]) for c in range(0, self.C, 1)]

        return weighted_class_prediction_losses_CBT

    def compute_weighted_class_prediction_losses_BT(self, predictions_classes, labels, axis_features=None):

        if self.C > 0:
            unweighted_class_prediction_losses_CBT = self.compute_unweighted_class_prediction_losses_CBT(
                predictions_classes, labels, axis_features=axis_features)

            weighted_class_prediction_losses_BT = (
                    unweighted_class_prediction_losses_CBT[0] * self.loss_weights_classifiers[0])
            for c in range(1, self.C, 1):
                weighted_class_prediction_losses_BT += (
                        unweighted_class_prediction_losses_CBT[c] * self.loss_weights_classifiers[c])
        else:
            weighted_class_prediction_losses_BT = 0

        return weighted_class_prediction_losses_BT

    def compute_weighted_class_prediction_losses_CT(self, predictions_classes, labels, axis_features=None, axis_batch=None):

        unweighted_class_prediction_losses_CT = self.compute_unweighted_class_prediction_losses_CT(
            predictions_classes, labels, axis_features=axis_features, axis_batch=axis_batch)

        weighted_class_prediction_losses_CT = [
            (unweighted_class_prediction_losses_CT[c] * self.loss_weights_classifiers[c]) for c in range(0, self.C, 1)]

        return weighted_class_prediction_losses_CT

    def compute_weighted_class_prediction_losses_T(self, predictions_classes, labels, axis_features=None, axis_batch=None):

        if self.C > 0:

            unweighted_class_prediction_losses_CT = self.compute_unweighted_class_prediction_losses_CT(
                predictions_classes, labels, axis_features=axis_features, axis_batch=axis_batch)

            weighted_class_prediction_losses_T = (
                    unweighted_class_prediction_losses_CT[0] * self.loss_weights_classifiers[0])
            for c in range(1, self.C, 1):
                weighted_class_prediction_losses_T += (
                        unweighted_class_prediction_losses_CT[c] * self.loss_weights_classifiers[c])
        else:
            weighted_class_prediction_losses_T = 0

        return weighted_class_prediction_losses_T

    def compute_weighted_class_prediction_losses_C(self, predictions_classes, labels, axis_features=None):

        unweighted_class_prediction_losses_C = self.compute_unweighted_class_prediction_losses_C(
            predictions_classes, labels, axis_features=axis_features)

        weighted_class_prediction_losses_C = [
            (unweighted_class_prediction_losses_C[c] * self.loss_weights_classifiers[c]) for c in range(0, self.C, 1)]

        return weighted_class_prediction_losses_C

    def compute_weighted_class_prediction_loss(self, predictions_classes, labels, axis_features=None):
        if self.C > 0:

            unweighted_class_prediction_losses_C = self.compute_unweighted_class_prediction_losses_C(
                predictions_classes, labels, axis_features=axis_features)

            weighted_class_prediction_loss = (
                    unweighted_class_prediction_losses_C[0] * self.loss_weights_classifiers[0])
            for c in range(1, self.C, 1):
                weighted_class_prediction_loss += (
                        unweighted_class_prediction_losses_C[c] * self.loss_weights_classifiers[c])
        else:
            weighted_class_prediction_loss = 0

        return weighted_class_prediction_loss

    def get_previous_rewards(self, predictions_classes, labels, axis_features=None):

        weighted_class_prediction_losses_BT = self.compute_weighted_class_prediction_losses_BT(
            predictions_classes, labels, axis_features=axis_features)

        previous_rewards = - weighted_class_prediction_losses_BT + self.reward_bias

        return previous_rewards

    def compute_expected_values_actions(self, next_states, rewards):

        # samples = replay_memory.sample()
        # states = samples['states']
        # states_labels = samples['states_labels']
        # actions = samples['actions']
        # next_states = samples['next_states']
        # rewards = samples['rewards']
        # # non_final = samples['non_final']

        next_values_actions, next_predictions_classes, (next_h, next_c) = self(next_states)

        max_next_values_actions = [
            next_values_actions_a.max(dim=self.axis_features_outs, keepdim=False)[0].detach()
            for next_values_actions_a in next_values_actions]

        expected_values_actions = [
            (rewards + (self.gamma * max_next_values_actions_a))
            for max_next_values_actions_a in max_next_values_actions]

        return expected_values_actions

    def remove_last_values_actions(self, values_actions: list):

        values_actions_out = [None for a in range(0, self.A, 1)]
        for a in range(self.A):
            ind = tuple(
                [slice(0, values_actions[a].shape[d], 1) if d != self.axis_time_outs
                 else slice(0, values_actions[a].shape[d] - 1, 1) for d in range(0, values_actions[a].ndim, 1)])

            values_actions_out[a] = values_actions[a][ind]

        return values_actions_out

    def gather_values_selected_actions(self, values_actions, actions):

        # values_selected_actions = [values_actions[a].gather(self.axis_features_outs, actions[a]).squeeze(
        #     dim=self.axis_features_outs) for a in range(0, self.A, 1)]

        values_selected_actions = [values_actions[a].gather(self.axis_features_outs, actions[a].unsqueeze(
            dim=self.axis_features_outs)).squeeze(dim=self.axis_features_outs) for a in range(0, self.A, 1)]

        return values_selected_actions

    def compute_unweighted_losses(
            self,
            values_selected_actions, expected_values_actions,
            predictions_classes, labels):

        unweighted_value_action_loss = self.compute_unweighted_value_action_loss(
            values_selected_actions=values_selected_actions, expected_values_actions=expected_values_actions)

        unweighted_class_prediction_loss = self.compute_unweighted_class_prediction_loss(
            predictions_classes=predictions_classes, labels=labels)

        weight = 1 / self.n_out_types

        unweighted_loss = ((unweighted_value_action_loss * weight) + (unweighted_class_prediction_loss * weight))

        return unweighted_loss, unweighted_value_action_loss, unweighted_class_prediction_loss

    def compute_weighted_losses(
            self,
            values_selected_actions, expected_values_actions,
            predictions_classes, labels):

        weighted_value_action_loss = self.compute_weighted_value_action_loss(
            values_selected_actions=values_selected_actions, expected_values_actions=expected_values_actions)#.detach()

        weighted_class_prediction_loss = self.compute_weighted_class_prediction_loss(
            predictions_classes=predictions_classes, labels=labels)

        weighted_loss = ((weighted_value_action_loss * self.loss_weights_types[0]) +
                         (weighted_class_prediction_loss * self.loss_weights_types[1]))

        # weighted_loss = weighted_class_prediction_loss #* self.loss_weights_types[1]

        return weighted_loss, weighted_value_action_loss, weighted_class_prediction_loss

    def compute_classifications_CBT(self, predictions_classes, axis_classes=None, keepdim=False):

        if axis_classes is None:
            axis_classes = self.axis_features_outs

        classifications_CBT = [
            pt.max(predictions_classes[c], dim=axis_classes, keepdim=keepdim)[1] for c in range(0, self.C, 1)]

        return classifications_CBT

    def compute_correct_classifications_CBT(
            self, classifications: typing.Union[pt.Tensor, np.ndarray],
            labels: typing.Union[pt.Tensor, np.ndarray]):

        correct_classifications_CBT = [(classifications[c] == labels[c]).long() for c in range(0, self.C, 1)]

        return correct_classifications_CBT

    def compute_n_corrects_C(self, correct_classifications: typing.Union[pt.Tensor, np.ndarray]):

        n_corrects_C = [pt.sum(correct_classifications[c]).item() for c in range(0, self.C, 1)]

        return n_corrects_C

    def compute_n_corrects_CT(
            self, correct_classifications: typing.Union[pt.Tensor, np.ndarray],
            axis_time: typing.Union[int, tuple] = None, keepdim: bool = False):

        if axis_time is None:
            axis_time = self.axis_time_outs

        axes_non_time = tuple([a for a in range(0, correct_classifications[0].ndim, 1) if a != axis_time])

        n_corrects_CT = [
            pt.sum(correct_classifications[c], dim=axes_non_time, keepdim=keepdim) for c in range(0, self.C, 1)]

        return n_corrects_CT

    def compute_n_corrects_T(
            self, correct_classifications: typing.Union[pt.Tensor, np.ndarray],
            axis_time: typing.Union[int, tuple] = None, keepdim: bool = False):

        n_corrects_CT = self.compute_n_corrects_CT(correct_classifications, axis_time=axis_time, keepdim=keepdim)

        n_corrects_T = sum(n_corrects_CT)

        return n_corrects_T

    def compute_n_corrects(self, correct_classifications: typing.Union[pt.Tensor, np.ndarray]):

        n_corrects_C = self.compute_n_corrects_C(correct_classifications=correct_classifications)

        n_corrects = sum(n_corrects_C)

        return n_corrects

    def compute_n_classifications_C(self, classifications):

        n_classifications_C = [cp_prod(classifications[c].shape) for c in range(0, self.C, 1)]

        return n_classifications_C

    def compute_n_classifications_CT(
            self, classifications, axis_time: typing.Union[int, tuple] = None):

        if axis_time is None:
            axis_time = self.axis_time_outs

        axes_non_time = [a for a in range(0, classifications[0].ndim, 1) if a != axis_time]

        n_classifications_CT = [
            cp_prod(np.asarray(classifications[c].shape, dtype='i')[axes_non_time]) for c in range(0, self.C, 1)]

        return n_classifications_CT

    def compute_n_classifications_T(
            self, classifications, axis_time: typing.Union[int, tuple] = None):

        n_classifications_CT = self.compute_n_classifications_CT(classifications, axis_time=axis_time)

        n_classifications_C = sum(n_classifications_CT)

        return n_classifications_C

    def compute_n_classifications(self, classifications):

        n_classifications_C = self.compute_n_classifications_C(classifications=classifications)

        n_classifications = sum(n_classifications_C)

        return n_classifications

    def compute_n_selected_actions_A(self, selected_actions):

        n_selected_actions_A = [cp_prod(selected_actions[a].shape) for a in range(0, self.A, 1)]

        return n_selected_actions_A

    def compute_n_selected_actions(self, selected_actions):

        n_selected_actions_A = self.compute_n_selected_actions_A(selected_actions=selected_actions)

        n_selected_actions = sum(n_selected_actions_A)

        return n_selected_actions

    def get_last_predictions_classes(self, predictions_classes, axis_time=None, keepdim=False):

        if axis_time is None:
            axis_time = self.axis_time_outs

        last_predictions_classes = [predictions_classes[c][tuple([
            slice(0, predictions_classes[c].shape[a], 1)
            if a != axis_time else slice(predictions_classes[c].shape[a] - 1, predictions_classes[c].shape[a], 1)
            if keepdim else predictions_classes[c].shape[a] - 1
            for a in range(0, predictions_classes[c].ndim, 1)])]
                                for c in range(0, self.C, 1)]

        return last_predictions_classes

    def get_last_labels(self, labels, axis_time=None, keepdim=False):

        if axis_time is None:
            axis_time = self.axis_time_outs

        last_labels = [labels[c][tuple([
            slice(0, labels[c].shape[a], 1)
            if a != axis_time else slice(labels[c].shape[a] - 1, labels[c].shape[a], 1)
            if keepdim else labels[c].shape[a] - 1
            for a in range(0, labels[c].ndim, 1)])]
                       for c in range(0, self.C, 1)]

        return last_labels

    def get_last_time_point(self, outputs, axis_time=None, keepdim=False):

        if axis_time is None:
            axis_time = self.axis_time_outs

        n_outputs = len(outputs)

        last_outputs = [outputs[o][tuple([
            slice(0, outputs[o].shape[d], 1)
            if d != axis_time else slice(outputs[o].shape[d] - 1, outputs[o].shape[d], 1)
            if keepdim else outputs[o].shape[d] - 1
            for d in range(0, outputs[o].ndim, 1)])]
                        for o in range(0, n_outputs, 1)]

        return last_outputs

    def compute_losses_trials(self, losses):

        if self.move_axes_losses_trials:

            if isinstance(losses[0], pt.Tensor):
                losses_trials = pt.moveaxis(
                    input=losses,
                    source=self.source_axes_losses_trials,
                    destination=self.destination_axes_losses_trials).tolist()
            else:
                losses_trials = np.moveaxis(
                    a=losses,
                    source=self.source_axes_losses_trials,
                    destination=self.destination_axes_losses_trials)
        else:
            losses_trials = losses.tolist()

        return losses_trials

    def compute_outs_trials(self, outs):

        M = len(outs)

        if isinstance(outs[0], pt.Tensor):
            if self.move_axes_outs_trials:
                outs_trials = pt.cat([
                    pt.moveaxis(
                        input=outs[m],
                        source=self.source_axes_outs_trials,
                        destination=self.destination_axes_outs_trials) for m in range(0, M, 1)],
                    dim=self.axis_features_outs_trials).tolist()
            else:
                outs_trials = pt.cat(
                    outs, dim=self.axis_features_outs_trials).tolist()
        else:
            if self.move_axes_outs_trials:

                outs_trials = np.concatenate([
                    np.moveaxis(
                        a=outs[m],
                        source=self.source_axes_outs_trials,
                        destination=self.destination_axes_outs_trials) for m in range(0, M, 1)],
                    axis=self.axis_features_outs_trials).tolist()
            else:
                outs_trials = np.concatenate(
                    outs, axis=self.axis_features_outs_trials).tolist()

        return outs_trials
