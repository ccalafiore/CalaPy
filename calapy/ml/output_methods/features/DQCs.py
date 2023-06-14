

from ... import torch
from .classifiers import *
from .DQNs import *
from .general import *
import numpy as np
import typing

__all__ = ['DQCMethods']


class DQCMethods(DQNMethods, TimedClassifierMethods):

    def __init__(
            self, tasks: typing.Union[str, list, tuple, np.ndarray],
            possible_actions: [list, tuple],
            axis_batch_outs: int, axis_features_outs: int, axis_models_losses: int,
            movement_type: str = 'proactive',
            same_actions: typing.Union[int, list, tuple, np.ndarray, torch.Tensor, None] = None,
            gamma: typing.Union[int, float] = .999, reward_bias: typing.Union[int, float] = .0,
            loss_weights: typing.Union[int, float, list, tuple, np.ndarray, torch.Tensor, None] = None) -> None:

        name_superclass = DQCMethods.__name__
        name_subclass = type(self).__name__
        if name_superclass == name_subclass:
            self.superclasses_initiated = []

        self.possible_tasks = ['A', 'C']
        self.n_possible_tasks = len(self.possible_tasks)

        if isinstance(tasks, str):
            self.tasks = [tasks.upper()]  # type: list
        elif isinstance(tasks, list):
            self.tasks = tasks
        elif isinstance(tasks, tuple):
            self.tasks = list(tasks)
        elif isinstance(tasks, np.ndarray):
            self.tasks = tasks.tolist()
        else:
            raise TypeError('tasks')

        self.n_tasks = len(self.tasks)
        self.indexes_outs_actors = []
        self.indexes_outs_classifiers = []
        for i in range(0, self.n_tasks, 1):
            if isinstance(self.tasks[i], str):
                self.tasks[i] = self.tasks[i].upper()
                if self.tasks[i] == 'A':
                    self.indexes_outs_actors.append(i)
                elif self.tasks[i] == 'C':
                    self.indexes_outs_classifiers.append(i)
                else:
                    raise ValueError('tasks[' + str(i) + ']')
            else:
                raise TypeError('tasks[' + str(i) + ']')

        A = len(self.indexes_outs_actors)
        C = len(self.indexes_outs_classifiers)
        M = A + C

        if TimedOutputMethods.__name__ not in self.superclasses_initiated:

            TimedOutputMethods.__init__(
                self=self, axis_batch_outs=axis_batch_outs, axis_features_outs=axis_features_outs,
                axis_models_losses=axis_models_losses, M=M, loss_weights=loss_weights)

        if DQNMethods.__name__ not in self.superclasses_initiated:

            loss_weights_actors = [self.loss_weights[self.indexes_outs_actors[a]] for a in range(0, A, 1)]

            DQNMethods.__init__(
                self=self, possible_actions=possible_actions, axis_batch_outs=axis_batch_outs,
                axis_features_outs=axis_features_outs, axis_models_losses=axis_models_losses,
                movement_type=movement_type, same_actions=same_actions, gamma=gamma, reward_bias=reward_bias,
                loss_weights_actors=loss_weights_actors)

        if TimedClassifierMethods.__name__ not in self.superclasses_initiated:
            loss_weights_classifiers = [
                self.loss_weights[self.indexes_outs_classifiers[c]] for c in range(0, C, 1)]
            TimedClassifierMethods.__init__(
                self=self, axis_batch_outs=axis_batch_outs, axis_features_outs=axis_features_outs,
                axis_models_losses=axis_models_losses, C=C, loss_weights_classifiers=loss_weights_classifiers)

        self.loss_weights_tasks = set_loss_weights(
            M=self.n_possible_tasks, loss_weights=[sum(self.loss_weights_actors), sum(self.loss_weights_classifiers)])

    def split(self, outs: typing.Union[list, tuple]):

        values_actions = [outs[self.indexes_outs_actors[a]] for a in range(0, self.A, 1)]

        predictions_classes = [outs[self.indexes_outs_classifiers[c]] for c in range(0, self.C, 1)]

        return values_actions, predictions_classes

    def get_previous_rewards(self, class_prediction_losses: torch.Tensor):

        axes = [a for a in range(0, class_prediction_losses.ndim, 1) if a != self.axis_models_losses]
        weighted_class_prediction_losses = self.reduce_class_prediction_losses(
            class_prediction_losses=class_prediction_losses, axes_not_included=axes,
            weighted=True, loss_weights_classifiers=None, format_weights=False)

        previous_rewards = - weighted_class_prediction_losses

        return previous_rewards

    def compute_multitask_losses(
            self, value_action_loss: torch.Tensor, class_prediction_loss: torch.Tensor, weighted: bool = False):

        if weighted:
            weights_tasks = self.loss_weights_tasks
        else:
            weights_tasks = [(1.0 / self.n_possible_tasks) for w in range(0, self.n_possible_tasks, 1)]

        multitask_losses = (
                (value_action_loss * weights_tasks[0]) + (class_prediction_loss * weights_tasks[1]))

        return multitask_losses
