

import torch as torch
from ..general import *
import numpy as np
import typing


__all__ = ['ClassifierMethods', 'TimedClassifierMethods']


class ClassifierMethods(OutputMethods):

    def __init__(
            self, axis_features_outs: int, axis_models_losses: int,  C: int,
            loss_scales_classifiers: (
                    typing.Union[None, int, float, list, tuple, np.ndarray, torch.Tensor]) = None) -> None:

        superclass = ClassifierMethods
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if isinstance(C, int):
            if C > 0:
                self.n_classifiers = self.C = C
            else:
                raise ValueError('M')
        else:
            raise TypeError('M')

        self.loss_scales_classifiers = _set_loss_scales(M=self.C, loss_scales=loss_scales_classifiers)

        if OutputMethods not in self.superclasses_initiated:
            OutputMethods.__init__(
                self=self, axis_features_outs=axis_features_outs,
                axis_models_losses=axis_models_losses, M=self.C, loss_scales=self.loss_scales_classifiers)
            if OutputMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(OutputMethods)

        self.criterion_predictions_classes = torch.nn.CrossEntropyLoss(reduction='none')
        self.criterion_predictions_classes_reduction = torch.nn.CrossEntropyLoss(reduction='mean')

        self.softmax = torch.nn.Softmax(dim=self.axis_features_outs)

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def compute_probabilities(self, predictions_classes):

        C = len(predictions_classes)
        probabilities = [self.softmax(predictions_classes[c]) for c in range(0, C, 1)]

        return probabilities

    def compute_class_prediction_losses(self, predictions_classes, labels):

        C = len(predictions_classes)
        shape_losses = self.compute_shape_losses(predictions_classes)

        device = predictions_classes[0].device
        class_prediction_losses = torch.empty(shape_losses, dtype=torch.float32, device=device, requires_grad=False)

        indexes_losses = [
            slice(0, class_prediction_losses.shape[a], 1)
            for a in range(0, class_prediction_losses.ndim, 1)]  # type: list

        for c in range(0, C, 1):
            indexes_losses[self.axis_models_losses] = c
            tuple_indexes_losses = tuple(indexes_losses)

            if self.axis_features_outs == 1:
                class_prediction_losses[tuple_indexes_losses] = self.criterion_predictions_classes(
                    predictions_classes[c], labels[tuple_indexes_losses])
            else:
                class_prediction_losses[tuple_indexes_losses] = self.criterion_predictions_classes(
                    torch.movedim(predictions_classes[c], self.axis_features_outs, 1), labels[tuple_indexes_losses])

        return class_prediction_losses

    def compute_classifications(self, predictions_classes):

        C = len(predictions_classes)
        shape_losses = self.compute_shape_losses(predictions_classes)

        device = predictions_classes[0].device

        classifications = torch.empty(shape_losses, dtype=torch.int64, device=device, requires_grad=False)
        indexes_classifications = [
            slice(0, classifications.shape[a], 1) for a in range(0, classifications.ndim, 1)]  # type: list

        for c in range(0, C, 1):

            indexes_classifications[self.axis_models_losses] = c

            classifications[tuple(indexes_classifications)] = torch.max(
                predictions_classes[c], dim=self.axis_features_outs, keepdim=False)[1]

        return classifications

    def compute_correct_classifications(
            self, classifications: typing.Union[torch.Tensor, np.ndarray],
            labels: typing.Union[torch.Tensor, np.ndarray]):

        correct_classifications = (classifications == labels).long()

        return correct_classifications

    def compute_n_corrects(
            self, correct_classifications: typing.Union[torch.Tensor, np.ndarray],
            axes_not_included: typing.Union[int, list, tuple, np.ndarray, torch.Tensor] = None, keepdim: bool = False):

        n_axes_classifications = correct_classifications.ndim

        if axes_not_included is None:
            axes_not = []
        elif isinstance(axes_not_included, int):
            axes_not = [axes_not_included + n_axes_classifications if axes_not_included < 0 else axes_not_included]
        elif isinstance(axes_not_included, (list, tuple)):
            axes_not = [a + n_axes_classifications if a < 0 else a for a in axes_not_included]
        elif isinstance(axes_not_included, (np.ndarray, torch.Tensor)):
            axes_not = [a + n_axes_classifications if a < 0 else a for a in axes_not_included.tolist()]
        else:
            raise TypeError('axes_not_included')

        n_axes_not = len(axes_not)

        if n_axes_not == 0:
            n_corrects = torch.sum(correct_classifications).item()
        else:
            axes_included = [a for a in range(0, correct_classifications.ndim, 1) if a not in axes_not]
            n_corrects = torch.sum(correct_classifications, dim=axes_included, keepdim=keepdim)

        return n_corrects

    def compute_n_classifications(
            self, classifications, axes_not_included: typing.Union[int, list, tuple, np.ndarray, torch.Tensor] = None):

        n_classifications = self.compute_n_losses(losses=classifications, axes_not_included=axes_not_included)

        return n_classifications

    def reduce_class_prediction_losses(
            self, class_prediction_losses: typing.Union[torch.Tensor, np.ndarray],
            axes_not_included: typing.Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
            scaled: bool = False,
            loss_scales_classifiers: typing.Union[list, tuple, np.ndarray, torch.Tensor] = None,
            format_scales: bool = True):

        if scaled and (loss_scales_classifiers is None):
            loss_scales_classifiers = self.loss_scales_classifiers
            format_scales = False

        reduced_class_prediction_losses = self.reduce_losses(
            losses=class_prediction_losses, axes_not_included=axes_not_included,
            scaled=scaled, loss_scales=loss_scales_classifiers, format_scales=format_scales)

        return reduced_class_prediction_losses


class TimedClassifierMethods(TimedOutputMethods, ClassifierMethods):

    def __init__(
            self, axis_time_outs, axis_batch_outs: int, axis_features_outs: int, axis_models_losses: int, C: int,
            loss_scales_classifiers: typing.Union[None, int, float, list, tuple, np.ndarray, torch.Tensor] = None) -> None:

        superclass = TimedClassifierMethods
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        if ClassifierMethods not in self.superclasses_initiated:
            ClassifierMethods.__init__(
                self, axis_features_outs=axis_features_outs,
                axis_models_losses=axis_models_losses, C=C, loss_scales_classifiers=loss_scales_classifiers)
            if ClassifierMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(ClassifierMethods)

        if TimedOutputMethods not in self.superclasses_initiated:
            TimedOutputMethods.__init__(
                self=self, axis_time_outs=axis_time_outs, axis_batch_outs=axis_batch_outs,
                axis_features_outs=axis_features_outs, axis_models_losses=axis_models_losses,
                M=self.C,  loss_scales=self.loss_scales_classifiers)
            if TimedOutputMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(TimedOutputMethods)

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)
