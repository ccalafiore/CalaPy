

import torch
from ..... import maths as cp_maths
import numpy as np
import typing

__all__ = ['OutputMethods', 'TimedOutputMethods', '_set_loss_scales']


def _set_loss_scales(M, loss_scales=None):
    """


    :param M:
    :type M: int
    :param loss_scales:
    :type loss_scales: list[float | int  | None] | tuple[float | int  | None] |
                       np.ndarray[float | int  | None] | torch.Tensor[float | int  | None] |
                       float | int  | None
    :rtype: list[float]

    """

    if isinstance(M, int):
        if M < 1:
            raise ValueError('M < 1')
    else:
        raise TypeError('M')

    if loss_scales is None:
        loss_scales = [1.0 for m in range(0, M, 1)]
    if isinstance(loss_scales, int):
        loss_scales = [float(loss_scales)]
    elif isinstance(loss_scales, float):
        loss_scales = [loss_scales]
    elif isinstance(loss_scales, list):
        loss_scales = loss_scales
    elif isinstance(loss_scales, tuple):
        loss_scales = list(loss_scales)
    elif isinstance(loss_scales, (np.ndarray, torch.Tensor)):
        loss_scales = loss_scales.tolist()
    else:
        raise TypeError('loss_scales')

    M_tmp = len(loss_scales)

    if M_tmp != M:
        if M_tmp == 1:
            loss_scales = [loss_scales[0] for m in range(0, M, 1)]
        else:
            raise ValueError('M and loss_scales ')

    for m in range(0, M, 1):
        if loss_scales[m] is None:
            loss_scales[m] = 1.0
        elif isinstance(loss_scales[m], (int, float)):
            if isinstance(loss_scales[m], int):
                loss_scales[m] = float(loss_scales[m])
            if loss_scales[m] < 0.0:
                raise ValueError('loss_scales[' + str(m) + ']')
        else:
            raise TypeError('loss_scales[' + str(m) + ']')

    return loss_scales


class OutputMethods:

    def __init__(
            self, axis_features_outs: int, axis_models_losses: int, M: int,
            loss_scales: typing.Union[None, int, float, list, tuple, np.ndarray, torch.Tensor] = None) -> None:

        superclass = OutputMethods
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        # if cp_ModelMethods not in self.superclasses_initiated:
        #     cp_ModelMethods.__init__(self=self)
        #     if cp_ModelMethods not in self.superclasses_initiated:
        #         self.superclasses_initiated.append(cp_ModelMethods)

        if isinstance(axis_features_outs, int):
            self.axis_features_outs = axis_features_outs
        else:
            raise TypeError('axis_models_losses')

        if axis_models_losses is None:
            self.axis_models_losses = 0
        elif isinstance(axis_models_losses, int):
            self.axis_models_losses = axis_models_losses
        else:
            raise TypeError('axis_models_losses')

        if isinstance(M, int):
            if M > 0:
                self.M = M
            else:
                raise ValueError('M')
        else:
            raise TypeError('M')

        self.loss_scales = _set_loss_scales(M=self.M, loss_scales=loss_scales)

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def reduce_losses(
            self, losses: typing.Union[torch.Tensor, np.ndarray],
            axes_not_included: typing.Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
            scaled: bool = False, loss_scales: typing.Union[list, tuple, np.ndarray, torch.Tensor] = None,
            format_scales: bool = True):

        n_axes_losses = losses.ndim
        if axes_not_included is None:
            axes_not = []
        elif isinstance(axes_not_included, int):
            axes_not = [axes_not_included + n_axes_losses if axes_not_included < 0 else axes_not_included]
        elif isinstance(axes_not_included, (list, tuple)):
            axes_not = [a + n_axes_losses if a < 0 else a for a in axes_not_included]
        elif isinstance(axes_not_included, (np.ndarray, torch.Tensor)):
            axes_not = [a + n_axes_losses if a < 0 else a for a in axes_not_included.tolist()]
        else:
            raise TypeError('axes_not_included')
        n_axes_not = len(axes_not)

        if scaled:

            M = losses.shape[self.axis_models_losses]
            if loss_scales is None:
                loss_scales = self.loss_scales
            elif format_scales:
                loss_scales = _set_loss_scales(M=M, loss_scales=loss_scales)

            if n_axes_not == 0:
                axes_reduced = [
                    a for a in range(0, losses.ndim, 1) if a != self.axis_models_losses]

                reduced_losses = torch.mean(losses, dim=axes_reduced, keepdim=False)
                for m in range(0, M, 1):
                    reduced_losses[m] *= loss_scales[m]
                reduced_losses = torch.sum(reduced_losses)

            elif self.axis_models_losses in axes_not:

                raise ValueError('axes_not_reduced')

            else:

                axes_reduced = [
                    a for a in range(0, losses.ndim, 1)
                    if (a not in axes_not) and (a != self.axis_models_losses)]
                if len(axes_reduced) > 0:
                    reduced_losses = torch.mean(losses, dim=axes_reduced, keepdim=False)

                    new_axis_models_losses = self.axis_models_losses
                    for a in axes_reduced:
                        if a < self.axis_models_losses:
                            new_axis_models_losses -= 1
                    indexes_losses = [
                        slice(0, reduced_losses.shape[a], 1)
                        for a in range(0, reduced_losses.ndim, 1)]  # type: list
                    for m in range(0, M, 1):
                        indexes_losses[new_axis_models_losses] = m
                        reduced_losses[tuple(indexes_losses)] *= loss_scales[m]
                    reduced_losses = torch.sum(
                        reduced_losses, dim=new_axis_models_losses, keepdim=False)

                else:
                    indexes_losses = [slice(0, losses.shape[a], 1) for a in range(0, losses.ndim, 1)]  # type: list
                    reduced_losses = losses + 0.0
                    for m in range(0, M, 1):
                        indexes_losses[self.axis_models_losses] = m

                        reduced_losses[tuple(indexes_losses)] *= loss_scales[m]

                    reduced_losses = torch.sum(
                        reduced_losses, dim=self.axis_models_losses, keepdim=False)
        else:
            if n_axes_not == 0:
                reduced_losses = torch.mean(losses)
            else:
                axes_reduced = [a for a in range(0, losses.ndim, 1) if a not in axes_not]
                if len(axes_reduced) > 0:
                    reduced_losses = torch.mean(losses, dim=axes_reduced, keepdim=False)
                else:
                    reduced_losses = losses + 0.0

        return reduced_losses

    def compute_n_losses(self, losses, axes_not_included=None):

        """

        :type losses: np.ndarray | torch.Tensor
        :type axes_not_included: int | list | tuple | np.ndarray | torch.Tensor

        :rtype:
        """

        n_axes_losses = losses.ndim
        if axes_not_included is None:
            axes_not = []
        elif isinstance(axes_not_included, int):
            axes_not = [axes_not_included + n_axes_losses if axes_not_included < 0 else axes_not_included]
        elif isinstance(axes_not_included, (list, tuple)):
            axes_not = [a + n_axes_losses if a < 0 else a for a in axes_not_included]
        elif isinstance(axes_not_included, (np.ndarray, torch.Tensor)):
            axes_not = [a + n_axes_losses if a < 0 else a for a in axes_not_included.tolist()]
        else:
            raise TypeError('axes_not_included')
        n_axes_not = len(axes_not)

        if n_axes_not == 0:
            n_losses = cp_maths.prod(losses.shape)
        else:
            axes_included = [a for a in range(0, losses.ndim, 1) if a not in axes_not]
            n_losses = cp_maths.prod(np.asarray(losses.shape, dtype='i')[axes_included])

        return n_losses

    def compute_shape_losses(self, predictions: typing.Union[list, tuple]):

        n_dims_outs = predictions[0].ndim
        axes_non_features_outs = np.asarray(
            [a for a in range(0, n_dims_outs, 1) if a != self.axis_features_outs], dtype='i')

        n_dims_losses = n_dims_outs
        axes_non_models_losses = np.asarray(
            [a for a in range(0, n_dims_losses, 1) if a != self.axis_models_losses], dtype='i')

        M = len(predictions)
        shape_losses = [None for a in range(0, n_dims_losses, 1)]  # type: list
        shape_losses[self.axis_models_losses] = M
        for a in range(0, len(axes_non_models_losses), 1):
            shape_losses[axes_non_models_losses[a]] = predictions[0].shape[axes_non_features_outs[a]]

        return shape_losses


class TimedOutputMethods(OutputMethods):

    def __init__(
            self, axis_time_outs, axis_batch_outs: int, axis_features_outs: int, axis_models_losses: int, M: int,
            loss_scales: typing.Union[None, int, float, list, tuple, np.ndarray, torch.Tensor] = None) -> None:

        superclass = TimedOutputMethods
        try:
            # noinspection PyUnresolvedReferences
            self.superclasses_initiated
        except AttributeError:
            self.superclasses_initiated = []
        except NameError:
            self.superclasses_initiated = []

        self.n_axes_outs = 3
        self.axes_outs = np.arange(0, self.n_axes_outs, 1, dtype='i')
        self.axis_batch_outs = axis_batch_outs
        if self.axis_batch_outs < 0:
            self.axis_batch_outs += self.n_axes_outs

        self.batch_first = self.axis_batch_outs == 0

        self.axis_features_outs = axis_features_outs
        if self.axis_features_outs < 0:
            self.axis_features_outs += self.n_axes_outs

        self.axis_time_outs = axis_time_outs
        if self.axis_time_outs is None:
            pass
        elif isinstance(self.axis_time_outs, int):
            for a in range(0, self.n_axes_outs, 1):
                if self.axis_time_outs in [self.axis_batch_outs, self.axis_features_outs]:
                    self.axis_time_outs += 1
                else:
                    break
        else:
            raise TypeError('axis_time_outs')

        self.axes_non_features_outs = np.asarray(
            [a for a in self.axes_outs if a != self.axis_features_outs], dtype='i')

        if axis_models_losses is None:
            self.axis_models_losses = 0
        else:
            self.axis_models_losses = axis_models_losses

        self.n_axes_losses = 3
        self.axes_losses = np.arange(0, self.n_axes_losses, 1, dtype='i')
        if self.axis_models_losses < 0:
            self.axis_models_losses += self.n_axes_losses

        self.axis_time_losses = self.axis_time_outs
        if isinstance(self.axis_time_losses, int):
            if self.axis_time_outs > self.axis_features_outs:
                self.axis_time_losses -= 1
            if self.axis_time_losses >= self.axis_models_losses:
                self.axis_time_losses += 1

        self.axis_batch_losses = self.axis_batch_outs
        if self.axis_batch_outs > self.axis_features_outs:
            self.axis_batch_losses -= 1
        if self.axis_batch_losses >= self.axis_models_losses:
            self.axis_batch_losses += 1
        self.axes_non_models_losses = np.asarray(
            [a for a in self.axes_losses if a != self.axis_models_losses], dtype='i')

        if OutputMethods not in self.superclasses_initiated:
            OutputMethods.__init__(
                self=self, axis_features_outs=self.axis_features_outs,
                axis_models_losses=self.axis_models_losses, M=M, loss_scales=loss_scales)
            if OutputMethods not in self.superclasses_initiated:
                self.superclasses_initiated.append(OutputMethods)

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

        # self.shape_losses = np.asarray(
        #     [self.M if a == self.axis_models_losses else -1 for a in range(0, self.n_axes_losses, 1)],
        #     dtype='i')
        # self.shape_losses = np.asarray([-1 for a in range(0, self.n_axes_losses, 1)], dtype='i')

        if superclass not in self.superclasses_initiated:
            self.superclasses_initiated.append(superclass)

    def compute_losses_trials(self, losses):

        if self.axis_time_losses is None:
            raise ValueError('self.axis_time_losses')
        else:
            if self.move_axes_losses_trials:

                if isinstance(losses, torch.Tensor):
                    losses_trials = torch.moveaxis(
                        input=losses,
                        source=self.source_axes_losses_trials,
                        destination=self.destination_axes_losses_trials).tolist()
                else:
                    losses_trials = np.moveaxis(
                        a=losses,
                        source=self.source_axes_losses_trials,
                        destination=self.destination_axes_losses_trials).tolist()
            else:
                losses_trials = losses.tolist()

            return losses_trials

    def compute_outs_trials(self, outs):

        if self.axis_time_outs is None:
            raise ValueError('self.axis_time_outs')
        else:
            M = len(outs)

            if isinstance(outs[0], torch.Tensor):
                if self.move_axes_outs_trials:
                    outs_trials = torch.cat([
                        torch.moveaxis(
                            input=outs[m],
                            source=self.source_axes_outs_trials,
                            destination=self.destination_axes_outs_trials) for m in range(0, M, 1)],
                        dim=self.axis_features_outs_trials).tolist()
                else:
                    outs_trials = torch.cat(
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
