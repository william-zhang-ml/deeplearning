from torch import Tensor
import torch.nn as nn


class WeightedBceLoss:
    """ Binary cross entropy w/class weights. """
    def __init__(self, pos_wt: float = 1, neg_wt: float = 1) -> None:
        """ Constructor,

        :param pos_wt: weight applied to positive sample pred, defaults to 1
        :type  pos_wt: float, optional
        :param neg_wt: weight applied to negative sample pred, defaults to 1
        :type  neg_wt: float, optional
        """
        self.loss = nn.BCELoss(reduction=None)
        self.pos_wt = pos_wt
        self.neg_wt = neg_wt

    def __call__(self, pred: Tensor, true: Tensor, mask=None) -> Tensor:
        """ Compute mean weighted-binary-cross-entropy.

        :param pred: prediction
        :type  pred: Tensor
        :param true: binary targets
        :type  true: Tensor
        :param mask: binary mask of loss-contributing indices, defaults to None
        :type  mask: Tensor, optional
        :return:     mean weighted-binary-cross-entropy (scalar)
        :rtype:      Tensor
        """
        losses = self.loss(pred, true)
        weight = (self.pos_wt - self.neg_wt) * true + self.neg_wt
        losses = weight * losses
        if mask is not None:
            losses = mask * losses
        return losses.mean()


class WeightedErrorLoss:
    """ Mean square error w/preference toward over/underestimating. """
    def __init__(self,
                 pos_wt: float = 1,
                 neg_wt: float = 1,
                 power: float = 2) -> None:
        """ Constructor,

        :param pos_wt: weight applied to positive error, defaults to 1
        :type  pos_wt: float, optional
        :param neg_wt: weight applied to negative error, defaults to 1
        :type  neg_wt: float, optional
        :param power:  power applied to error (1: MAE, 2: MSE), defaults to 2
        :type  power   float, optional
        """
        self.pos_wt = pos_wt
        self.neg_wt = neg_wt
        self.power = power

    def __call__(self, pred: Tensor, true: Tensor, mask=None) -> Tensor:
        """ Compute mean weighted-power-error.

        :param pred: prediction
        :type  pred: Tensor
        :param true: targets
        :type  true: Tensor
        :param mask: binary mask of loss-contributing indices, defaults to None
        :type  mask: Tensor, optional
        :return:     mean weighted-power-error (scalar)
        :rtype:      Tensor
        """
        error = (pred - true)
        sign = error >= 0
        weight = (self.pos_wt - self.neg_wt) * sign + self.neg_wt
        losses = error.abs().pow(self.power)
        losses = weight * losses
        if mask is not None:
            losses = mask * losses
        return losses.mean()


class BinaryFocalLoss:
    """ Enveloped BCE that deals w/class asymmetry in binary classification.

        Focal Loss for Dense Object Detection
        https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, gamma: float = 1, reduction='mean') -> None:
        """ Constructor.

        :param gamma:     knob that focuses on underconfident predictions,
                          defaults to 1
        :type  gamma:     float
        :param reduction: how to aggregate loss across elements (mean/none),
                          defaults to 'mean'
        :type             str
        """
        self.gamma = gamma
        self.reduction = reduction
        if reduction not in ['mean', 'none']:
            raise ValueError('reduction must be "mean" or "none".')

    def __call__(self, pred: Tensor, true: Tensor) -> Tensor:
        """ Compute focal loss.

        :param pred: prediction
        :type  pred: Tensor
        :param true: targets
        :type  true: Tensor
        :return:     focal loss
        :rtype:      Tensor
        """
        pos_samp_loss = true * (pred + 1e-9).log()  # vanilla BCE
        pos_samp_loss = pos_samp_loss * (1 - pred).pow(self.gamma)
        neg_samp_loss = (1 - true) * (1 - pred + 1e-9).log()  # vanilla BCE
        neg_samp_loss = neg_samp_loss * pred.pow(self.gamma)
        loss = pos_samp_loss + neg_samp_loss
        if self.reduction == 'mean':
            loss = -loss.mean()
        else:
            loss = -loss
        return loss

    def __repr__(self) -> str:
        """ Representation.

        :return: instance representation
        :rtype:  str
        """
        return (
            'BinaryFocalLoss('
            f'gamma={self.gamma}, '
            f'reduction="{self.reduction}")'
        )
