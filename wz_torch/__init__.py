from typing import Iterable, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, Sequential


def count_params(module: Module) -> int:
    """ Count the number of trainable parameters.

    :param module: PyTorch module
    :type  module: Module
    :return:       number of trainable parameters
    :rtype:        int
    """
    return sum([p.nelement() for p in module.parameters()])


class OneHiddenEmbedding(Module):
    """ Pointwise embedding block w/one hidden layer. """
    def __init__(self, inp_dim: int, embed_dim: int) -> None:
        """ Constructor.

        :param inp_dim:   input dimension
        :type  inp_dim:   int
        :param embed_dim: desired embedding dimension
        :type  embed_dim: int
        """
        super(OneHiddenEmbedding, self).__init__()
        self.embed = Sequential(
            nn.Linear(inp_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim))

    def forward(self, *args) -> Tensor:
        """ Compute sequence embedding.

        :return:    embedded sequences (T, B, E)
        :rtype:     Tensor
        """
        seq = torch.cat(args, dim=-1)
        return self.embed(seq)


class PredictOnAverage(Module):
    """ Multiclass logistic regression on sequence average representation. """
    def __init__(self, inp_dim: int, num_classes: int) -> None:
        """ Constructor.

        :param inp_dim:     input dimension
        :type  inp_dim:     int
        :param num_classes: number of target classes
        :type  num_classes: int
        """
        super(PredictOnAverage, self).__init__()
        self.pred_head = nn.Sequential(
            nn.Linear(inp_dim, num_classes),
            nn.Softmax(dim=-1))

    def forward(self, seq: Tensor, mask: Tensor = None) -> Tensor:
        """ Compute class confidence scores.

        :param seq:  input sequences (T, B, D)
        :type  seq:  Tensor
        :param mask: marks where seq is not padded (T, B), defaults to None
        :type  mask: Tensor, optional
        :return:     class confidence scores (B, num_classes)
        :rtype:      Tensor
        """
        if mask is None:
            ave_repr = seq.mean(dim=0)            # B, D
        else:
            mask = mask.view(*mask.shape, 1)      # T, B, 1
            seq_lens = mask.sum(dim=0)            # B, 1
            sum_repr = (seq * mask).sum(dim=0)    # B, D
            ave_repr = sum_repr / seq_lens        # B, D
        return self.pred_head(ave_repr)           # B, num_classes


class ChannelMseLoss:
    """ Mean square error but allows asymmetric feature/channel weighting. """
    def __init__(self, lams: Iterable[float]):
        """ Constructor.

        :param lams: channel weights
        :type  lams: Iterable[float]
        """
        self.lams = torch.tensor(lams)

    def __call__(self, x: Tensor, y: Tensor, mask: Tensor = None) -> Tensor:
        """ Compute loss

        :param x:    target (T, B, D)
        :type  x:    Tensor
        :param y:    prediction (T, B, D)
        :type  y:    Tensor
        :param mask: marks where seq is not padded (T, B), defaults to None
        :type  mask: Tensor, optional
        :return:     MSE loss, weighted by channel
        :rtype:      Tensor
        """
        sq_err = (x - y) ** 2
        if mask is None:
            samp_ave = sq_err.mean(dim=0)          # B, D
        else:
            mask = mask.view(*mask.shape, 1)       # T, B, 1
            seq_lens = mask.sum(dim=0)             # B, 1
            samp_sum = (sq_err * mask).sum(dim=0)  # B, D zero out padding
            samp_ave = samp_sum / seq_lens         # B, D
        view_shape = [1] * samp_ave.ndim
        view_shape[-1] = len(self)
        lam_view = self.lams.reshape(view_shape)   # 1, D

        return (lam_view * samp_ave).mean()

    def __len__(self) -> int:
        """ Length.

        :return: Expected number of channels
        :rtype:  int
        """
        return len(self.lams)


class GaussDistrLayer(Module):
    """ Output isotropic Gaussian distribution parameters. """
    def __init__(self, inp_dim: int, gauss_dim: int):
        """ Constructor.

        :param inp_dim:   input dimension
        :type  inp_dim:   int
        :param gauss_dim: Gaussian dimension
        :type  gauss_dim: int
        """
        super(GaussDistrLayer, self).__init__()
        self.mean = nn.Sequential(
            nn.Linear(inp_dim, gauss_dim),
            nn.LayerNorm(gauss_dim),
            nn.ReLU(),
            nn.Linear(gauss_dim, gauss_dim))
        self.logvar = nn.Sequential(
            nn.Linear(inp_dim, gauss_dim),
            nn.LayerNorm(gauss_dim),
            nn.ReLU(),
            nn.Linear(gauss_dim, gauss_dim),
            nn.Softplus())

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        """ Compute distribution parameters from features (B, D).

        :return: mean, log variance (B, DG), (B, DG)
        :rtype:  Tuple[Tensor, Tensor]
        """
        inp = torch.cat(args, dim=-1)
        return self.mean(inp), self.logvar(inp)
