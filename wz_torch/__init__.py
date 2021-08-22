from torch import Tensor
import torch.nn as nn
from torch.nn import Module, Sequential


class OneHiddenEmbedding(Sequential):
    """ Pointwise embedding block w/one hidden layer. """
    def __init__(self, inp_dim: int, embed_dim: int) -> None:
        """ Constructor.

        :param inp_dim:   input dimension
        :type  inp_dim:   int
        :param embed_dim: desired embedding dimension
        :type  embed_dim: int
        """
        super(OneHiddenEmbedding, self).__init__()
        self.add_module('Lin0', nn.Linear(inp_dim, embed_dim, bias=False))
        self.add_module('Norm0', nn.LayerNorm(embed_dim))
        self.add_module('Act0', nn.ReLU())
        self.add_module('Lin1', nn.Linear(embed_dim, embed_dim))


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
