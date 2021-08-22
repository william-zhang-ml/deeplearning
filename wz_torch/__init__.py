import torch.nn as nn
from torch.nn import Sequential


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
