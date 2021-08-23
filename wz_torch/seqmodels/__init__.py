from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, Sequential


class ConvEmbedding(Module):
    """ Temporal conv embedding block w/one hidden layer. """
    def __init__(self, inp_dim: int, embed_dim: int) -> None:
        """ Constructor.

        :param inp_dim:   input dimension
        :type  inp_dim:   int
        :param embed_dim: desired embedding dimension
        :type  embed_dim: int
        """
        super(ConvEmbedding, self).__init__()
        self.embed = Sequential(
            nn.Conv1d(in_channels=inp_dim,
                      out_channels=embed_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=embed_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1))

    def forward(self, seq: Tensor) -> Tensor:
        """ Compute sequence embedding.

        :param seq: input sequences (T, B, D)
        :type  seq: Tensor
        :return:    embedded sequences (T, B, E)
        :rtype:     Tensor
        """
        seq = seq.transpose(0, 1).transpose(1, 2)   # (B, D, T)
        emb = self.embed(seq)                       # (B, E, T)
        return emb.transpose(1, 2).transpose(0, 1)  # (T, B, E)


class ConvEmbedding2(Module):
    """ Temporal conv embedding block w/one hidden layer. """
    def __init__(self, inp_dim: int, embed_dim: int) -> None:
        """ Constructor.

        :param inp_dim:   input dimension
        :type  inp_dim:   int
        :param embed_dim: desired embedding dimension
        :type  embed_dim: int
        """
        super(ConvEmbedding2, self).__init__()
        self.embed = Sequential(
            nn.Conv1d(in_channels=inp_dim,
                      out_channels=embed_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=embed_dim,
                      kernel_size=3,
                      stride=2,
                      padding=1))

    def forward(self, seq: Tensor) -> Tensor:
        """ Compute sequence embedding.

        :param seq: input sequences (T, B, D)
        :type  seq: Tensor
        :return:    embedded sequences (T, B, E)
        :rtype:     Tensor
        """
        seq = seq.transpose(0, 1).transpose(1, 2)   # (B, D, T)
        emb = self.embed(seq)                       # (B, E, T)
        return emb.transpose(1, 2).transpose(0, 1)  # (T, B, E)


class TransformerEncoderLayer2(nn.Module):
    """ Redefine transformer encoder block based on follow-on paper:
        https://arxiv.org/pdf/2002.04745.pdf, Figure 1. """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: Optional[int] = 2048,
                 dropout: Optional[float] = 0.1) -> None:
        """ Constructor.

        :param d_model:         expected input embedding dimension
        :type  d_model:         int
        :param nhead:           num of multiattention heads
        :type  nhead:           int
        :param dim_feedforward: FF network hidden layer size, defaults to 2048
        :type  dim_feedforward: int, optional
        :param dropout:         prob of 0-ing node activation, defaults to 0.1
        :type  dropout:         float, optional
        """
        super(TransformerEncoderLayer2, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)  # block 1
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout))  # block 2

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """ Pass input through the encoder layer.

        :param src:                  sequence to the encoder layer
        :type  src:                  Tensor
        :param src_mask:             mask for src sequence, defaults to None
        :type  src_mask:             Optional[Tensor], optional
        :param src_key_padding_mask: src keys mask per batch, defaults to None
        :type  src_key_padding_mask: Optional[Tensor], optional
        :return:                     encoder layer encoded output
        :rtype:                      Tensor
        """
        src2 = self.norm(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = src + self.ffn(src)
        return src


class ConvUpdecoding(Module):
    """ Temporal conv upsampling block w/one hidden layer. """
    def __init__(self, inp_dim: int, out_dim: int) -> None:
        """ Constructor.

        :param inp_dim: input dimension
        :type  inp_dim: int
        :param out_dim: desired output dimension
        :type  out_dim: int
        """
        super(ConvEmbedding2, self).__init__()
        self.decode = Sequential(
            nn.ConvTranspose1d(in_channels=inp_dim,
                               out_channels=inp_dim,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False),
            nn.BatchNorm1d(inp_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_dim,
                      out_channels=out_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1))

    def forward(self, seq: Tensor) -> Tensor:
        """ Compute sequence embedding.

        :param seq: input sequences (T, B, D)
        :type  seq: Tensor
        :return:    embedded sequences (T, B, E)
        :rtype:     Tensor
        """
        seq = seq.transpose(0, 1).transpose(1, 2)   # (B, D, T)
        dec = self.decode(seq)                      # (B, E, T)
        return dec.transpose(1, 2).transpose(0, 1)  # (T, B, E)
