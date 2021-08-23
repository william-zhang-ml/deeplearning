from typing import Optional
import torch
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
            nn.Dropout(dropout))            # block 2

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
        src = src + self.dropout(src2)  # block 1
        src = src + self.ffn(src)       # block 2
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
        super(ConvUpdecoding, self).__init__()
        self.decode = Sequential(
            nn.ConvTranspose1d(in_channels=inp_dim,
                               out_channels=inp_dim,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False),
            nn.BatchNorm1d(inp_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=inp_dim,
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


class PositionalEncoderA(Module):
    """ Alternative positional encoding for transformers. """
    def __init__(self, length: int, d_model: int, dropout: float) -> None:
        """ Constructor.

        :param length:  maximum supported sequence length
        :type  length:  int
        :param d_model: expected input dimension
        :type  d_model: int
        :param dropout: prob of nulling an input activation
        :type  dropout: float, optional
        """
        super(PositionalEncoderA, self).__init__()
        row = torch.arange(length).unsqueeze(-1) / length  # [0, 1) time
        col = torch.linspace(5, 1, d_model).unsqueeze(0)   # Hz
        phase = 6.283185307179586 * row * col              # const = 2 * pi
        pe = phase.sin().unsqueeze(1)
        self.register_buffer('pe', pe)                     # torch state var
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor: Tensor) -> Tensor:
        """ Add positional encoding and apply dropout.

        :param tensor: batch of sequences (S, B, D)
        :type  tensor: Tensor
        :return:       batch w/pos enc and dropout applied
        :rtype:        Tensor
        """
        tensor = tensor + self.pe[:tensor.size(0)]
        return self.dropout(tensor)


class SimpleTransAutoenc(Module):
    """ Transformer encoder arch based on arxiv.org/pdf/1706.03762.pdf. """
    def __init__(self,
                 inp_dim: int,
                 n_layers: int,
                 length: int,
                 d_model: int,
                 n_head: int,
                 d_feedforward: int,
                 dropout: float = 0.1) -> None:
        """ General transformer arch for pre-training and later fine-tuning.

        :param inp_dim:       input dimension
        :type  inp_dim:       int
        :param n_layers:      num of stacked encoders
        :type  n_layers:      int
        :param length:        maximum supported sequence length
        :type  length:        int
        :param d_model:       embedding dimension
        :type  d_model:       int
        :param n_head:        num of multi-attention heads
        :type  n_head:        int
        :param d_feedforward: FF network hidden layer size, defaults to 2048
        :type  d_feedforward: int, optional
        :param dropout:       prob of nulling a transformer activations
        :type  dropout:       float, optional
        """
        super(SimpleTransAutoenc, self).__init__()
        self.pos_encoder = PositionalEncoderA(length, d_model, dropout)
        enc_lay = TransformerEncoderLayer2(
            d_model,
            n_head,
            d_feedforward,
            dropout)
        self.encoder = nn.TransformerEncoder(enc_lay, n_layers)
        self.embed = ConvEmbedding2(inp_dim, d_model)
        self.recon = ConvUpdecoding(d_model, inp_dim)

    def forward(self,
                seq: Tensor,
                attn_mask: Tensor = None,
                pad_mask: Tensor = None) -> Tensor:
        """ Encode batch of sequence tensors and reconstruct.

        :param seq:       batch of sequences (T, B, D)
        :type  seq:       Tensor
        :param attn_mask: mask that indicates what positions to ignore (T, T)
        :type  attn_mask: Tensor
        :param pad_mask:  mask that indicates what positions are padded (B, T)
        :type  pad_mask:  Tensor
        :return:          batch of reconstructed sequences (T, B, D)
        :rtype:           Tensor
        """
        emb = self.embed(seq)
        feats = self.pos_encoder(emb)
        feats = self.encoder(feats, attn_mask, pad_mask)
        return self.recon(feats)
