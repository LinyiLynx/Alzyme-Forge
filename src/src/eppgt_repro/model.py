from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


def sequence_mask(x: torch.Tensor, valid_lengths: torch.Tensor, value: float | None = None) -> torch.Tensor:
    max_length = x.size(1)
    mask = torch.arange(max_length, dtype=torch.float32, device=x.device)[None, :] < valid_lengths[:, None]
    x = x.clone()
    if value is None:
        value = torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e10
    x[~mask] = value
    return x


def masked_softmax(x: torch.Tensor, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
    if valid_lengths is None:
        return F.softmax(x, dim=-1)
    shape = x.shape
    if valid_lengths.dim() == 1:
        valid_lengths = torch.repeat_interleave(valid_lengths, shape[1])
    else:
        valid_lengths = valid_lengths.reshape(-1)
    masked = sequence_mask(x.reshape(-1, shape[-1]), valid_lengths)
    return F.softmax(masked.reshape(shape), dim=-1)


def transpose_qkv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class AddNorm(nn.Module):
    def __init__(self, norm_shape: int, drop_rate: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(norm_shape)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.dropout(y) + x)


class DotProductAttention(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.attention_weights = None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dim = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(dim)
        self.attention_weights = masked_softmax(scores, valid_lengths)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
        num_hiddens: int,
        num_heads: int,
        drop_rate: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(drop_rate)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        if valid_lengths is not None:
            valid_lengths = torch.repeat_interleave(valid_lengths, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lengths)
        return self.w_o(transpose_output(output, self.num_heads))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        norm_shape: int,
        drop_rate: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, drop_rate)
        self.addnorm1 = AddNorm(norm_shape, drop_rate)
        self.ffn = FFN(hidden_dim, hidden_dim, hidden_dim, drop_rate=drop_rate)
        self.addnorm2 = AddNorm(norm_shape, drop_rate)

    def forward(self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
        y = self.addnorm1(x, self.attention(x, x, x, valid_lengths))
        return self.addnorm2(y, self.ffn(y))


class Encoder(nn.Module):
    def __init__(self, protein_dim: int, hidden_dim: int, norm_shape: int, drop_rate: float, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.fc = nn.Linear(protein_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [EncoderBlock(hidden_dim, norm_shape, drop_rate, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.fc(x)
        for block in self.blocks:
            x = block(x, valid_lengths)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, norm_shape: int, drop_rate: float, num_heads: int, index: int) -> None:
        super().__init__()
        self.index = index
        self.attention1 = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, drop_rate)
        self.attention2 = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, drop_rate)
        self.addnorm1 = AddNorm(norm_shape, drop_rate)
        self.addnorm2 = AddNorm(norm_shape, drop_rate)
        self.ffn = FFN(hidden_dim, hidden_dim, hidden_dim, drop_rate=drop_rate)
        self.addnorm3 = AddNorm(norm_shape, drop_rate)

    def forward(self, x: torch.Tensor, state, valid_lengths: torch.Tensor | None):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.index] is None:
            key_values = x
        else:
            key_values = torch.cat((state[2][self.index], x), dim=1)
        state[2][self.index] = key_values
        y = self.addnorm1(x, self.attention1(x, key_values, key_values, valid_lengths))
        z = self.addnorm2(y, self.attention2(y, enc_outputs, enc_outputs, enc_valid_lens))
        return self.addnorm3(z, self.ffn(z)), state


class Decoder(nn.Module):
    def __init__(self, atom_dim: int, hidden_dim: int, norm_shape: int, drop_rate: float, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.fc = nn.Linear(atom_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(hidden_dim, norm_shape, drop_rate, num_heads, index=i) for i in range(num_layers)]
        )
        self.drop_final = nn.Dropout(0.2)
        self.dense1 = nn.Linear(hidden_dim, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 2)
        self.weight_1 = nn.Parameter(torch.empty(atom_dim, atom_dim))
        self.weight_2 = nn.Parameter(torch.empty(atom_dim, atom_dim))
        nn.init.xavier_uniform_(self.weight_1)
        nn.init.xavier_uniform_(self.weight_2)

    def init_state(self, enc_outputs: torch.Tensor, enc_valid_lens: torch.Tensor | None = None):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def gcn(self, compound: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(compound, self.weight_1)
        output = torch.bmm(adj, support)
        support = torch.matmul(output, self.weight_2)
        output = torch.bmm(adj, support)
        return output

    def forward(self, x: torch.Tensor, adjs: torch.Tensor, state, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.gcn(x, adjs)
        x = self.fc(x)
        for block in self.blocks:
            x, state = block(x, state, valid_lengths)
        norm = F.softmax(sequence_mask(torch.norm(x, dim=2), valid_lengths), dim=1)
        pooled = torch.sum(x * norm[:, :, None], dim=1)
        pooled = self.drop_final(F.leaky_relu(self.dense1(pooled)))
        pooled = self.drop_final(F.leaky_relu(self.dense2(pooled)))
        return self.dense3(pooled)


class ModelCat(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        protein: torch.Tensor,
        protein_lengths: torch.Tensor,
        compounds: torch.Tensor,
        adjs: torch.Tensor,
        compound_lengths: torch.Tensor,
    ) -> torch.Tensor:
        enc_outputs = self.encoder(protein, protein_lengths)
        state = self.decoder.init_state(enc_outputs, protein_lengths)
        return self.decoder(compounds, adjs, state, compound_lengths)


def build_model(config: ModelConfig) -> ModelCat:
    encoder = Encoder(
        protein_dim=config.protein_dim,
        hidden_dim=config.hidden_dim,
        norm_shape=config.hidden_dim,
        drop_rate=config.dropout,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    decoder = Decoder(
        atom_dim=config.atom_dim,
        hidden_dim=config.hidden_dim,
        norm_shape=config.hidden_dim,
        drop_rate=config.dropout,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    return ModelCat(encoder, decoder)
