#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/28 12:25
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   position_encoders.py
# @Desc     :   

from torch import (Tensor, zeros, device, float32, sin, cos,
                   arange, exp, log, tensor,
                   randn, abs)
from typing import override

from src.nets.base_encoder4positions import BasePositionEncoder
from src.utils.helper import Timer


class ArtificialPositionEncoder(BasePositionEncoder):
    """ Base Class for Position Encoders """

    def __init__(self, max_len: int, dim_model: int) -> None:
        """ Initialize Position Encoder
        :param max_len: Maximum length of the input sequence
        :param dim_model: Dimension of the model embeddings
        """
        super().__init__(
            max_len=max_len,
            dim_model=dim_model
        )

    @override
    def _encode_positions(self, accelerator: device):
        """ Encode Positions for the Input Sequence
        :param accelerator: Device to place the position encodings on
        :return: Position Encodings Tensor of shape (1, max_len, dim_model)
        """
        # Initialize Positional Encoding Matrix
        self._positions: Tensor = zeros((self._max_len, self._dim_model), dtype=float32, device=accelerator)
        for position in range(self._max_len):
            for i in range(0, self._dim_model, 2):
                self._positions[position, i] = float(position) / (10_000 ** (float(i) / self._dim_model))
                if i + 1 < self._dim_model:
                    self._positions[position, i + 1] = float(position) / (10_000 ** (float(i) / self._dim_model))
        self._positions[:, 0::2] = sin(self._positions[:, 0::2])
        self._positions[:, 1::2] = cos(self._positions[:, 1::2])

    @override
    def forward(self, X: Tensor) -> Tensor:
        """ Forward Pass to Add Position Encodings to Input
        :param X: Input Tensor of shape (batch_size, seq_len, dim_model)
        :return: Tensor with Position Encodings added, shape (batch_size, seq_len, dim_model)
        """
        if self._positions is None:
            raise RuntimeError("Position encoding not initialized. Call set_device() first.")

        seq_len = X.shape[1]
        assert seq_len <= self._max_len, "Sequence length exceeds maximum length."

        return X + self._positions[:seq_len, :].unsqueeze(0)


class TransformerPositionEncoder(BasePositionEncoder):
    """ Transformer Position Encoder Class """

    def __init__(self, max_len: int, dim_model: int) -> None:
        """ Initialize Transformer Position Encoder
        :param max_len: Maximum length of the input sequence
        :param dim_model: Dimension of the model embeddings
        """
        super().__init__(
            max_len=max_len,
            dim_model=dim_model
        )
        self._scale: float | None = None

    def set_scale(self, scale: float = 1.0):
        """ Set Scale for the Position Encoder
        :param scale: Scale factor to apply to the position encodings
        """
        self._scale = scale

        return self

    @override
    def _encode_positions(self, accelerator: device):
        """ Encode Positions for the Input Sequence
        :param accelerator: Device to place the position encodings on
        :return: Position Encodings Tensor of shape (1, max_len, dim_model)
        """
        # Initialize Positional Encoding Matrix, which is from Torch Website
        positions = arange(self._max_len, dtype=float32, device=accelerator).unsqueeze(1)
        div_term = exp(
            arange(0, self._dim_model, 2, dtype=float32, device=accelerator) * (-log(tensor(10000.0)) / self._dim_model)
        )

        self._positions = zeros(self._max_len, self._dim_model, dtype=float32, device=accelerator)
        self._positions[:, 0::2] = sin(positions * div_term)
        self._positions[:, 1::2] = cos(positions * div_term)

    @override
    def forward(self, X: Tensor) -> Tensor:
        """ Forward Pass to Add Position Encodings to Input
        :param X: Input Tensor of shape (batch_size, seq_len, dim_model)
        :return: Tensor with Position Encodings added, shape (batch_size, seq_len, dim_model)
        """
        if self._positions is None:
            raise RuntimeError("Position encoding not initialized. Call set_device() first.")

        seq_len = X.shape[1]
        assert seq_len <= self._max_len, "Sequence length exceeds maximum length."

        return X + self._positions[:seq_len, :].unsqueeze(0) * self._scale

    @property
    def scale(self) -> float | None:
        """ Get the Scale of the Position Encoder
        :return: Scale factor
        """
        return self._scale


if __name__ == "__main__":
    max_len: int = 100
    embedding_dims: int = 8
    accelerator: str = "cpu"
    X = randn(2, 5, 8)

    with Timer("Artificial Position Encoders"):
        ape = (ArtificialPositionEncoder(max_len=max_len, dim_model=embedding_dims)
               .set_device(accelerator=accelerator))
        out_ape = ape(X)
        print("Output Shape (APE):", out_ape.shape)

    with Timer("Transformer Position Encoders"):
        tf = TransformerPositionEncoder(max_len=max_len, dim_model=embedding_dims)
        tf.set_device(accelerator=accelerator)
        tf.set_scale(1.0)
        out_default = tf(X)
        print("Output Shape (Transformer):", out_default.shape)

        tf.set_scale(2.0)
        out_scaled = tf(X)
        print("Output Shape (Transformer Scaled):", out_scaled.shape)

        print("Input X:", X[0, 0, :])
        print("Scale=1.0 out:", out_default[0, 0, :])
        print("Scale=2.0 out:", out_scaled[0, 0, :])
        print("Diffs:", abs(out_default - out_scaled).mean())
