#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/28 13:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base_encoder4positions.py
# @Desc     :   

from abc import ABC, abstractmethod
from torch import nn, Tensor, device
from typing import final, Literal


class BasePositionEncoder(nn.Module, ABC):
    """ Base Class for Position Encoders """

    def __init__(self, max_len: int, dim_model: int) -> None:
        """ Initialize Position Encoder
        :param max_len: Maximum length of the input sequence
        :param dim_model: Dimension of the model embeddings
        """
        super().__init__()
        # Initialize Parameters
        self._max_len: int = max_len
        self._dim_model: int = dim_model
        self._accelerator: device | None = None
        # Initialize Position Encodings Buffer
        self.register_buffer("_positions", None, persistent=False)

    @final
    def set_device(self, accelerator: str | Literal["cuda", "cpu"] = "cpu"):
        """ Set Accelerator for the Module
        :param accelerator: Accelerator type, either 'cpu' or 'cuda'
        """
        accelerator = accelerator.lower()

        if accelerator not in ["cuda", "cpu"]:
            raise ValueError("Accelerator must be either 'cpu' or 'cuda'")
        self._accelerator = device(accelerator)

        # Activate Position Encodings
        self._encode_positions(self._accelerator)

        return self

    @abstractmethod
    def _encode_positions(self, accelerator: device):
        """ Encode Positions for the Input Sequence
        :param accelerator: Device to place the position encodings on
        :return: Position Encodings Tensor of shape (1, max_len, dim_model)
        """
        # Initialize Positional Encoding Matrix
        pass

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """ Forward Pass to Add Position Encodings to Input
        :param X: Input Tensor of shape (batch_size, seq_len, dim_model)
        :return: Tensor with Position Encodings added, shape (batch_size, seq_len, dim_model)
        """
        pass

    @property
    def max_len(self) -> int:
        """ Get Maximum Length of the Input Sequence
        :return: Maximum Length of the Input Sequence
        """
        return self._max_len

    @property
    def dim_model(self) -> int:
        """ Get Dimension of the Model Embeddings
        :return: Dimension of the Model Embeddings
        """
        return self._dim_model


if __name__ == "__main__":
    pass
