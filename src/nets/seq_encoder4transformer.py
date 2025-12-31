#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/28 15:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_encoder4transformer.py
# @Desc     :   

from torch import (nn, device, Tensor,
                   tensor)
from typing import Literal, final

from src.nets.position_encoders import TransformerPositionEncoder

WIDTH: int = 64


class TransformerSeqEncoder(nn.Module):
    """ Pytorch Transformer Encoder Network """

    def __init__(self,
                 vocab_size: int, embedding_dims: int = 512,
                 *,
                 max_len: int = 100, scale: float = 1.0,
                 num_heads: int = 8, feedforward_dims: int = 2048, activation: str | Literal["relu", "gelu"] = "relu",
                 num_layers: int = 6,
                 dropout_rate: float = 0.3,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD: int = 0
                 ) -> None:
        """ Initialize the Transformer Encoder
        :param vocab_size: Size of the source Vocabulary
        :param embedding_dims: Dimension of the Embeddings
        :param max_len: Maximum Length of Input Sequences
        :param scale: Scaling Factor for Embeddings
        :param num_heads: Number of Attention Heads
        :param feedforward_dims: Dimension of Feedforward Networks
        :param activation: Activation Function
        :param num_layers: Number of Transformer Layers
        :param dropout_rate: Dropout Rate
        :param accelerator: Device to place the model on
        :param PAD: Padding Token Index
        """
        super().__init__()
        # Initialise embedding layer parameters
        self._vocab_size: int = vocab_size
        self._embedding_dims: int = embedding_dims
        self._PAD: int = PAD
        # Initialise positional encoding parameters
        self._max_len: int = max_len
        self._scale: float = scale
        # Initialise Transformer parameters
        self._heads: int = num_heads
        self._feedforward_dims: int = feedforward_dims
        self._activation: str = activation.lower()
        self._dropout_rate: float = dropout_rate
        self._accelerator: str = accelerator.lower()
        # Initialise number of layers
        self._layers: int = num_layers

        # Initialise embedding layer
        self._embedder = self._init_embeddings()
        # Initialise positional encodings
        self._positions = self._init_positions()
        # Initialise Transformer Encoder
        self._layer = self._init_encoder_layer()
        self._encoder = nn.TransformerEncoder(
            self._layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dims, eps=1e-5, bias=True, device=device(self._accelerator)),
        )

        # Initialise weights
        self._init_weights()

    @final
    def _init_embeddings(self) -> nn.Embedding:
        """ Initialise Embedding Layer
        :return: Embedding Layer
        """
        return nn.Embedding(
            num_embeddings=self._vocab_size,
            embedding_dim=self._embedding_dims,
            padding_idx=self._PAD,
            device=device(self._accelerator)
        )

    @final
    def _init_positions(self) -> TransformerPositionEncoder:
        """ Initialise Position Encoder
        :return: Position Encoder
        """
        return (TransformerPositionEncoder(max_len=self._max_len, dim_model=self._embedding_dims)
                .set_device(self._accelerator)
                .set_scale(self._scale))

    @final
    def _init_encoder_layer(self) -> nn.TransformerEncoderLayer:
        """ Initialise a Single Transformer Encoder Layer
        :return: Transformer Encoder Layer
        """
        return nn.TransformerEncoderLayer(
            d_model=self._embedding_dims,
            nhead=self._heads,
            dim_feedforward=self._feedforward_dims,
            dropout=self._dropout_rate,
            activation=self._activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            bias=True,
            device=device(self._accelerator)
        )

    @final
    def _init_weights(self) -> None:
        """ Initialise Weights of the Model """
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    @final
    def trap_padding_mask(self, src: Tensor) -> Tensor:
        """ Create Padding Mask for Input Sequences
        :param src: Input Tensor of shape (batch_size, seq_len)
        :return: Padding Mask Tensor of shape (batch_size, seq_len)
        """
        return src == self._PAD

    def forward(self, src: Tensor) -> Tensor:
        """ Forward Pass through the Transformer Encoder
        :param src: Input Tensor of shape (batch_size, seq_len)
        :return: Output Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Token Embedding [batch_size, seq_len, embedding_dim]
        embeddings = self._embedder(src)

        # Add Position Encoding
        embeddings_with_pos = self._positions(embeddings)

        # Create padding mask [batch_size, seq_len]
        memory_key_padding_mask = self.trap_padding_mask(src)

        # Transformer Encoding [seq_len, batch_size, embedding_dim]
        tf_last_hn: Tensor = self._encoder(src=embeddings_with_pos, src_key_padding_mask=memory_key_padding_mask)

        return tf_last_hn

    @final
    def _count_parameters(self) -> tuple[int, int]:
        """ Count total and trainable parameters
        :return: total and trainable parameters
        """
        total_params: int = 0
        trainable_params: int = 0

        for param in self.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return total_params, trainable_params

    def summary(self):
        """ Print Model Summary """
        print("*" * WIDTH)
        print(f"Model Summary for {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"Vocabulary Size:          {self.vocab_size}")
        print(f"Embedding Dimension:      {self.embedding_dims}")
        print(f"Maximum Length:           {self.max_len}")
        print(f"Scale size:               {self.scale}")
        print(f"Attention Heads:          {self.num_heads}")
        print(f"Feed-Forward Dim:         {self.feedforward_dims}")
        print(f"Encoder Layers:           {self.num_layers}")
        print(f"Dropout Rate:             {self._dropout_rate}")
        print(f"Activation:               {self.activation}")
        print(f"PAD Token ID:             {self._PAD}")
        print("-" * WIDTH)
        # Calculate parameters
        total_params, trainable_params = self._count_parameters()
        print(f"Total parameters:         {total_params:,}")
        print(f"Trainable parameters:     {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("*" * WIDTH)

    @property
    def encoder(self) -> nn.Module:
        """ Get Transformer Encoder Module
        :return: Transformer Encoder Module
        """
        return self._encoder

    @property
    def vocab_size(self) -> int:
        """ Get Vocabulary Size
        :return: Vocabulary Size
        """
        return self._vocab_size

    @property
    def embedding_dims(self) -> int:
        """ Get Embedding Dimensions
        :return: Embedding Dimensions
        """
        return self._embedding_dims

    @property
    def pad_token(self) -> int:
        """ Get Padding Token Index
        :return: Padding Token Index
        """
        return self._PAD

    @property
    def max_len(self) -> int:
        """ Get Maximum Sequence Length
        :return: Maximum Sequence Length
        """
        return self._max_len

    @property
    def scale(self) -> float:
        """ Get Scaling Factor
        :return: Scaling Factor
        """
        return self._scale

    @property
    def num_heads(self) -> int:
        """ Get Number of Attention Heads
        :return: Number of Attention Heads
        """
        return self._heads

    @property
    def feedforward_dims(self) -> int:
        """ Get Dimension of Feedforward Networks
        :return: Dimension of Feedforward Networks
        """
        return self._feedforward_dims

    @property
    def activation(self) -> str:
        """ Get Activation Function
        :return: Activation Function String
        """
        return self._activation

    @property
    def num_layers(self) -> int:
        """ Get Number of Transformer Layers
        :return: Number of Transformer Layers
        """
        return self._layers


if __name__ == "__main__":
    encoder = TransformerSeqEncoder(
        vocab_size=8700,
        embedding_dims=512,
        max_len=100,
        num_layers=6,
        num_heads=8,
        accelerator="cpu"
    )
    encoder.summary()

    # Test
    batch_size = 2
    seq_len = 10
    input_ids = tensor([
        [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ])

    # Forward
    output = encoder(input_ids)

    print("*" * WIDTH)
    print("Transformer Encoder Test")
    print("-" * WIDTH)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 512), f"Expected ({batch_size}, {seq_len}, 512), got {output.shape}"
    # Check positions result without padding
    print(f"First token outputs without padding (batch 0): {output[0, 0, :5]}")
    print(f"First token outputs with 1 padding (batch 0): {output[0, 0, :6]}")
    print(f"Padding token outputs (batch 0): {output[0, 5, :5]}")
    print("*" * WIDTH)
