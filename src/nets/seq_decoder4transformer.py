#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/28 19:30
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_decoder4transformer.py
# @Desc     :   

from torch import (nn, Tensor, device,
                   triu, bool as torch_bool, full, ones,
                   randint)
from typing import Literal, final

from src.nets.position_encoders import TransformerPositionEncoder
from src.nets.seq_encoder4transformer import TransformerSeqEncoder
from src.utils.highlighter import lines
from src.utils.PT import TorchRandomSeed

WIDTH: int = 64


class TransformerSeqDecoder(nn.Module):
    """ Pytorch Transformer Decoder Network """

    def __init__(self,
                 vocab_size: int, embedding_dims: int = 512,
                 *,
                 max_len: int = 100, scale: float = 1.0,
                 num_heads: int = 8, feedforward_dims: int = 2048, activation: str | Literal["relu", "gelu"] = "relu",
                 num_layers: int = 6,
                 dropout_rate: float = 0.3,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD: int = 0, SOS: int = 2, EOS: int = 3
                 ) -> None:
        """ Initialize the Transformer Decoder
        :param vocab_size: Size of the target Vocabulary
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
        :param SOS: Start-of-Sequence Token Index
        :param EOS: End-of-Sequence Token Index
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
        # Initialise special token indices
        self._SOS: int = SOS
        self._EOS: int = EOS

        # Initialise embedding layer
        self._embedder = self._init_embeddings()
        # Initialise positional encodings
        self._positions = self._init_positions()
        # Initialise Transformer Encoder
        self._layer = self._init_decoder_layer()
        self._decoder = nn.TransformerDecoder(
            self._layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dims, eps=1e-5, bias=True, device=device(self._accelerator)),
        )
        # Initialise output projection layer
        self._projection = self._init_projection()

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
    def _init_decoder_layer(self) -> nn.TransformerDecoderLayer:
        """ Initialise a Single Transformer Decoder Layer
        :return: Transformer Decoder Layer
        """
        return nn.TransformerDecoderLayer(
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
    def _init_projection(self) -> nn.Linear:
        """ Initialise Output Projection Layer
        :return: Output Projection Layer
        """
        return nn.Linear(
            in_features=self._embedding_dims,
            out_features=self._vocab_size,
            bias=True,
            device=device(self._accelerator)
        )

    @final
    def _init_weights(self) -> None:
        """ Initialise Weights of the Model """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if "_projection" in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(param)

    @final
    def trap_padding_mask(self, sequence: Tensor) -> Tensor:
        """ Create Padding Mask for Input Sequences
        :param sequence: Input Tensor of shape (batch_size, seq_len)
        :return: Padding Mask of shape (batch_size, seq_len)
        """
        return sequence == self._PAD

    @final
    def _trap_causal_mask(self, size: int) -> Tensor:
        """ Create Causal Mask for Target Sequences
        :param size: Size of the sequence
        :return: Causal Mask Tensor of shape (size, size)
        """
        # Create lower triangular matrix (upper diagonal)
        return triu(ones((size, size), dtype=torch_bool, device=device(self._accelerator)), diagonal=1)

    def forward(self,
                tgt: Tensor, memory: Tensor,
                *,
                tgt_key_padding_mask: Tensor | None = None, memory_key_padding_mask: Tensor | None = None
                ) -> Tensor:
        """ Forward Pass through the Transformer Decoder
        :param tgt: Target Input Tensor of shape [batch_size, tgt_seq_len]
        :param memory: Encoder Output Tensor of shape [batch_size, src_seq_len, embedding_dim]
        :param tgt_key_padding_mask: Padding mask for target sequences [batch_size, tgt_seq_len]
        :param memory_key_padding_mask: Padding mask for source sequences [batch_size, src_seq_len]
        :return: Output Tensor of shape [batch_size, tgt_seq_len, vocab_size]
        """
        # Embed target sequences [batch_size, tgt_seq_len, embedding_dim]
        embeddings: Tensor = self._embedder(tgt)

        # Add positional encodings
        embeddings_with_pos: Tensor = self._positions(embeddings)

        # Create target causal mask - Causal mask for decoder self-attention [tgt_len, tgt_len]
        tgt_len: int = tgt.size(1)
        tgt_mask: Tensor = self._trap_causal_mask(tgt_len)

        # Create target padding mask - Padding mask for target sequences [batch_size, tgt_seq_len]
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask: Tensor = self.trap_padding_mask(tgt)

        # Transformer Decoding [batch_size, tgt_seq_len, embedding_dim]
        outputs: Tensor = self._decoder(
            tgt=embeddings_with_pos,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Output Projection to Vocabulary Size [batch_size, tgt_seq_len, vocab_size]
        logits: Tensor = self._projection(outputs)

        return logits

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
        print(f"SOS Token ID:             {self._SOS}")
        print(f"EOS Token ID:             {self._EOS}")
        print("-" * WIDTH)
        # Calculate parameters
        total_params, trainable_params = self._count_parameters()
        print(f"Total parameters:         {total_params:,}")
        print(f"Trainable parameters:     {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("*" * WIDTH)

    @property
    def decoder(self) -> nn.Module:
        """ Get Transformer Decoder Module
        :return: Transformer Decoder Module
        """
        return self._decoder

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
    def pad_token(self):
        """ Get Padding Token Index
        :return: Padding Token Index
        """
        return self._PAD

    @property
    def max_len(self) -> int:
        """ Get Maximum Length of Input Sequences
        :return: Maximum Length
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
        :return: Feedforward Dimension
        """
        return self._feedforward_dims

    @property
    def activation(self) -> str:
        """ Get Activation Function
        :return: Activation Function
        """
        return self._activation

    @property
    def num_layers(self) -> int:
        """ Get Number of Transformer Layers
        :return: Number of Layers
        """
        return self._layers

    @property
    def sos_token(self) -> int:
        """ Get Start-of-Sequence Token Index
        :return: SOS Token Index
        """
        return self._SOS

    @property
    def eos_token(self) -> int:
        """ Get End-of-Sequence Token Index
        :return: EOS Token Index
        """
        return self._EOS


if __name__ == "__main__":
    with TorchRandomSeed("Transformer Encoder-Decoder Test", tick_tock=True):
        vocab_size_src = 10_000
        vocab_size_tgt = 8_000
        embedding_dims = 512
        max_len = 100
        batch_size = 4
        src_seq_len = 20
        tgt_seq_len = 15
        accelerator = "cpu"

        print(f"Model Parameters:")
        print(f"SRC Vocab Size:      {vocab_size_src}")
        print(f"TGT Vocab Size:      {vocab_size_tgt}")
        print(f"Embedding Dims:      {embedding_dims}")
        print(f"Max Length:          {max_len}")
        print(f"Batch size:          {batch_size}")
        print(f"SRC sequence length: {src_seq_len}")
        print(f"TGT sequence length: {tgt_seq_len}")
        lines()
        print()

        encoder = TransformerSeqEncoder(
            vocab_size=vocab_size_src,
            embedding_dims=embedding_dims,
            max_len=max_len,
            num_heads=8,
            feedforward_dims=2048,
            activation="relu",
            num_layers=6,
            dropout_rate=0.1,
            accelerator=accelerator,
            PAD=0
        )
        encoder.summary()

        decoder = TransformerSeqDecoder(
            vocab_size=vocab_size_tgt,
            embedding_dims=embedding_dims,
            max_len=max_len,
            num_heads=8,
            feedforward_dims=2048,
            activation="relu",
            num_layers=6,
            dropout_rate=0.1,
            accelerator=accelerator,
            PAD=0,
            SOS=2,
            EOS=3,
        )
        decoder.summary()

        lines()
        src = randint(3, vocab_size_src, (batch_size, src_seq_len)).to(accelerator)
        print(f"SRC Shape: {src.shape}")
        print(f"SRC Sequence:\n{src[0]}")

        lines()
        tgt = randint(3, vocab_size_tgt, (batch_size, tgt_seq_len)).to(accelerator)
        print(f"TGT Shape: {tgt.shape}")
        print(f"TGT Sequence:\n{tgt[0]}")
