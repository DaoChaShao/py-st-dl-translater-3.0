#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/29 15:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_transformer.py
# @Desc     :   

from pathlib import Path
from torch import (nn, Tensor, save, load,
                   no_grad, device, full, long, zeros, bool as torch_bool,
                   cat, topk, sort, cumsum, zeros_like, stack,
                   tensor,
                   randint)
from typing import final, Literal

from src.nets.seq_encoder4transformer import TransformerSeqEncoder
from src.nets.seq_decoder4transformer import TransformerSeqDecoder

WIDTH: int = 64


class Seq2SeqTransformerNet(nn.Module):
    """ Sequence-to-Sequence Transformer Model """

    def __init__(self,
                 vocab_size_src: int, vocab_size_tgt: int, embedding_dims: int | Literal[256, 512, 768, 1024],
                 *,
                 scaler: float = 1.0, max_len: int = 100,
                 num_heads: int = 8, feedforward_dims: int = 2048, num_layers: int = 6,
                 dropout: float = 0.3,
                 activation: str | Literal["gelu", "relu"] = "relu",
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD: int = 0, SOS: int = 2, EOS: int = 3,
                 ) -> None:
        """ Initialize the Seq2Seq Transformer Model
        :param vocab_size_src: Vocabulary size of the source language
        :param vocab_size_tgt: Vocabulary size of the target language
        :param embedding_dims: Dimension of the embeddings
        :param scaler: Scaling factor for embeddings
        :param max_len: Maximum length of the input sequences
        :param num_heads: Number of attention heads
        :param feedforward_dims: Dimension of the feedforward network
        :param num_layers: Number of transformer layers
        :param dropout: Dropout rate
        :param activation: Activation function to use ('gelu' or 'relu')
        :param accelerator: Accelerator type, either 'cpu' or 'cuda'
        :param PAD: Padding token index
        :param SOS: Start-of-sequence token index
        :param EOS: End-of-sequence token index
        """
        super().__init__()
        self._src_size: int = vocab_size_src
        self._dim_model: int = embedding_dims
        self._max_len: int = max_len
        self._scaler: float = scaler
        self._heads: int = num_heads
        self._feedforward_dims: int = feedforward_dims
        self._activation: str | Literal["gelu", "relu"] = activation.lower()
        self._layers: int = num_layers
        self._dropout_rate: float = dropout
        self._accelerator: str | Literal["cuda", "cpu"] = accelerator.lower()
        self._PAD: int = PAD

        self._tgt_size: int = vocab_size_tgt
        self._SOS: int = SOS
        self._EOS: int = EOS

        # Initialize Encoder and Decoder
        self._encoder: TransformerSeqEncoder = self._init_transformer_encoder()
        self._decoder: TransformerSeqDecoder = self._init_transformer_decoder()

        # Initialize Weights
        self._init_weights()

    @final
    def _init_transformer_encoder(self) -> TransformerSeqEncoder:
        """ Initialize the Transformer Encoder """
        return TransformerSeqEncoder(
            vocab_size=self._src_size,
            embedding_dims=self._dim_model,
            max_len=self._max_len,
            scale=self._scaler,
            num_heads=self._heads,
            feedforward_dims=self._feedforward_dims,
            activation=self._activation,
            num_layers=self._layers,
            dropout_rate=self._dropout_rate,
            accelerator=self._accelerator,
            PAD=self._PAD,
        )

    @final
    def _init_transformer_decoder(self) -> TransformerSeqDecoder:
        """ Initialize the Transformer Decoder """
        return TransformerSeqDecoder(
            vocab_size=self._tgt_size,
            embedding_dims=self._dim_model,
            max_len=self._max_len,
            scale=self._scaler,
            num_heads=self._heads,
            feedforward_dims=self._feedforward_dims,
            activation=self._activation,
            num_layers=self._layers,
            dropout_rate=self._dropout_rate,
            accelerator=self._accelerator,
            PAD=self._PAD,
            SOS=self._SOS,
            EOS=self._EOS,
        )

    @final
    def _init_weights(self) -> None:
        """ Initialize Weights of the Model """
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    @final
    def get_memory_padding_mask(self, src: Tensor) -> Tensor:
        """ Get Padding Mask for Encoder Memory
        :param src: Source sequence tensor
        :return: Padding mask tensor
        """
        return self._encoder.trap_padding_mask(src)

    @final
    def _get_tgt_padding_mask(self, tgt: Tensor) -> Tensor:
        """ Get Padding Mask for Target Sequences
        :param tgt: Target sequence tensor
        :return: Padding mask tensor
        """
        return self._decoder.trap_padding_mask(tgt)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward Pass of the Seq2Seq Transformer Model
        :param src: Source sequence tensor
        :param tgt: Target sequence tensor
        :return: Output tensor
        """
        # Create padding masks
        memory_key_padding_mask: Tensor = self.get_memory_padding_mask(src)
        tgt_key_padding_mask: Tensor = self._get_tgt_padding_mask(tgt)

        # Forward pass
        memory: Tensor = self._encoder(src)
        # [batch_size, tgt_seq_len, vocab_size_tgt]
        logits: Tensor = self._decoder(
            tgt, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return logits

    @no_grad()
    def generate(self,
                 memory: Tensor,
                 memory_key_padding_mask: Tensor | None = None,
                 *,
                 top_k: int = 50, top_p: float = 0.7, temperature: float = 1.0,
                 beams: int | Literal[1, 3, 5, 10, 20] = 1,
                 early_stopper: bool = True,
                 do_sample: bool = True,
                 length_penalty: float = 0.6
                 ) -> Tensor:
        """ Generate Sequences Autoregressively from Encoder Memory
        :param memory: Encoder Output Tensor of shape [batch_size, src_seq_len, embedding_dim]
        :param memory_key_padding_mask: Padding mask for source sequences [batch_size, src_seq_len]
        :param top_k: Top-K Sampling
        :param top_p: Top-P (Nucleus) Sampling
        :param temperature: Decoder Output Tensor of shape [batch_size, tgt_seq_len]
        :param beams: Number of Beams for Beam Search
        :param early_stopper: Whether to Stop Early if all sequences are finished
        :param memory_key_padding_mask: Padding mask for source sequences [batch_size, src_seq_len]
        :param do_sample: Whether to Do Sampling
        :param length_penalty: Length Penalty for Beam Search
        :return: Generated Sequences Tensor of shape [batch_size, generated_seq_len]
        """
        batches = memory.size(0)
        hardware: device = memory.device

        # Initialize with SOS token
        generated = full((batches, 1), self._SOS, dtype=long, device=hardware)
        finished = zeros(batches, dtype=torch_bool, device=hardware)

        # For beam search
        if beams > 1:
            seq, _ = self._beam_search(memory, memory_key_padding_mask, beams, early_stopper, length_penalty)
            return seq

        # For greedy/search sampling
        for _ in range(self._max_len - 1):  # -1 because we already have BOS
            if early_stopper and finished.all():
                break

            # Get logits for current sequence [batch_size, current_len, vocab_size]
            logits = self._decoder(
                tgt=generated,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=self._get_tgt_padding_mask(generated)
            )

            # Get last token logits [batch_size, vocab_size]
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Apply top-k filtering
            if top_k > 0:
                kth = topk(next_logits, top_k)[0][..., -1, None]
                next_logits[next_logits < kth] = float("-inf")

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = sort(next_logits, descending=True)
                probs = nn.functional.softmax(sorted_logits, dim=-1)
                cum_probs = cumsum(probs, dim=-1)

                mask = cum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False

                for b in range(batches):
                    next_logits[b, sorted_indices[b][mask[b]]] = float("-inf")

            if (next_logits == float("-inf")).all():
                next_logits[:, :] = 0.0

            probs = nn.functional.softmax(next_logits, dim=-1)

            if do_sample:
                # Sample next token
                next_tokens = probs.multinomial(num_samples=1)
            else:
                # Greedy argmax
                next_tokens = probs.argmax(dim=-1, keepdim=True)

            generated = cat([generated, next_tokens], dim=1)
            finished |= (next_tokens.squeeze() == self._EOS)

        return generated

    def _beam_search(self,
                     memory: Tensor,
                     memory_key_padding_mask: Tensor | None,
                     beams: int | Literal[5, 10, 20],
                     early_stopper: bool,
                     alpha: float = 0.6
                     ) -> tuple[Tensor, Tensor]:
        """ Beam Search Generation
        :param memory: Encoder Output Tensor of shape [batch_size, src_seq_len, embedding_dim]
        :param memory_key_padding_mask: Padding mask for source sequences [batch_size, src_seq_len]
        :param beams: Number of Beams for Beam Search
        :param early_stopper: Whether to Stop Early if all sequences are finished
        :param alpha: Length Penalty Factor
        :return: Generated Sequences Tensor of shape [batch_size, generated_seq_len]
        """
        B, S, D = memory.shape
        hardware: device = memory.device

        # Initialize beams
        beam_scores = zeros((B, beams), device=hardware)
        beam_sequence = full((B, beams, 1), self._SOS, dtype=long, device=hardware)
        beam_finished = zeros((B, beams), dtype=torch_bool, device=hardware)

        # Expand memory for beam search
        memory = memory.unsqueeze(1).repeat(1, beams, 1, 1).view(B * beams, S, D)

        # Get memory key padding mask
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).repeat(1, beams, 1).view(B * beams, -1)

        for _ in range(self._max_len - 1):
            if early_stopper and beam_finished.all():
                break

            flat_seq = beam_sequence.view(B * beams, -1)

            logits = self._decoder(
                tgt=flat_seq,
                memory=memory,
                tgt_key_padding_mask=self._decoder.trap_padding_mask(flat_seq),
                memory_key_padding_mask=memory_key_padding_mask
            )

            logits_prob = nn.functional.log_softmax(logits[:, -1, :], dim=-1)
            logits_prob = logits_prob.view(B, beams, -1)

            # log prob: [B, beams, vocab]
            if beam_finished.any():
                # [B, beams, 1]
                mask = beam_finished.unsqueeze(-1)

                # Mask all tokens for finished beams,
                # then re-enable EOS as the only valid continuation
                logits_prob = logits_prob.masked_fill(mask, float("-inf"))
                # Cause a type mismatch between attn_mask (bool) and float(-inf)
                # logits_prob[..., self._EOS] = logits_prob[..., self._EOS].masked_fill(
                #     ~mask.squeeze(-1), logits_prob[..., self._EOS]
                # )
                # Now EOS positions in unfinished beams are set to 0.0, matching the bool mask type
                logits_prob[..., self._EOS] = logits_prob[..., self._EOS].masked_fill(~mask.squeeze(-1), 0.0)

            scores = beam_scores.unsqueeze(-1) + logits_prob

            flat_scores = scores.view(B, -1)
            top_scores, top_idx = topk(flat_scores, beams)

            beam_idx = top_idx // logits_prob.size(-1)
            token_idx = top_idx % logits_prob.size(-1)

            new_seq = []
            new_finished = zeros_like(beam_finished)

            for b in range(B):
                seqs = []
                for k in range(beams):
                    old = beam_idx[b, k]
                    tok = token_idx[b, k]
                    seq = cat([beam_sequence[b, old], tok.view(1)])
                    seqs.append(seq)

                    new_finished[b, k] = beam_finished[b, old] or tok == self._EOS
                new_seq.append(stack(seqs))

            beam_sequence = stack(new_seq)
            beam_scores = top_scores
            beam_finished = new_finished

        seq_lengths = (beam_sequence != self._PAD).sum(dim=-1)  # [B, beams]
        # Compute length penalty
        # - The formula is from GNMT(Google Neural Machine Translation)
        # - 5: constant to avoid penalizing short sequences too much, which is named as "baseline length"
        # - 6: constant to normalize the penalty, which is named as "normalization factor"
        # - alpha: hyperparameter to control the strength of the penalty
        # -- alpha = 0.6: better performance in mid and long sequences
        # -- alpha = 1.0: better performance in short sequences
        # -- alpha > 1.0: stronger penalty on long sequences
        length_penalty = ((5 + seq_lengths).float() / 6) ** alpha  # [B, beams]
        # Apply length penalty
        beam_scores = beam_scores / length_penalty

        best = beam_scores.argmax(dim=-1)
        final_seq = stack([beam_sequence[b, best[b]] for b in range(B)])

        return final_seq, beam_scores.max(dim=-1).values

    @final
    def save_params(self, path: str | Path) -> None:
        """ Save the model - all networks share the same method
        :param path: path to save the model
        """
        save(self.state_dict(), path)
        print("The model has been saved successfully.")

    @final
    def load_params(self, path: str | Path, strict: bool = False) -> None:
        """ Load the model - all networks share the same method
        :param path: path to load the model from
        :param strict: whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict function
        """
        self.load_state_dict(load(path, map_location=device(self._accelerator)), strict=strict)
        print("The model has been loaded successfully.")

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
        print(f"Source Vocabulary Size:   {self.src_vocab_size}")
        print(f"Target Vocabulary Size:   {self.tgt_vocab_size}")
        print(f"Embedding Dimension:      {self.dim_model}")
        print(f"Maximum Length:           {self.max_len}")
        print(f"Scale size:               {self.scaler}")
        print(f"Number of Heads:          {self.num_heads}")
        print(f"Feedforward Dimension:    {self.feedforward_dims}")
        print(f"Number of Layers:         {self.num_layers}")
        print(f"Dropout Rate:             {self._dropout_rate}")
        print(f"Activation:               {self._activation}")
        print(f"Accelerator Location:     {self._accelerator}")
        print(f"Pad Token Index:          {self.pad_token}")
        print(f"SOS Token Index:          {self.sos_token}")
        print(f"EOS Token Index:          {self.eos_token}")
        print("-" * WIDTH)
        # Calculate parameters
        total_params, trainable_params = self._count_parameters()
        print(f"Total Parameters:         {total_params:,}")
        print(f"Trainable Parameters:     {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("*" * WIDTH)

    @property
    def encoder(self) -> nn.Module:
        """ Get the Transformer Encoder """
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        """ Get the Transformer Decoder """
        return self._decoder

    @property
    def src_vocab_size(self) -> int:
        """ Get source vocabulary size """
        return self._src_size

    @property
    def tgt_vocab_size(self) -> int:
        """ Get target vocabulary size """
        return self._tgt_size

    @property
    def dim_model(self) -> int:
        """ Get model dimension """
        return self._dim_model

    @property
    def max_len(self) -> int:
        """ Get maximum sequence length """
        return self._max_len

    @property
    def scaler(self) -> float:
        """ Get scaling factor """
        return self._scaler

    @property
    def num_heads(self) -> int:
        """ Get number of attention heads """
        return self._heads

    @property
    def feedforward_dims(self) -> int:
        """ Get feedforward network dimension """
        return self._feedforward_dims

    @property
    def num_layers(self) -> int:
        """ Get number of transformer layers """
        return self._layers

    @property
    def pad_token(self) -> int:
        """ Get padding token index """
        return self._PAD

    @property
    def sos_token(self) -> int:
        """ Get start-of-sequence token index """
        return self._SOS

    @property
    def eos_token(self) -> int:
        """ Get end-of-sequence token index """
        return self._EOS


if __name__ == "__main__":
    SRC_VOCAB = 50
    TGT_VOCAB = 50
    EMB_DIM = 32
    MAX_LEN = 10
    BATCH = 2
    BEAMS = 3

    net = Seq2SeqTransformerNet(
        vocab_size_src=SRC_VOCAB,
        vocab_size_tgt=TGT_VOCAB,
        embedding_dims=EMB_DIM,
        max_len=MAX_LEN,
        num_heads=4,
        num_layers=2,
        PAD=0, SOS=1, EOS=2
    )

    # random tensor [batch_size, src_len]
    src = randint(3, SRC_VOCAB, (BATCH, MAX_LEN))
    memory = net._encoder(src)
    mem_mask = net.get_memory_padding_mask(src)

    # Greedy Test
    print("*" * WIDTH)
    print("Memory Greedy Test with do_sample=False")
    print("-" * WIDTH)
    greedy_seq = net.generate(memory, memory_key_padding_mask=mem_mask, beams=1, do_sample=False)
    print(greedy_seq)
    print("*" * WIDTH)
    print()

    # Sampling (top-k + top-p + temperature)
    sample_seq = net.generate(
        memory,
        memory_key_padding_mask=mem_mask,
        beams=1,
        do_sample=True,
        top_k=10,
        top_p=0.8,
        temperature=1.0
    )
    print("*" * WIDTH)
    print("Memory Sampling Test with top-k=10, top-p=0.8, temperature=1.0 and do_sample=True")
    print("-" * WIDTH)
    print(sample_seq)
    print("*" * WIDTH)

    # Beam Search Test
    print("*" * WIDTH)
    print("Memory Beam Search Test")
    print("-" * WIDTH)
    beam_seq = net.generate(memory, memory_key_padding_mask=mem_mask, beams=BEAMS)
    print(beam_seq)
    print("*" * WIDTH)
