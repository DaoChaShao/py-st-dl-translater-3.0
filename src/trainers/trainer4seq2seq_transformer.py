#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/30 15:53
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer4seq2seq_transformer.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from datetime import datetime
from json import dumps
from torch import nn, optim, no_grad, save, device, tensor, exp, Tensor
from typing import Literal

from src.dataloaders import TorchDataLoader
from src.trainers.calc4seq_text_quilty import TextQualityScorer
from src.utils.logger import record_log
from src.utils.PT import get_device

WIDTH: int = 64


class SeqToSeqTransformerTrainer(QObject):
    """ Trainer class for managing seq2seq transformer training process """
    losses: Signal = Signal(int, float, float)

    def __init__(self,
                 vocab_size_tgt: int,
                 model: nn.Module, optimiser, criterion, scheduler=None,
                 PAD: int = 0, SOS: int = 2, EOS: int = 3,
                 accelerator: str = "auto", clip_grad: bool = True,
                 *,
                 top_k: int = 50, top_p: float = 0.7, temperature: float = 1.0,
                 beam_width: int | Literal[1, 3, 5, 10, 20] = 1,
                 early_stopper: bool = True, do_sample: bool = True,
                 length_penalty_factor: float = 0.6,
                 ) -> None:
        """ Initialise the TorchTrainer4SeqToSeq class
        :param vocab_size_tgt: size of the targert vocabulary
        :param model: the seq2seq model to be trained
        :param optimiser: the optimiser for training
        :param criterion: the loss function
        :param scheduler: learning rate scheduler (optional)
        :param PAD: padding token index
        :param SOS: start-of-sequence token index
        :param EOS: end-of-sequence token index
        :param accelerator: device to use for training ("cpu", "cuda", "auto", etc.)
        :param clip_grad: whether to apply gradient clipping
        :param top_k: top-k sampling parameter
        :param top_p: top-p (nucleus) sampling parameter
        :param temperature: sampling temperature
        :param beam_width: beam width for beam search decoding
        :param early_stopper: whether to use early stopping
        :param length_penalty_factor: length penalty factor for beam search
        :param do_sample: whether to use sampling during generation
        """
        super().__init__()
        self._vocab_size: int = vocab_size_tgt
        self._PAD: int = PAD
        self._SOS: int = SOS
        self._EOS: int = EOS
        self._clip: bool = clip_grad

        self._K: int = top_k
        self._P: float = top_p
        self._T: float = temperature
        self._beams: int = beam_width
        self._stopper: bool = early_stopper
        self._do_sample: bool = do_sample
        self._length_penalty: float = length_penalty_factor

        self._accelerator: str = get_device(accelerator.lower())
        self._model = model.to(device(self._accelerator))
        self._optimiser = optimiser
        self._criterion = criterion
        self._scheduler = scheduler

    def _epoch_train(self, dataloader: TorchDataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _total: float = 0.0
        for src, tgt in dataloader:
            src, tgt = src.to(device(self._accelerator)), tgt.to(device(self._accelerator))

            self._optimiser.zero_grad()

            # Teacher Forcing
            logits = self._model(src, tgt)
            # print(logits.shape)  # [batch_size, tgt_seq_len, vocab_size]

            # Calculate loss
            loss = self._criterion(
                logits[:, :-1, :].reshape(-1, self._vocab_size),  # [batch*(tgt_len-1), vocab]
                tgt[:, 1:].reshape(-1)  # [batch*(tgt_len-1)]
            )

            loss.backward()

            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0) if self._clip else None

            self._optimiser.step()

            _loss += loss.item() * src.size(0)
            _total += src.size(0)

        return _loss / _total

    def _epoch_valid(self, dataloader: TorchDataLoader) -> tuple[float, dict[str, float]]:
        """ Inference (validation) for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        _total: float = 0.0

        _predictions: list[list[int]] = []
        _references: list[list[int]] = []
        with no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(device(self._accelerator)), tgt.to(device(self._accelerator))

                logits = self._model(src, tgt[:, :-1])
                # print(logits.shape)

                # Calculate loss as same as training
                loss = self._criterion(
                    logits.reshape(-1, logits.size(-1)),  # [batch*(tgt_len-1), vocab]
                    tgt[:, 1:].reshape(-1)  # [batch*(tgt_len-1)]
                )

                _loss += loss.item() * src.size(0)
                _total += src.size(0)

                # Collect predictions and references for evaluation
                memory: Tensor = self._model.encoder(src)
                memory_key_padding_mask: Tensor = self._model.get_memory_padding_mask(src)
                pred_seq: Tensor = self._model.generate(
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    top_k=self._K,
                    top_p=self._P,
                    temperature=self._T,
                    beams=self._beams,
                    early_stopper=self._stopper,
                    do_sample=self._do_sample,
                    length_penalty=self._length_penalty,
                )

                for i in range(tgt.size(0)):
                    _pred = self._clean_tokens(pred_seq[i], PAD=self._PAD, SOS=self._SOS, EOS=self._EOS)
                    _ref = self._clean_tokens(tgt[i], PAD=self._PAD, SOS=self._SOS, EOS=self._EOS)

                    _predictions.append(_pred)
                    _references.append(_ref)

        with TextQualityScorer(self._accelerator) as scorer:
            _metrics: dict[str, float] = {
                **scorer.calculate(_predictions, _references),
            }

        avg_val_loss = _loss / _total if _total > 0 else float("inf")
        dps: int = 4
        _metrics["perplexity"] = round(float(exp(tensor(avg_val_loss)).item()), dps)
        _metrics["avg_val_loss"] = round(float(avg_val_loss), dps)

        return _loss / _total, _metrics

    @staticmethod
    def _clean_tokens(seq, *, PAD: int, SOS: int, EOS: int):
        """ Clean token sequence by removing PAD, SOS, and truncating at EOS
        :param seq: token sequence
        :param PAD: padding token index
        :param SOS: start-of-sequence token index
        :param EOS: end-of-sequence token index
        """
        seq = seq.tolist()

        if EOS in seq:
            seq = seq[:seq.index(EOS)]

        return [t for t in seq if t not in (PAD, SOS)]

    def fit(self,
            train_loader: TorchDataLoader, valid_loader: TorchDataLoader,
            epochs: int, model_save_path: str | None = None, log_name: str | None = None
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :param log_name: name for the training log file
        :return: None
        """
        # Initialize logger
        timer = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        logger = record_log(f"train_at_{timer}-{log_name}")

        _best_valid_loss = float("inf")
        _min_delta: float = 5e-4
        _patience: int = 10
        _patience_counter: int = 0
        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss, _metrics = self._epoch_valid(valid_loader)

            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss)

            # Log epoch results
            dps: int = 4
            logger.info(dumps({
                "epochs": epochs,
                "epoch": epoch + 1,
                "strategy": "greedy" if self._beams == 1 else f"{self._beams} beams",
                "alpha": self._optimiser.param_groups[0]["lr"],
                "train_loss": round(float(train_loss), dps),
                "valid_loss": round(float(valid_loss), dps),
                **_metrics,
            }))

            # Save the model if it has the best validation loss so far
            if valid_loss < _best_valid_loss - _min_delta:
                _patience_counter = 0
                _best_valid_loss = valid_loss

                if model_save_path:
                    save(self._model.state_dict(), model_save_path)
                    print(f"ooo | Model's parameters saved to {model_save_path}\n")
            else:
                _patience_counter += 1
                print(f"xxx | No improvement [{_patience_counter}/{_patience}]\n")
                if _patience_counter >= _patience:
                    print("*" * WIDTH)
                    print("Early Stopping Triggered")
                    print("-" * WIDTH)
                    print(f"Early stopping at epoch {epoch}, the best value is {_best_valid_loss:.4f}.")
                    print("*" * WIDTH)
                    print()
                    break

            if self._scheduler is not None:
                if isinstance(self._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(valid_loss)
                else:
                    self._scheduler.step()

        if _patience_counter < _patience:
            print(f"Training completed after {epochs} epochs.")


if __name__ == "__main__":
    pass
