#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/2 22:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer.py
# @Desc     :   

from pathlib import Path
from torch import optim, nn

from src.configs.cfg_seq2seq_transformer import CONFIG4S2STF
from src.configs.cfg_types import Tokens, SeqNets, SeqStrategies
from src.configs.parser import set_argument_parser
from src.trainers.trainer4seq2seq_transformer import SeqToSeqTransformerTrainer
from src.nets.seq2seq_transformer import Seq2SeqTransformerNet
from src.utils.stats import load_json
from src.utils.PT import TorchRandomSeed

from pipeline.prepper import prepare_data


def main() -> None:
    """ Main Function """
    # Set up argument parser
    args = set_argument_parser()

    with TorchRandomSeed("Chinese to English (Seq2Seq) Translation", tick_tock=True):
        # Get the dictionary
        dic_cn: Path = Path(CONFIG4S2STF.FILEPATHS.DICTIONARY_CN)
        dictionary_cn = load_json(dic_cn) if dic_cn.exists() else print("Dictionary file not found.")
        dic_en: Path = Path(CONFIG4S2STF.FILEPATHS.DICTIONARY_EN)
        dictionary_en = load_json(dic_en) if dic_en.exists() else print("Dictionary file not found.")
        # print(dictionary_cn[Tokens.PAD])
        # print(dictionary_en[Tokens.PAD])

        # Get the input size and number of classes
        vocab_size4cn: int = len(dictionary_cn)
        vocab_size4en: int = len(dictionary_en)
        print(vocab_size4cn, vocab_size4en)

        # Get the data
        train, valid = prepare_data()

        # Initialize model
        model = Seq2SeqTransformerNet(
            vocab_size_src=vocab_size4cn,
            vocab_size_tgt=vocab_size4en,
            embedding_dims=CONFIG4S2STF.PARAMETERS.EMBEDDING_DIMS,
            scaler=CONFIG4S2STF.PARAMETERS.SCALER,
            max_len=CONFIG4S2STF.PARAMETERS.MAX_LEN,
            num_heads=CONFIG4S2STF.PARAMETERS.HEADS,
            feedforward_dims=CONFIG4S2STF.PARAMETERS.FEEDFORWARD_DIMS,
            num_layers=CONFIG4S2STF.PARAMETERS.LAYERS,
            dropout=CONFIG4S2STF.PREPROCESSOR.DROPOUT_RATIO,
            activation="relu",
            accelerator=CONFIG4S2STF.HYPERPARAMETERS.ACCELERATOR,
            PAD=dictionary_cn[Tokens.PAD],
            SOS=dictionary_cn[Tokens.SOS],
            EOS=dictionary_cn[Tokens.EOS],
        )
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.alpha, weight_decay=CONFIG4S2STF.HYPERPARAMETERS.DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss(ignore_index=dictionary_en[Tokens.PAD])
        model.summary()
        """
        ****************************************************************
        Data Processing Summary:
        ----------------------------------------------------------------
        Train dataset: 20408 Samples
        Valid dataset: 4373 Samples
        Chinese Dictionary Size: 5235
        English Dictionary Size: 3189
        The min length of the sequence: 1
        The average length of the sequence: 6.93
        The max length of the sequence: 28
        ****************************************************************
        ****************************************************************
        Model Summary for Seq2SeqTransformerNet
        ----------------------------------------------------------------
        Source Vocabulary Size:   5235
        Target Vocabulary Size:   3189
        Embedding Dimension:      256
        Maximum Length:           100
        Scale size:               1.0
        Number of Heads:          2
        Feedforward Dimension:    512
        Number of Layers:         2
        Dropout Rate:             0.5
        Activation:               relu
        Accelerator Location:     cpu
        Pad Token Index:          0
        SOS Token Index:          2
        EOS Token Index:          3
        ----------------------------------------------------------------
        Total Parameters:         6,930,805
        Trainable Parameters:     6,930,805
        Non-trainable parameters: 0
        ****************************************************************
        """

        # Setup trainer
        trainer = SeqToSeqTransformerTrainer(
            vocab_size_tgt=vocab_size4en,
            model=model,
            optimiser=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            PAD=dictionary_en[Tokens.PAD],
            SOS=dictionary_en[Tokens.SOS],
            EOS=dictionary_en[Tokens.EOS],
            accelerator=CONFIG4S2STF.HYPERPARAMETERS.ACCELERATOR,
            clip_grad=True,
            top_k=CONFIG4S2STF.PARAMETERS.TOP_K,
            top_p=CONFIG4S2STF.PARAMETERS.TOP_P,
            temperature=CONFIG4S2STF.PREPROCESSOR.TEMPERATURE,
            beam_width=CONFIG4S2STF.PARAMETERS.BEAMS,
            early_stopper=CONFIG4S2STF.PARAMETERS.STOPPER,
            do_sample=CONFIG4S2STF.PARAMETERS.SAMPLER,
            length_penalty_factor=CONFIG4S2STF.PARAMETERS.LEN_PENALTY_FACTOR
        )
        # Train the model
        trainer.fit(
            train_loader=train,
            valid_loader=valid,
            epochs=args.epochs,
            model_save_path=str(CONFIG4S2STF.FILEPATHS.SAVED_NET),
            log_name=f"{SeqNets.TF}-{SeqStrategies.BEAM if CONFIG4S2STF.PARAMETERS.BEAMS > 1 else SeqStrategies.GREEDY}",
        )
        """
        ****************************************************************
        Training Summary:
        ----------------------------------------------------------------
        "epoch": 92/100, "strategy": "greedy", "bleu": 0.2688, "rouge": 0.6271, "avg_pred_len": 7.5065, "perplexity": 7.3681
        ****************************************************************
        """


if __name__ == "__main__":
    main()
