#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/30 14:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_seq2seq_transformer.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import Database, FilePaths, Punctuations
from src.configs.cfg_dl import DataPreprocessor, Hyperparameters


@dataclass
class TransformerParams:
    BEAMS: int = 1
    EMBEDDING_DIMS: int = 256
    FEEDFORWARD_DIMS: int = 512
    HEADS: int = 2
    LAYERS: int = 2
    LEN_PENALTY_FACTOR: float = 0.6  # 0.6 is better for mid-length seq, 1.0 for short seq and 1.2 for long seq
    MAX_LEN: int = 100
    SCALER: float = 1.0
    TOP_K: int = 50
    TOP_P: float = 0.7
    STOPPER: bool = True
    SAMPLER: bool = True


@dataclass
class Configuration4S2STransformer:
    DATABASE: Database = field(default_factory=Database)
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PARAMETERS: TransformerParams = field(default_factory=TransformerParams)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


CONFIG4S2STF = Configuration4S2STransformer()
