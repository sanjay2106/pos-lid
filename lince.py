from typing import Optional

import numpy as np
from sklearn.model_selection import KFold
import torch
import pytorch_lightning as pl

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import datasets as ds

from config import (
    BATCH_SIZE,
    GLOBAL_SEED,
    K_CROSSFOLD_VALIDATION_SPLITS,
    LABEL2ID,
    LID2ID,
    MAX_SEQUENCE_LENGTH,
    NUM_WORKERS,
    PADDING,
    PATH_BASE_MODELS,
    PATH_CACHE_DATASET,
    PATH_LINCE_DATASET
)

class LinceDM(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_dir=PATH_LINCE_DATASET,
        batch_size: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        label2id: dict = LABEL2ID,
        lid2id: dict = LID2ID,
        num_workers: int = NUM_WORKERS,
    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.label2id = label2id
        self.lid2id = lid2id
        self.num_workers = num_workers

        # Define data paths
        self.data_map = {
            "lince": {
                "train": [f"{self.dataset_dir}/train.json"],
                "validation": [f"{self.dataset_dir}/val.json"]
            }
        }

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            cache_dir=PATH_BASE_MODELS,
            use_fast=True,
        )

    def prepare_data(self) -> None:
        # Prepare dataset
        ds.load_dataset(
            'json',
            data_files=self.data_map[self.dataset_name],
            field='data',
            cache_dir=PATH_CACHE_DATASET
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Load dataset
        self.dataset = ds.load_dataset(
            'json',
            data_files=self.data_map[self.dataset_name],
            field="data",
            cache_dir=PATH_CACHE_DATASET
        )

        # Tokenize and align labels for training and validation datasets
        self.dataset['train'] = self.dataset['train'].map(
            self._convert_to_features,
            batched=True,
            drop_last_batch=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        self.dataset['validation'] = self.dataset['validation'].map(
            self._convert_to_features,
            batched=True,
            drop_last_batch=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        # Set format for PyTorch
        self.dataset['train'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'lids'])
        self.dataset['validation'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'lids'])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def _convert_to_features(self, batch, indices=None):
        # Tokenize input text
        features = self.tokenizer(
            text=batch["tokens"],
            max_length=self.max_seq_len,
            padding=self.padding,
            truncation=True,
            is_split_into_words=True,
        )

        # Align POS tags and LID tags with tokenized input
        features["labels"], features["lids"] = self._align_tags(features, batch['pos_tags'], batch["lang_tags"])
        return features

    def _align_tags(self, tokenized_outs, tags, lids):
        # Align POS tags with tokenized outputs
        batch_tags = []
        batch_lids = []
        for example_id in range(0, len(tags)):
            example_tags = []
            example_lids = []
            current_word = None

            for word_id in tokenized_outs.word_ids(example_id):
                # Align POS tags
                if word_id != current_word:
                    current_word = word_id
                    tag = len(self.label2id) if word_id is None else self.label2id[tags[example_id][word_id]]
                    example_tags.append(tag)
                elif word_id is None:
                    example_tags.append(len(self.label2id))
                else:
                    tag = self.label2id[tags[example_id][word_id]]
                    example_tags.append(tag)

                # Align LID tags
                if word_id != current_word:
                    lid = len(self.lid2id) if word_id is None else self.lid2id[lids[example_id][word_id]]
                    example_lids.append(lid)
                elif word_id is None:
                    example_lids.append(len(self.lid2id))
                else:
                    lid = self.lid2id[lids[example_id][word_id]]
                    example_lids.append(lid)

            batch_tags.append(example_tags)
            batch_lids.append(example_lids)

        return batch_tags, batch_lids


class CrossValidationLinceDM(LinceDM):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        k: int,
        dataset_dir=PATH_LINCE_DATASET,
        batch_size: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        label2id: dict = LABEL2ID,
        lid2id: dict = LID2ID,
        num_splits: int = K_CROSSFOLD_VALIDATION_SPLITS,
        num_workers: int = NUM_WORKERS,
        split_seed: int = GLOBAL_SEED,
    ) -> None:
        super().__init__(model_name, dataset_name)
        self.save_hyperparameters(logger=False)

    def prepare_data(self) -> None:
        # Load dataset
        ds.load_dataset(
            'json',
            data_files=f"{self.hparams.dataset_dir}/data.json",
            field='data',
            cache_dir=PATH_CACHE_DATASET
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Load dataset and perform K-Fold split
        self.dataset = ds.load_dataset(
            'json',
            data_files=f"{self.hparams.dataset_dir}/data.json",
            field='data',
            cache_dir=PATH_CACHE_DATASET
        )

        # Perform K-Fold cross-validation split
        kf = KFold(n_splits=10, shuffle=True, random_state=self.hparams.split_seed)
        splits = kf.split(np.zeros(self.dataset['train'].num_rows))
        all_splits = [k for k in splits]
        train_idxs, val_idxs = all_splits[self.hparams.k]

        # Split the dataset into training, validation, and test sets
        self.dataset = ds.DatasetDict({
            'train': self.dataset['train'].select(train_idxs),
            'validation': self.dataset['train'].select(val_idxs),
            'test': self.dataset['train'].select(val_idxs)
        })

        # Tokenize and align labels
        self.dataset['train'] = self.dataset['train'].map(
            self._convert_to_features,
            batched=True,
            drop_last_batch=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        self.dataset['validation'] = self.dataset['validation'].map(
            self._convert_to_features,
            batched=True,
            drop_last_batch=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        self.dataset['test'] = self.dataset['test'].map(
            self._convert_to_features,
            batched=True,
            drop_last_batch=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        # Set format for PyTorch
        self.dataset['train'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'lids'])
        self.dataset['validation'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'lids'])
        self.dataset['test'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'lids'])

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )
