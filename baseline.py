from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional import precision, recall, f1_score
from TorchCRF import CRF
from transformers.optimization import AdamW

from src.modules.base_model import BaseModel
from src.modules.mtl_loss import MultiTaskLossWrapper

from config import (
    LABEL2ID,
    LEARNING_RATE,
    LID2ID,
    WARM_RESTARTS,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    MAX_SEQUENCE_LENGTH,
    PADDING
)

class BaseLine(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        label2id: dict = LABEL2ID,
        lid2id: dict = LID2ID,
        learning_rate: float = LEARNING_RATE,
        pos_learning_rate: float = LEARNING_RATE,  # Updated parameter
        lid_learning_rate: float = LEARNING_RATE,
        warm_restart_epochs: int = WARM_RESTARTS,
        weight_decay: float = WEIGHT_DECAY,
        pos_wd: float = WEIGHT_DECAY,  # Updated parameter
        lid_wd: float = WEIGHT_DECAY,
        dropout_rate: float = DROPOUT_RATE,
        freeze: bool = False
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.lid_pad_token_label = len(self.hparams.lid2id)
        self.pos_pad_token_label = len(self.hparams.label2id)

        # Shared params
        self.base_model = BaseModel(self.hparams.model_name)

        # Freeze pre-trained model
        if self.hparams.freeze:
            self.base_model.freeze()

        self.bi_lstm = nn.LSTM(
            input_size=self.base_model.model.config.hidden_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.shared_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )

        # POS Task params
        self.pos_net = nn.Sequential(
            nn.Linear(32, len(self.hparams.label2id) + 1),
            nn.LayerNorm(len(self.hparams.label2id) + 1),
        )

        self.pos_crf = CRF(
            num_tags=len(self.hparams.label2id) + 1,
            batch_first=True
        )

        # LID Task params
        self.lid_net = nn.Sequential(
            nn.Linear(32, len(self.hparams.lid2id) + 1),
            nn.LayerNorm(len(self.hparams.lid2id) + 1)
        )

        self.lid_crf = CRF(
            num_tags=len(self.hparams.lid2id) + 1,
            batch_first=True
        )

        self.weighted_loss = MultiTaskLossWrapper(num_tasks=2)  # LID and POS: two tasks

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        base_model_outs = self.base_model(
            input_ids,
            attention_mask
        )

        base_outs = base_model_outs.last_hidden_state
        lstm_outs, _ = self.bi_lstm(base_outs)
        shared_net_outs = self.shared_net(lstm_outs)

        # POS
        pos_net_outs = self.pos_net(shared_net_outs)

        # LID
        lid_net_outs = self.lid_net(shared_net_outs)

        return pos_net_outs, lid_net_outs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        lids = batch['lids']

        pos_emissions, lid_emissions = self(input_ids, attention_mask)

        pos_loss = -self.pos_crf(pos_emissions, labels, attention_mask.bool())
        lid_loss = -self.lid_crf(lid_emissions, lids, attention_mask.bool())

        pos_path = self.pos_crf.decode(pos_emissions)
        pos_path = torch.tensor(pos_path, device=self.device).long()

        lid_path = self.lid_crf.decode(lid_emissions)
        lid_path = torch.tensor(lid_path, device=self.device).long()

        # Weighted Loss
        loss = self.weighted_loss(pos_loss, lid_loss)

        pos_metrics = self._compute_metrics(pos_path, labels, "train", "pos")
        lid_metrics = self._compute_metrics(lid_path, lids, "train", "lid")

        self.log("loss/train", loss)
        self.log("loss-pos/train", pos_loss)
        self.log("loss-lid/train", lid_loss)

        self.log_dict(pos_metrics, on_step=False, on_epoch=True)
        self.log_dict(lid_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        lids = batch['lids']

        pos_emissions, lid_emissions = self(input_ids, attention_mask)

        pos_loss = -self.pos_crf(pos_emissions, labels, attention_mask.bool())
        lid_loss = -self.lid_crf(lid_emissions, lids, attention_mask.bool())

        pos_path = self.pos_crf.decode(pos_emissions)
        pos_path = torch.tensor(pos_path, device=self.device).long()

        lid_path = self.lid_crf.decode(lid_emissions)
        lid_path = torch.tensor(lid_path, device=self.device).long()

        loss = pos_loss + lid_loss
        pos_metrics = self._compute_metrics(pos_path, labels, "val", "pos")
        lid_metrics = self._compute_metrics(lid_path, lids, "val", "lid")

        self.log("loss/val", loss)
        self.log("loss-pos/val", pos_loss)
        self.log("loss-lid/val", lid_loss)

        self.log_dict(pos_metrics, on_step=False, on_epoch=True)
        self.log_dict(lid_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        lids = batch['lids']

        pos_emissions, lid_emissions = self(input_ids, attention_mask)

        pos_loss = -self.pos_crf(pos_emissions, labels, attention_mask.bool())
        lid_loss = -self.lid_crf(lid_emissions, lids, attention_mask.bool())

        pos_path = self.pos_crf.decode(pos_emissions)
        pos_path = torch.tensor(pos_path, device=self.device).long()

        lid_path = self.lid_crf.decode(lid_emissions)
        lid_path = torch.tensor(lid_path, device=self.device).long()

        loss = pos_loss + lid_loss
        pos_metrics = self._compute_metrics(pos_path, labels, "test", "pos")
        lid_metrics = self._compute_metrics(lid_path, lids, "test", "lid")

        self.log("loss/test", loss)
        self.log("loss-pos/test", pos_loss)
        self.log("loss-lid/test", lid_loss)

        # Ensure metrics are properly logged
        if pos_metrics:
            self.log_dict(pos_metrics)
        if lid_metrics:
            self.log_dict(lid_metrics)

    
    def configure_optimizers(self):
        # Parameters without specific lr or weight_decay will use global settings
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                'params': [
                    p
                    for n, p in self.bi_lstm.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
            },
            {
                'params': [
                    p
                    for n, p in self.shared_net.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
            },
            {
                'params': [
                    p
                    for n, p in self.pos_net.named_parameters()  # Updated from ner_net to pos_net
                    if not any(nd in n for nd in no_decay)
                ],
                'lr': self.hparams.pos_learning_rate,  # Updated parameter
                'weight_decay': self.hparams.pos_wd  # Updated parameter
            },
            {
                'params': [
                    p
                    for n, p in self.lid_net.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'lr': self.hparams.lid_learning_rate,
                'weight_decay': self.hparams.lid_wd
            },
            {
                'params': [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]

        if self.hparams.freeze != "freeze":
            optimizer_grouped_parameters.append({
                'params': [
                    p
                    for n, p in self.base_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
            })

        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.warm_restart_epochs
        )

        return [optimizer], [lr_scheduler]
    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor, mode: str, task: str):
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        metrics = {}

        if task == "pos":
            metrics[f"prec/{mode}-{task}"] = precision(
                preds, targets, 
                average="macro", 
                num_classes=len(self.hparams.label2id) + 1, 
                ignore_index=self.pos_pad_token_label,  # Updated from ner_pad_token_label
                task="multiclass"
            )
            
            metrics[f"rec/{mode}-{task}"] = recall(
                preds, targets, 
                average="macro", 
                num_classes=len(self.hparams.label2id) + 1,
                ignore_index=self.pos_pad_token_label,  # Updated from ner_pad_token_label
                task="multiclass"
            )

            metrics[f"f1/{mode}-{task}"] = f1_score(
                preds, targets, 
                average="macro", 
                num_classes=len(self.hparams.label2id) + 1,
                ignore_index=self.pos_pad_token_label,  # Updated from ner_pad_token_label
                task="multiclass"
            )

        elif task == "lid":
            metrics[f"prec/{mode}-{task}"] = precision(
                preds, targets, 
                average="macro", 
                num_classes=len(self.hparams.lid2id) + 1, 
                ignore_index=self.lid_pad_token_label,
                task="multiclass"
            )
            metrics[f"rec/{mode}-{task}"] = recall(
                preds, targets, 
                average="macro", 
                num_classes=len(self.hparams.lid2id) + 1, 
                ignore_index=self.lid_pad_token_label,
                task="multiclass"
            )

            metrics[f"f1/{mode}-{task}"] = f1_score(
                preds, targets, 
                average="macro", 
                num_classes=len(self.hparams.lid2id) + 1, 
                ignore_index=self.lid_pad_token_label,
                task="multiclass"
            )

        return metrics
