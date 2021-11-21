import os, sys, re, datetime, random, gzip, json
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import accumulate
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from time import time
from math import ceil
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer, models, losses, InputExample

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

PROJ_PATH = Path(os.path.join(re.sub("/BERT_ABSA.*$", '', os.getcwd()), 'BERT_ABSA'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))
import utils

class SentimentClassifier(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_bert_name)
        self.bert = BertForSequenceClassification.from_pretrained(
            self.hparams.pretrained_bert_name, num_labels=3, output_hidden_states=True, output_attentions=True, return_dict=False)
        self.hidden_size = self.bert.config.hidden_size
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer  
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        loss, logits, hidden, _ = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            labels=labels,
        )
        
        return logits
    
    def margin_loss(self,  embedding_query, embedding_pos, embedding_neg):
        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1)
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        return scores_pos - scores_neg
        
    def training_step(self, batch, batch_idx):
        # ['seq1', 'seq2', 'term', 'label', 'input_ids', 'token_type_ids', 'attention_mask']
        logits = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['label'],
        )
        
        labels = batch['label']
        ce_loss = self.cross_entropy_loss(logits, labels)        
#         acc = utils.calc_accuracy(logits, labels).squeeze()
#         logs = {
#             'loss': ce_loss,
#             'acc': acc,
#         }
#         self.log_dict(logs, prog_bar=True)
        return ce_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['label'],
        )
        
        labels = batch['label']
        ce_loss = self.cross_entropy_loss(logits, labels)        
        acc = utils.calc_accuracy(logits, labels).squeeze()
        macro_f1 = utils.calc_f1(logits, labels, avg_type='macro').squeeze()
        micro_f1 = utils.calc_f1(logits, labels, avg_type='micro').squeeze()

        logs = {
            'loss': ce_loss, 
            'acc': acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
        }
        self.log_dict(logs, prog_bar=True)
        return logs
    
    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in val_step_outputs]).mean().cpu()
        avg_macro_f1 = torch.stack([x['macro_f1'] for x in val_step_outputs]).mean().cpu()
        avg_micro_f1 = torch.stack([x['micro_f1'] for x in val_step_outputs]).mean().cpu()
        logs = {
            'val_loss': avg_loss, 
            'val_acc': avg_acc,
            'val_macro_f1': avg_macro_f1,
            'val_micro_f1': avg_micro_f1,
        }
        self.log_dict(logs, prog_bar=True)
     
    def test_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['label'],
        )
        
        labels = batch['label']
        ce_loss = self.cross_entropy_loss(logits, labels)        
        acc = utils.calc_accuracy(logits, labels).squeeze()
        macro_f1 = utils.calc_f1(logits, labels, avg_type='macro').squeeze()
        micro_f1 = utils.calc_f1(logits, labels, avg_type='micro').squeeze()

        logs = {
            'loss': ce_loss, 
            'acc': acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
        }
        return logs
    
    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in test_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in test_step_outputs]).mean().cpu()
        avg_macro_f1 = torch.stack([x['macro_f1'] for x in test_step_outputs]).mean().cpu()
        avg_micro_f1 = torch.stack([x['micro_f1'] for x in test_step_outputs]).mean().cpu()
        logs = {
            'test_loss': avg_loss, 
            'test_acc': avg_acc,
            'test_macro_f1': avg_macro_f1,
            'test_micro_f1': avg_micro_f1,
        }
        self.log_dict(logs, prog_bar=True)
        return logs