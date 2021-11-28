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
from torch.nn import Parameter
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

from torch_geometric.nn import Sequential, HeteroConv, GINConv, GCNConv, SAGEConv, GATConv

PROJ_PATH = Path(os.path.join(re.sub("/BERT_ABSA.*$", '', os.getcwd()), 'BERT_ABSA'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))
import utils
from utils import *
from attention import *

class SentimentClassifier(pl.LightningModule):
    '''
    Bert-based sentiment classifier
    '''
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

class SynSentimentClassifier(pl.LightningModule):
    '''
    Bert-based sentiment classifier with Syntactic-aware from Dependency Tree
    '''
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        # Bert
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_bert_name)
        self.bert = BertModel.from_pretrained(
            self.hparams.pretrained_bert_name, output_hidden_states=True, output_attentions=True, return_dict=False)
        self.bert_size = self.bert.config.hidden_size
        self.hidden_size =  self.hparams.hidden_size
        
        # GNN
        heads = 9
        self.convs = Sequential('x, edge_index', [
            (GATConv(self.bert_size, self.hidden_size, heads=heads), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(heads * self.hidden_size, self.hidden_size, heads=heads), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(heads * self.hidden_size, self.hidden_size, heads=1), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            ])
#         self.convs = torch.nn.ModuleList()
#         heads = 9
#         self.convs.append(GATConv(self.bert_size, self.hidden_size, heads=heads))
#         for _ in range(self.n_layers - 2):
#             self.convs.append(GATConv(heads * self.hidden_size, self.hidden_size, heads=heads))
#         self.convs.append(GATConv(heads * self.hidden_size, self.hidden_size, heads=1))
        
        # Attention
        self.attn_vector = Parameter(torch.zeros((self.hidden_size, 1), dtype=torch.float), requires_grad=True)   
        nn.init.xavier_uniform_(self.attn_vector)
        self.attention = AdditiveAttention(self.hidden_size, self.hidden_size)
        
        # 
        self.lin =  nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=self.hparams.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, 3)
        
        # Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer  
    
    def forward(self, input_ids, attention_mask, token_type_ids, edge_index, labels):
        bsize = input_ids.shape[0]
        bert_features = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
        )[0]
        
        features = torch.reshape(
            bert_features, (bert_features.shape[0]*bert_features.shape[1], bert_features.shape[2]))
        
        h1 = self.convs(features, edge_index)
        batch_graph_shape = (bert_features.shape[0], bert_features.shape[1], h1.shape[1])
        h2 = torch.reshape(h1, batch_graph_shape)
        mask = copy.deepcopy(attention_mask).reshape(-1).tile(self.hidden_size, 1).T.reshape(batch_graph_shape)
        
        h3 = mask * h2
        graph_attn = self.attn_vector.squeeze().unsqueeze(0).repeat(bsize, 1)
        attn_weights = self.attention(graph_attn, h3) 
        h4 = weighted_sum(h3, attn_weights)
        graph_embedding = F.relu(self.lin(h4))
        graph_embedding = self.dropout(graph_embedding)
        logits = self.classifier(graph_embedding)
        return logits
    
    def margin_loss(self,  embedding_query, embedding_pos, embedding_neg):
        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1)
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        return scores_pos - scores_neg
        
    def training_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            edge_index=batch['edge_index'],
            labels=batch['label'],
        )
        
        labels = batch['label']
        ce_loss = self.cross_entropy_loss(logits, labels)        
        return ce_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            edge_index=batch['edge_index'],
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
            edge_index=batch['edge_index'],
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
     