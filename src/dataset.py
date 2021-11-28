import os, sys, re, datetime, random, gzip, json, copy
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import accumulate
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel

from time import time
from math import ceil
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer, models, losses, InputExample

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from torch_geometric.utils import dense_to_sparse, to_dense_adj

PROJ_PATH = Path(os.path.join(re.sub("/BERT_ABSA.*$", '', os.getcwd()), 'BERT_ABSA'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))


class Dataset(Dataset):
    def __init__(self, data_dir, transformation='KW_M', num_classes=3, bert_tokenizer=None, max_length=0, seed=0):
        random.seed(seed)
        assert transformation in ['QA_M', 'MLI_M', 'KW_M'], 'Invalid transformation method'
        assert num_classes in [2, 3], 'Invalid num_classes'
        
        self.transformation = transformation
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length
        self.polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        # load data
        self.data = pd.read_pickle(data_dir)
    
        if num_classes == 2:
            self.data = [d for d in self.data if d['label'] != 'neutral']
    
    def transform(self, sample):
        # Transform input text to 
        seq1 = sample['text'].lower()
        term = sample['term'].lower()
        
        if self.transformation == 'QA_M':
            seq2 = f'what is the polarity of {term} ?'
            label = self.polarity_dict[sample['label']]
        elif self.transformation == 'MLI_M':
            seq2 = term.lower()
            label = self.polarity_dict[sample['label']]
        elif self.transformation == 'KW_M':
            seq2 = term
            label = self.polarity_dict[sample['label']]
        
        return seq1, seq2, label
        
    def encode_text(self, seq1, seq2):
        # Encode text for BERT model
        encoded_text = self.bert_tokenizer.encode_plus(
            seq1,
            seq2,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_length,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            truncation=True, # Truncate up to maximum length
            return_attention_mask=True,  # Generate the attention mask
            return_tensors='pt',  # Ask the function to return PyTorch tensors
        )
        return encoded_text

    def padding_adj(self, adj):
        # Padding adj to fixed size
        pad_size = self.max_length - adj.shape[0]
        return np.pad(adj, [(0, pad_size), (0, pad_size)], 'constant')
    
    def get_adj_of_dependency_graph(self, edge_index):
        edge_reindex = torch.tensor(self.reindex_edge_index(edge_index))
        dense_adj = to_dense_adj(edge_reindex).squeeze().numpy()
        return self.padding_adj(dense_adj)
    
    def reindex_edge_index(self, edge_index):
        '''
        Reindex for special token id in BERT
        '''
        edge_reindex = [
            [i+1 for i in edge_index[0]], 
            [i+1 for i in edge_index[1]],
        ]
        return edge_reindex
    
    def __getitem__(self, item):
        '''
        sample = {
        'id': '813', 
        'text': 'All the appetizers and salads were fabulous, \\
        the steak was mouth watering and the pasta was delicious!!!', 
        'term': 'appetizers', 
        'from': '8', 
        'to': '18', 
        'source_dep': [2, 2, 2, 2, 2, 2, 9, 2, 2, 6, 6, 2, 2, 2, 6, 9, 12, 15, 15, 15, 9, 20, 18, 20, 20, 9, 9], 
        'target_dep': [0, 3, 4, 1, 3, 4, 2, 3, 4, 5, 7, 6, 3, 4, 7, 8, 11, 12, 13, 14, 15, 16, 11, 18, 19, 20, 21], 
        'edge_type': ['dep', 'sprwrd', 'sprwrd', 'det', 'sprwrd', 'sprwrd', 'nsubj',\\
        'sprwrd', 'sprwrd', 'cc', 'sprwrd', 'conj', 'sprwrd', 'sprwrd', 'sprwrd', 'cop',\\
        'det', 'nsubj', 'cop', 'compound', 'ccomp', 'cc', 'det', 'nsubj', 'cop', 'conj', 'punct'],
        'label': 'positive'}
        '''
            
        # BERT Encoder
        sample = self.data[item]
        seq1, seq2, label = self.transform(sample)
        encoded_text = self.encode_text(seq1, seq2)
        edge_index = [sample['source_dep'], sample['target_dep']]
        edge_reindex = self.reindex_edge_index(edge_index)
        
        single_input = {
            'id': sample['id'],
            'text': sample['text'],
            'seq1': seq1,
            'seq2': seq2,
            'term': sample['term'],
            'label': label, 
            'input_ids': encoded_text['input_ids'].flatten(),
            'token_type_ids': encoded_text['token_type_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
            'edge_index': edge_reindex, 
        }
        return single_input

    def __len__(self):
        return len(self.data)
    
class DataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

    def setup(self, stage=None):
        bert_tokenizer = BertTokenizer.from_pretrained(self.hparams.bert_name)
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            data_fit = Dataset(
                data_dir=self.hparams.data_train_dir,
                transformation=self.hparams.transformation,
                num_classes=self.hparams.num_classes,
                bert_tokenizer=bert_tokenizer,
                max_length=self.hparams.max_length,
                seed=self.hparams.seed)
            
            total_samples = data_fit.__len__()
            train_samples = int(data_fit.__len__() * 0.8)
            val_samples = total_samples - train_samples
            self.data_train, self.data_val = random_split(
                data_fit, [train_samples, val_samples], generator=torch.Generator().manual_seed(self.hparams.seed))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = Dataset(
                data_dir=self.hparams.data_test_dir,
                transformation=self.hparams.transformation,
                num_classes=self.hparams.num_classes,
                bert_tokenizer=bert_tokenizer,
                max_length=self.hparams.max_length,
                seed=self.hparams.seed)
    
    def reindex_node_by_graph(self, nodes, graph_order):
        new_nodes = [n + graph_order*self.hparams.max_length for n in nodes]
        return new_nodes
    
    def collate_fn(self, batch):        
        label = torch.tensor([i['label'] for i in batch], dtype=torch.long)
        input_ids = torch.stack([i['input_ids'] for i in batch])
        token_type_ids = torch.stack([i['token_type_ids'] for i in batch])
        attention_mask = torch.stack([i['attention_mask'] for i in batch])
        source = [self.reindex_node_by_graph(j['edge_index'][0], i) for i,j in enumerate(batch)]
        target = [self.reindex_node_by_graph(j['edge_index'][1], i) for i,j in enumerate(batch)]
        edge_index = torch.tensor([
            [item for sublist in source for item in sublist], 
            [item for sublist in target for item in sublist], 
        ], dtype=torch.long)
        adj = to_dense_adj(edge_index).squeeze() 
        return {
            'label': label,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'edge_index': edge_index,
            'adj': adj,
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=4, 
            shuffle=False, # Already shuffle in random_split() 
            drop_last=True, 
            collate_fn=self.collate_fn,
        )
    
    def mytrain_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=4, 
            shuffle=False, # Already shuffle in random_split() 
            drop_last=False, 
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=4, 
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            num_workers=4, 
            shuffle=False,
            collate_fn=self.collate_fn,
        )