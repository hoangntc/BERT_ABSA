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

PROJ_PATH = Path(os.path.join(re.sub("/BERT_ABSA.*$", '', os.getcwd()), 'BERT_ABSA'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))


class Dataset(Dataset):
    def __init__(self, data_dir, transformation='QA_M', num_classes=3, bert_tokenizer=None, max_length=0, seed=0):
        random.seed(seed)
        assert transformation in ['QA_M', 'QA_B', 'MLI_M', 'MLI_B'], 'Invalid transformation method'
        assert num_classes in [2, 3], 'Invalid num_classes'
        
        self.transformation = transformation
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length
        self.polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        # load data
        self.data = list(pd.read_csv(data_dir).T.to_dict().values())
        if num_classes == 2:
            self.data = [d for d in self.data if d['label'] != 'neutral']
    
    def transform(self, sample):
        seq1 = sample['text'].lower()
        term = sample['term'].lower()
        
        if self.transformation == 'QA_M':
            seq2 = f'what is the polarity of {term} ?'
            label = self.polarity_dict[sample['label']]
        elif self.transformation == 'MLI_M':
            seq2 = term.lower()
            label = self.polarity_dict[sample['label']]
#         elif self.transformation == 'QA_B':
#         elif self.transformation == 'MLI_B':
        
        return seq1, seq2, label
        
    def encode_text(self, seq1, seq2):
        # encode
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
        
    def __getitem__(self, item):
        '''
        example = {
            'id': 1000,
            'text': 'The food is good, especially their more basic dishes, and the drinks are delicious.',
            'term': 'food',
            'label': 'positive',
            }
        '''
            
        # encoder
        sample = self.data[item]
        seq1, seq2, label = self.transform(sample)
        encoded_text = self.encode_text(seq1, seq2)

        single_input = {
            'seq1': seq1,
            'seq2': seq2,
            'term': sample['term'],
            'label': label, 
            'input_ids': encoded_text['input_ids'].flatten(),
            'token_type_ids': encoded_text['token_type_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
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

    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=4, 
            shuffle=False, # Already shuffle in random_split() 
            drop_last=True, 
#             collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=4, 
            shuffle=False,
#             drop_last=True, 
#             collate_fn=lambda x: x,
        )
    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            num_workers=4, 
            shuffle=False,
#             drop_last=True, 
#             collate_fn=lambda x: x,
        )