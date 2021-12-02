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
from sklearn.metrics import roc_auc_score

PROJ_PATH = Path(os.path.join(re.sub("/BERT_ABSA.*$", '', os.getcwd()), 'BERT_ABSA'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))
from utils import *
from dataset import DataModule
from model import SentimentClassifier, SynSentimentClassifier, SynSemSentimentClassifier
from main import load_model_test

class InferenceAgent:
    def __init__(self,
                 proj_path,
                 model_name='syn',
                 ckpt_filename='epoch=4-val_loss=0.6445-val_acc=0.8003-val_macro_f1=0.7335-val_micro_f1=0.8003.ckpt',
                 ckpt_dirname='model/laptops',
                 hparams_filename='../src/config/laptop_config.json',
                 device='cpu'):
        
        hparams = read_json(hparams_filename)
        
        self.proj_path = Path(proj_path)
        assert self.proj_path.is_dir(), 'proj_path must be an existing directory'

        self.checkpoint_path = self.proj_path / ckpt_dirname / ckpt_filename
        
        assert self.checkpoint_path.is_file(), f'checkpoint_path={self.checkpoint_path} must be an existing file'
        print(f'Load model: {self.checkpoint_path}')
        
        # data
        self.data = DataModule(hparams['data_params'])
        self.polarity_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.transformation = hparams['data_params']['transformation']
        self.max_length = hparams['data_params']['max_length']
        self.bert_tokenizer = BertTokenizer.from_pretrained(hparams['data_params']['bert_name'])
        
        # model
        if torch.cuda.is_available() and device != 'cpu':
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = 'cpu'
            
#         self.model = SentimentClassifier.load_from_checkpoint(
#             checkpoint_path=str(self.checkpoint_path), map_location=map_location)
        self.model_name = model_name
        self.model = load_model_test(
            self.model_name, checkpoint_path=str(self.checkpoint_path), map_location=self.map_location)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def transform(self, sample):
        seq1 = sample['text'].lower()
        term = sample['term'].lower()
        
        if self.transformation == 'QA_M':
            seq2 = f'what is the polarity of {term} ?'
        elif self.transformation == 'MLI_M':
            seq2 = term.lower()
        elif self.transformation == 'KW_M':
            seq2 = term
            
        if 'label' in sample:
            label = self.polarity_dict[sample['label']]
        else:
            label = 0
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
    
#     def infer_single_instance(self, sample):
#         '''
#         Sample format: {'text': 'Food is good', 'term': 'food', 'label': 'positive'}
#         '''
#         seq1, seq2, label = self.transform(sample)
#         encoded_text = self.encode_text(seq1, seq2)
#         softmax = nn.Softmax(dim=1)
#         logits = self.model(
#             encoded_text['input_ids'],
#             encoded_text['attention_mask'],
#             encoded_text['token_type_ids'],
#             torch.tensor([label]))
#         probas = softmax(logits).squeeze().detach().numpy()
#         return probas

    def predict(self, sample):
        softmax = nn.Softmax(dim=1)
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        token_type_ids = sample['token_type_ids']
        edge_index = sample['edge_index'] if 'edge_index' in sample else None
        label = sample['label']
        
        if self.model_name == 'bert':
            prediction = softmax(self.model(
                sample['input_ids'].to(self.device), 
                sample['attention_mask'].to(self.device),
                sample['token_type_ids'].to(self.device), 
                sample['label'].to(self.device),
            ))
        elif self.model_name == 'syn' or self.model_name == 'synsem':
            prediction = softmax(self.model(
                sample['input_ids'].to(self.device), 
                sample['attention_mask'].to(self.device),
                sample['token_type_ids'].to(self.device), 
                sample['edge_index'].to(self.device),
                sample['label'].to(self.device),
            ))
        return prediction.detach().cpu().numpy().tolist()
    
    def get_aux_information(self, sample, dataset='train'):
        ids = sample['id']
        text = sample['text']
        label = sample['label'].detach().cpu().numpy().tolist()
        term = sample['term']
        tvt = len(sample['term']) * [dataset] # batch_size * ['train'/'val'/'test']
        return ids, text, label, term, tvt
    
    def infer_dataset(self):
        self.data.setup()
        
        ids = []
        text = []
        term = []
        label = []
        prediction = []
        tvt = []
        
        print('Inferring train ...')
        for sample in self.data.mytrain_dataloader():
            # aux information
            aux = self.get_aux_information(sample, 'train')
            ids += aux[0]
            text += aux[1]
            label += aux[2]
            term += aux[3]
            tvt += aux[4]
            # predict
            prediction += self.predict(sample)
        
        print('Inferring validation ...')
        for sample in self.data.val_dataloader():
            # aux information
            aux = self.get_aux_information(sample, 'val')
            ids += aux[0]
            text += aux[1]
            label += aux[2]
            term += aux[3]
            tvt += aux[4]
            # predict
            prediction += self.predict(sample)
        
        print('Inferring test ...')    
        for sample in self.data.test_dataloader():
            # aux information
            aux = self.get_aux_information(sample, 'test')
            ids += aux[0]
            text += aux[1]
            label += aux[2]
            term += aux[3]
            tvt += aux[4]
            # predict
            prediction += self.predict(sample)
        
        return ids, text, term, label, tvt, prediction
    
    def get_prediction(self):
        out = self.infer_dataset()
        df = pd.DataFrame({
            'id': out[0], 'text': out[1], 'term': out[2], 'label_id': out[3], 'tvt': out[4], 'pred': out[5]})
        df['label'] = df['label_id'].map({v:k for k,v in self.polarity_dict.items()})
        df[['pred_0', 'pred_1', 'pred_2']] = pd.DataFrame(df.pred.tolist(), index= df.index)
        return df[['id', 'text', 'term', 'label_id', 'label', 'tvt', 'pred_0', 'pred_1', 'pred_2']]

def main():
    device = 'cpu'
    # Restaurant
    ## Bert
    restaurant_agent = InferenceAgent(
        proj_path=str(PROJ_PATH),
        model_name='bert',
        ckpt_filename='model=bert-epoch=7-val_loss=0.8514-val_acc=0.7893-val_auc=0.9102-val_macro_f1=0.7312-val_micro_f1=0.7893.ckpt',
        ckpt_dirname='model/restaurants',
        hparams_filename='../src/config/bert_restaurant_config.json',
        device=device,
    )
    save_path = '../output/bert_restaurant.csv'
    print(f'Save to {save_path}')
    pred_res = restaurant_agent.get_prediction()
    pred_res.to_csv(save_path, index=False)
    
    ## Syn
    restaurant_agent = InferenceAgent(
        proj_path=str(PROJ_PATH),
        model_name='syn',
        ckpt_filename='model=syn-epoch=5-val_loss=0.8056-val_acc=0.7803-val_auc=0.9062-val_macro_f1=0.7209-val_micro_f1=0.7803.ckpt',
        ckpt_dirname='model/restaurants',
        hparams_filename='../src/config/syn_restaurant_config.json',
        device=device,
    )
    save_path = '../output/syn_restaurant.csv'
    print(f'Save to {save_path}')
    pred_res = restaurant_agent.get_prediction()
    pred_res.to_csv(save_path, index=False)
    
#     ## Sem
#     restaurant_agent = InferenceAgent(
#         proj_path=str(PROJ_PATH),
#         model_name='synsem',
#         ckpt_filename='epoch=4-val_loss=0.6445-val_acc=0.8003-val_macro_f1=0.7335-val_micro_f1=0.8003.ckpt',
#         ckpt_dirname='model/restaurants',
#         hparams_filename='../src/config/sem_restaurant_config.json',
#         device='cpu',
#     )
#     pred_res = restaurant_agent.get_prediction()
#     pred_res.to_csv('../output/synsem_restaurant.csv', index=False)
    
    # Laptop
    ## Bert
    laptop_agent = InferenceAgent(
        proj_path=str(PROJ_PATH), 
        model_name='bert',
        ckpt_filename='model=bert-epoch=4-val_loss=0.6608-val_acc=0.7861-val_auc=0.9221-val_macro_f1=0.7576-val_micro_f1=0.7861.ckpt',
        ckpt_dirname='model/laptops',
        hparams_filename='../src/config/bert_laptop_config.json',
        device=device,
    )
    save_path = '../output/bert_laptop.csv'
    print(f'Save to {save_path}')
    pred_res = laptop_agent.get_prediction()
    pred_res.to_csv(save_path, index=False)
    
#     ## Syn
#     laptop_agent = InferenceAgent(
#         proj_path=str(PROJ_PATH), 
#         model_name='syn',
#         ckpt_filename='model=syn-epoch=3-val_loss=0.6911-val_acc=0.7665-val_auc=0.9204-val_macro_f1=0.7420-val_micro_f1=0.7665.ckpt',
#         ckpt_dirname='model/laptops',
#         hparams_filename='../src/config/syn_laptop_config.json',
#         device=device,
#     )
#     save_path = '../output/syn_laptop.csv'
#     print(f'Save to {save_path}')
#     pred_res = laptop_agent.get_prediction()
#     pred_res.to_csv(save_path, index=False)
    
#     ## SynSem
#     laptop_agent = InferenceAgent(
#         proj_path=str(PROJ_PATH), 
#         model_name='synsem',
#         ckpt_filename='epoch=4-val_loss=0.6445-val_acc=0.8003-val_macro_f1=0.7335-val_micro_f1=0.8003.ckpt',
#         ckpt_dirname='model/laptops',
#         hparams_filename='../src/config/sem_laptop_config.json',
#         device=device,
#     )
#     save_path = '../output/sem_laptop.csv'
#     print(f'Save to {save_path}')
#     pred_res = laptop_agent.get_prediction()
#     pred_res.to_csv(save_path, index=False)
        
if __name__ == "__main__":
    main()