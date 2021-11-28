import os, sys, re, datetime, random, gzip, json
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from itertools import accumulate
from collections import OrderedDict
import argparse
from time import time
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel

from multiprocessing import Pool
from sentence_transformers import SentenceTransformer, models, losses, InputExample

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import MedianStoppingRule

PROJ_PATH = Path(os.path.join(re.sub("/BERT_ABSA.*$", '', os.getcwd()), 'BERT_ABSA'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))
import utils
from utils import read_json

from dataset import DataModule
from model import SentimentClassifier

def build_model(config, model_name):
    data_params, model_params = config['data_params'], config['model_params']
    data = DataModule(data_params)
    if model_name == 'bert':
        model = SentimentClassifier(model_params)
    elif model_name == 'syn':
        model = SynSentimentClassifier(model_params)
    return data, model

def build_trainer(config, phase=None):
    trainer_params = config['trainer_params']
    data_params = config['data_params']
    
    # callbacks
    checkpoint = ModelCheckpoint(
        dirpath=trainer_params['checkpoint_dir'], 
        filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}-{val_macro_f1:.4f}-{val_micro_f1:.4f}',
        save_top_k=trainer_params['top_k'],
        verbose=True,
        monitor=trainer_params['metric'],
        mode=trainer_params['mode'],
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.00, 
        patience=trainer_params['patience'],
        verbose=False,
        mode=trainer_params['mode'],
    )
    metrics = {'loss': 'val_loss', 'acc': 'val_acc', 'macro_f1': 'val_macro_f1', 'micro_f1': 'val_micro_f1'}
    tuner = TuneReportCallback(metrics, on='validation_end')
    
    if phase == 'tune':
        callbacks = [tuner]
    else:
        callbacks = [checkpoint, early_stopping]
    
    # trainer_kwargs
    trainer_kwargs = {
        'max_epochs': trainer_params['max_epochs'],
        'gpus': 1 if torch.cuda.is_available() else 0,
    #     "progress_bar_refresh_rate": p_refresh,
    #     'gradient_clip_val': hyperparameters['grad_clip'],
        'weights_summary': 'full',
        'deterministic': True,
        'callbacks': callbacks,
    }

    trainer = Trainer(**trainer_kwargs)
    return trainer, trainer_kwargs

def load_model_test(model_name, checkpoint):
    if model_name == 'bert':
        return SentimentClassifier.load_from_checkpoint(checkpoint) 
    elif model_name == 'syn':
    return SynSentimentClassifier.load_from_checkpoint(checkpoint) 

def execute(args, phase='train'):
    config = args.config
    assert phase in ['train', 'test'], 'Invalid phase!'
    seed_everything(config['data_params']['seed'], workers=True)
    data, clf = build_model(config, args.model_name)
    trainer, trainer_kwargs = build_trainer(config, phase=phase)

    if phase == 'train':
        trainer.fit(clf, data)
    elif phase == 'test':
        checkpoint_dir = Path(config['trainer_params']['checkpoint_dir'])
        print(f'Load checkpoint from: {str(checkpoint_dir)}')
        paths = sorted(checkpoint_dir.glob('*.ckpt'))
        for p in paths:
            print(p)
            model_test = load_model_test(args.model_name, p)
            result = trainer.test(model_test, datamodule=data)
            
            del model_test
    
    del data
    del clf 
    del trainer
    
def main():
    parser = argparse.ArgumentParser(description='Training.')

    parser.add_argument('-config_file', help='config file path', default='../src/config/restaurant_config.json', type=str)
    parser.add_argument('-model_name', help='model name', default='SYN', type=str)
    parser.add_argument('-f', '--fff', help='a dummy argument to fool ipython', default='1')
    args = parser.parse_args()
    args.config = read_json(args.config_file)
    
    if args.config['trainer_params']['train']:
        print('Starting training...')
        execute(args, phase='train')
        
    if args.config['trainer_params']['test']:
        print('Starting testing...')
        execute(args, phase='test')
        
if __name__ == "__main__":
    main()