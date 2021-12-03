import os, sys, re, datetime, random, gzip, json
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from itertools import accumulate
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import MedianStoppingRule, ASHAScheduler

PROJ_PATH = Path(os.path.join(re.sub("/BERT_ABSA.*$", '', os.getcwd()), 'BERT_ABSA'))
print(f'PROJ_PATH={PROJ_PATH}')
sys.path.insert(1, str(PROJ_PATH))
sys.path.insert(1, str(PROJ_PATH/'src'))
import utils

from dataset import DataModule
from model import SentimentClassifier, SynSentimentClassifier, SynSemSentimentClassifier

import commentjson
from collections import OrderedDict
from main import read_json, build_model, build_trainer

os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
# MODEL_NAME = 'bert'
MODEL_NAME = 'syn'

def tuning(config):
    seed_everything(config['data_params']['seed'], workers=True)
    data, clf = build_model(config, model_name=MODEL_NAME)
    trainer, trainer_kwargs = build_trainer(config, model_name=MODEL_NAME, phase='tune')
    trainer.fit(clf, data)
        
def get_config(experiment_name):
    if experiment_name == 'restaurant':
        config = {
            "data_params": {
                "data_train_dir": str(PROJ_PATH / "dataset/preprocessed_data/Restaurants_Train_data.pkl"),
                "data_test_dir": "../dataset/preprocessed_data/Restaurants_Test_data.pkl",
                "transformation": 'KW_M', # tune.choice(['QA_M', 'MLI_M', 'KW_M']),
                "num_classes": 3,
                "batch_size": 128,
                "bert_name": "bert-base-uncased",
                "max_length": 100,
                "seed": 12345,
            },

            "model_params": {
                "pretrained_bert_name": "bert-base-uncased",
                "hidden_size": 256,
                "hidden_dropout_prob": tune.choice([0.05, 0.1, 0.3, 0.5]),
                "lr": tune.loguniform(1e-4, 1e-1),
            },
    
            "trainer_params": {
                "checkpoint_dir": "../model/restaurants",
                "top_k": 3,
                "max_epochs": 100,
                "metric": "val_auc",
                "patience": 20,
                "mode": "max",
            }
        }
            
    if experiment_name == 'laptop':
        config = {
            "data_params": {
                "data_train_dir": str(PROJ_PATH / "dataset/preprocessed_data/Laptops_Train_data.pkl"),
                "data_test_dir": "../dataset/preprocessed_data/Laptops_Test_data.pkl",
                "transformation": 'KW_M', # tune.choice(['QA_M', 'MLI_M', 'KW_M']),
                "num_classes": 3,
                "batch_size": 128,
                "bert_name": "bert-base-uncased",
                "max_length": 128,
                "seed": 12345,
            },

            "model_params": {
                "pretrained_bert_name": "bert-base-uncased",
                "hidden_size": 256,
                "hidden_dropout_prob": tune.choice([0.05, 0.1, 0.3, 0.5]),
                "lr": tune.loguniform(1e-4, 1e-1),
            },
    
            "trainer_params": {
                "checkpoint_dir": "../model/laptops",
                "top_k": 3,
                "max_epochs": 100,
                "metric": "val_auc",
                "patience": 20,
                "mode": "max",
            }
        }
    return config

def main():
    parser = argparse.ArgumentParser(description='Tuning.')

#     parser.add_argument('-config_file', help='config file path', default='../src/config/restaurant_config.json', type=str)
    parser.add_argument('-e', '--experiment', default='restaurant', type=str)
    args = parser.parse_args()
    experiment_name = args.experiment
    experiment_dir = str(PROJ_PATH / 'experiment' / experiment_name / MODEL_NAME )
    n_experiment = 100
    
    if not os.path.exists(experiment_dir): os.mkdir(experiment_dir)

    print('Starting tuning...')    
    scheduler = ASHAScheduler(
        metric='acc',
        mode='max',
        max_t=50,
        grace_period=1,
        reduction_factor=2)
    config = get_config(experiment_name)
    analysis = tune.run(
        tune.with_parameters(tuning),
        config=config,
        resources_per_trial={'gpu': 1},
        local_dir=experiment_dir,
        scheduler=scheduler,
        num_samples=n_experiment)
        
if __name__ == "__main__":
    main()