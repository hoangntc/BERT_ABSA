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

import os
import json
import pandas as pd
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parseXML(data_path):
    tree = ET.ElementTree(file=data_path)
    objs = list()
    for sentence in tree.getroot():
        obj = dict()
        obj['id'] = sentence.attrib['id']
        for item in sentence:
            if item.tag == 'text':
                obj['text'] = item.text
            elif item.tag == 'aspectTerms':
                obj['aspects'] = list()
                for aspectTerm in item:
                    if aspectTerm.attrib['polarity'] != 'conflict':
                        obj['aspects'].append(aspectTerm.attrib)
            elif item.tag == 'aspectCategories':
                obj['category'] = list()
                for category in item:
                    obj['category'].append(category.attrib)
        if 'aspects' in obj and len(obj['aspects']):
            objs.append(obj)
    return objs

def convert_to_dataframe(objs):
    output = []
    for sentence in objs:
        id = sentence['id']
        text = sentence['text']
        aspects = sentence['aspects']
        for aspect in aspects:
            term = aspect['term']
            label = aspect['polarity']
            output.append([id, text, term, label])
    output = sorted(output, key=lambda x: x[0])
    df = pd.DataFrame(output, columns=['id', 'text', 'term', 'label'])
    return df


dataset_files = {
    'restaurant': {
        'train': 'Restaurants_Train.xml',
        'test': 'Restaurants_Test.xml',
        'trial': 'Restaurants_Trial.xml'
    },
    'laptop': {
        'train': 'Laptops_Train.xml',
        'test': 'Laptops_Test.xml',
        'trial': 'Laptops_Trial.xml'
    }
}

for dsname, fnames in dataset_files.items():
    for g, fname in fnames.items():
        input_path = str(PROJ_PATH/ 'dataset/raw_data' / fname)
        output_path01 = str(PROJ_PATH/ 'dataset/preprocessed_data' / fname.replace('.xml', '.pkl'))
        output_path02 = str(PROJ_PATH/ 'dataset/preprocessed_data' / fname.replace('.xml', '.csv'))
        print(f'Load: {input_path}')
        print(f'Save: {output_path01}\n')
        objs = parseXML(input_path)
        df = convert_to_dataframe(objs)
        pd.to_pickle(objs, output_path01)
        df.to_csv(output_path02, index=False)