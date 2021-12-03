import os, sys, re, datetime, random, gzip, json
from tqdm import tqdm
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
# from stanfordcorenlp import StanfordCoreNLP
import json
from nltk.parse.corenlp import CoreNLPDependencyParser
# nlps = StanfordCoreNLP(str(PROJ_PATH / 'misc/stanford-corenlp-4.3.2'))
to_strip_chars = ".,\(\)"
host = '9000'
BERT_MODEL = 'bert-base-uncased'

################################################################################################
# XML PARSER
################################################################################################
def parseXML(data_path):
    '''
    Parse XML file to dictionary
    '''
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

################################################################################################
# DEPENDENCY PARSER
################################################################################################
def process_bert_tokens(tokens, DEBUG=False):
    
    to_strip_chars = ".,\(\)"
    token_check_list_1 = ["s", "re", "m", "ve", "ll", "d"]
    
    tokens += [""]
    
    current_token_group = []
    output = []
    last_token = None
    for token_idx, token in enumerate(tokens):
        
        next_token = None
        if token_idx + 1 < len(tokens):
            next_token = tokens[token_idx + 1]

        reset = True
        if token == "'":
            reset = False
        elif token.startswith("##"):
            reset = False
        elif last_token is not None:
            if token in token_check_list_1:
                reset = False

        keep = True
        if (token == '.') and (next_token is None):
            keep = True
        elif (token == '.') and (next_token is not None) and (next_token != '.'):
            keep = True
        else:
            token = token.strip(to_strip_chars)
            if token == "":
                keep = False

        if reset:
            if len(current_token_group) > 0:
                output += [current_token_group]
            current_token_group = []
        if keep:
            if DEBUG:
                current_token_group += [token]
            else:
                current_token_group += [token_idx]
            last_token = token
            
    return output

def process_core_nlp_tokens(tokens, DEBUG=False):
    
    to_strip_chars = ".,\(\)"
    
    output = []
    for token_idx, token in enumerate(tokens):
        
        point_stripped_token = token.strip(".")
        if point_stripped_token == "":
            if DEBUG:
                output += ["."]
            else: 
                output += [token_idx]
        elif token.startswith("'"):
            if DEBUG:
                output[-1] += token
        else:
            isTokenEndsWithPoint = token.endswith(".")
            
            token = token.strip(to_strip_chars)
            if token != "":
                if DEBUG:
                    output += [token]
                    if isTokenEndsWithPoint:
                        output += [""]
                else: 
                    output += [token_idx]
                    if isTokenEndsWithPoint:
                        output += [-1]

    return output

# def tokenize_and_depparse(text):
#     text+=' '
#     text = re.sub(r'\. ',' . ',text).strip()
#     text = re.sub(r' {2,}',' ',text)
#     nlp_properties = {
#         'annotators': 'depparse',
# #         'tokenize.options': 'splitHyphenated=false,normalizeParentheses=false',
#         'tokenize.whitespace': True,  # all tokens have been tokenized before
#         'ssplit.isOneSentence': False,
#         'outputFormat': 'json',
#     }
    
#     try:
#         parsed = json.loads(nlps.annotate(text.strip(), nlp_properties))
#     except:
#         print('Error')
        
#     parsed = parsed['sentences']
#     tokens = []
#     tokens_dict = {}
#     tuples = []
#     tmplen = 0
#     for item in parsed:
#         for ite in item['tokens']:
#             tokens.extend([ite['word']])
#             tokens_dict[ite['index']] = ite['word']
# #         tokens.extend([ite['word'] for ite in item['tokens']])
#         tuples.extend([
#             (
#                 ite['dep'],
#                 ite['governor']-1+tmplen,
#                 ite['dependent']-1+tmplen
#             ) for ite in item['basicDependencies'] if ite['dep']!='ROOT'
#         ])
#         tmplen=len(tokens)
        
#     return tokens, tuples

def tokenize_and_depparse(text):
    '''
    Parse dependency tree by CoreNLP
    
    # to_conll(10) will return the result in a format as follows:
    # id word lemma ctag tag feats head(head's id) rel(syntactic relation)
    # return values that is unknown will be shown as '_'
    # tag and ctag are considered to be equal
    '''
    parser = CoreNLPDependencyParser(url=f'http://localhost:{host}')
    dep_parsed_sentence = parser.raw_parse(text)
    deps = dep_parsed_sentence.__next__()
    
    lines = deps.to_conll(10).split('\n')
    tokens = []
    tuples = []
    for line in lines:
        if line != '':
            result = line.split('\t')
            # id word lemma ctag tag feats head(head's id) rel(syntactic relation)
            tokens.append(result[1])
            if result[7] != 'ROOT':
                tuples.append((result[7], int(result[6])-1 , int(result[0])-1))   
    return tokens, tuples

def build_bert_token_whole(bert_token):
    result = ""
    for bert_sub_token in bert_token:
        if bert_sub_token.startswith("##"):
            result += bert_sub_token[2:]
        else:
            result += bert_sub_token
    return result

def map_corenlp_to_bert_from_indexes(corenlp_processed_indexes, bert_processed_indexes):
    output = {}
    for (corenlp_processed_index, bert_processed_index) in zip(corenlp_processed_indexes, bert_processed_indexes):
        output[corenlp_processed_index] = bert_processed_index
    return output

def map_corenlp_to_bert_from_indexes_2(corenlp_tokens, bert_tokens, corenlp_processed_indexes, bert_processed_indexes):

    output = {}
    
    bert_run_idx_global = 0
    for corenlp_idx in corenlp_processed_indexes:
        for bert_run_idx, bert_idx_group in enumerate(bert_processed_indexes[bert_run_idx_global:]):
        
            corenlp_token = corenlp_tokens[corenlp_idx]

            bert_token_group = map(lambda bert_idx: bert_tokens[bert_idx], bert_idx_group)
            bert_token = build_bert_token_whole(bert_token_group)

            lower_corenlp_token = corenlp_token.lower()
            lower_bert_token = bert_token.lower()
            if lower_corenlp_token.startswith(lower_bert_token) or lower_corenlp_token.endswith(lower_bert_token) or lower_bert_token.startswith(lower_corenlp_token) or lower_bert_token.endswith(lower_corenlp_token):
                bert_run_idx_global = bert_run_idx + 1
                output[corenlp_idx] = bert_idx_group
                break;
        
    return output
        
def map_corenlp_to_bert(corenlp_tokens, bert_tokens, DEBUG=False):
    corenlp_processed_indexes = process_core_nlp_tokens(corenlp_tokens, DEBUG)
    bert_processed_indexes = process_bert_tokens(bert_tokens, DEBUG)
#     return map_corenlp_to_bert_from_indexes(corenlp_processed_indexes, bert_processed_indexes)
    return map_corenlp_to_bert_from_indexes_2(corenlp_tokens, bert_tokens, corenlp_processed_indexes, bert_processed_indexes)


def clean_text(text):
    cleaned_text = text.lower().strip()
    return cleaned_text

def build_dep_parse_tree(text, verbose=False):
    '''
    Parse dependency tree and map CoreNLP index to BERT index
    
    Returns
    -------
        output_bert_v1s: list of source node indexes
        output_bert_v2s: list of target node indexes
        types: list of dependency relation
        
    Usage:
    ----------
        build_dep_parse_tree("I'm waiting ... It's 9am now.", True)
    '''
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    bert_tokens = bert_tokenizer.tokenize(clean_text(text))
    (corenlp_tokens, parse_tree_corenlp) = tokenize_and_depparse(text)
    corenlp_to_bert_map = map_corenlp_to_bert(corenlp_tokens, bert_tokens)
    
    if verbose:
        print(f'BERT tokens: {bert_tokens}')
        print(f'CoreNLP tokens: {corenlp_tokens}')
        print(f'CoreNLP dependency tree: {parse_tree_corenlp}')
        print(f'CoreNLP to BERT: {corenlp_to_bert_map}')
        
    output_bert_v1s = []
    output_bert_v2s = []
    types = []
    for edge in parse_tree_corenlp:
        (t, corenlp_v1, corenlp_v2) = edge
        
        if (corenlp_v1 not in corenlp_to_bert_map) or (corenlp_v2 not in corenlp_to_bert_map):
            continue;
        
        bert_v1 = corenlp_to_bert_map[corenlp_v1]
        bert_v2 = corenlp_to_bert_map[corenlp_v2]

        if (len(bert_v1) > 0) and (len(bert_v2) > 0):
            bert_v1_super = bert_v1[0]
            bert_v2_super = bert_v2[0]
            
            output_bert_v1s.append(bert_v1_super)
            output_bert_v2s.append(bert_v2_super)
            types.append(t)
            
            for bert_v1_sub in bert_v1[1:]:
                output_bert_v1s.append(bert_v1_super)
                output_bert_v2s.append(bert_v1_sub)
                types.append("sprwrd")
                
            for bert_v2_sub in bert_v2[1:]:
                output_bert_v1s.append(bert_v2_super)
                output_bert_v2s.append(bert_v2_sub)
                types.append("sprwrd")
                
    return output_bert_v1s, output_bert_v2s, types
    
################################################################################################

def parse_data_sample(objs):
    '''
    Parse data into samples for training/testing
    '''
    output = []
    for sentence in tqdm(objs, total=len(objs)):
        aspects = sentence['aspects']
        for aspect in aspects:
            source_dep, target_dep, edge_type = build_dep_parse_tree(sentence['text'])
            output.append({
                'id': sentence['id'],
                'text': sentence['text'],
                'term': aspect['term'],
                'from': aspect['from'],
                'to': aspect['to'],
                'source_dep': source_dep,
                'target_dep': target_dep,
                'edge_type': edge_type,
                'label': aspect['polarity'],
            })
    return output

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
        output_path02 = str(PROJ_PATH/ 'dataset/preprocessed_data' / fname.replace('.xml', '_data.pkl'))
        print(f'Load: {input_path}')
        print(f'Save parsed XML to: {output_path01}')
        print(f'Save preprocessed data to: {output_path02}\n')
        objs = parseXML(input_path)
        output = parse_data_sample(objs)
        pd.to_pickle(objs, output_path01)
        pd.to_pickle(output, output_path02)