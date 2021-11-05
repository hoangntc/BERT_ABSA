import os, sys, re, datetime, random, gzip, json
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import accumulate
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
import torch
import torch.nn as nn

def calc_f1(logits, labels, avg_type='macro', multilabel_binarizer=None):
    '''
    Calculates the F1 score (either macro or micro as defined by 'avg_type') for the specified logits and labelss
    '''
    pred = torch.argmax(logits, dim=-1) #get predictions by finding the indices with max logits
    score = f1_score(labels.cpu().detach(), pred.cpu().detach(), average=avg_type)
    return torch.tensor([score])

def calc_accuracy(logits, labels):
    '''
    Calculates the accuracy for the specified logits and labels
    '''
    pred = torch.argmax(logits, 1) #get predictions by finding the indices with max logits
    acc = accuracy_score(labels.cpu().detach(), pred.cpu().detach())
    return torch.tensor([acc])