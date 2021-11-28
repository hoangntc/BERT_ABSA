import os, sys, subprocess, re, time
from sys import stderr
from subprocess import Popen
from collections import defaultdict
import math
import csv
import pandas as pd
import numpy as np
import json


def run_cmd(cmd_str):
    return subprocess.check_output(cmd_str.split(' '))