import argparse
import os
import re
import string
import numpy as np
from tqdm import tqdm
import json

import torch

from training.data_selection.get_training_dataset import load_raw_dataset
from training.data_selection.get_validation_dataset import get_raw_val_dataset

argparser = argparse.ArgumentParser(
    description='Script for preparing the datasets for Perpelxity Based Selection (PBS)')
argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')

args = argparser.parse_args()
output_path = args.output_path

# load in training datasets
raw_datasets = load_raw_dataset(
        args.train_files, sample_percentage=1.0)

raw_datasets_dict = raw_datasets.to_dict() # have to first turn it into dict to be efficient

raw_messages = []
for idx in tqdm(range(len(raw_datasets['messages']))):
    message = raw_datasets_dict['messages'][idx][0]['content']+ " A:" + raw_datasets_dict['messages'][idx][1]['content']
    raw_messages.append(message)

output_path_files = f"{output_path}/raw_datasets_messages.jsonl"
json.dump(raw_messages, open(output_path_files, 'w'))
print("Saved raw datasets messages to {}".format(output_path_files))