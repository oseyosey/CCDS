import argparse
import os
import re
import string
import numpy as np

import torch
from rank_bm25 import BM25Okapi

from training.data_selection.get_training_dataset import load_raw_dataset
from training.data_selection.get_validation_dataset import get_raw_val_dataset

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training w. BM25 (Word Frequency)')
argparser.add_argument('--data_dir', type=str, nargs='+',
                           help='"The path to the data')
argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks (BBH, TYDIQA, MMLU)")
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')


args = argparser.parse_args()

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_text(text):
    text = text.lower()

    text = text.replace('\n', ' ')
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# arg parse
# target_task_names = args.target_task_names[0]
data_dir = args.data_dir[0]

# load in training datasets
raw_datasets = load_raw_dataset(
        args.train_files, sample_percentage=1.0)

# feed training datasets to BM25
messages_corpus = [msg[0]['content'] for msg in raw_datasets['messages']]
messages_corpus_processed = [preprocess_text(doc) for doc in messages_corpus]
tokenized_corpus = [doc.split(" ") for doc in messages_corpus_processed]
bm25 = BM25Okapi(tokenized_corpus)


# calculate the BM25 score for each validation task (sub-stack)
for target_task_name in args.target_task_names:
    # load in validation datasets for each task
    raw_val_datasets = get_raw_val_dataset(task_name=target_task_name, 
                                        data_dir=data_dir)
    raw_val_datasets_processed = [preprocess_text(doc) for sub_task_name, doc in raw_val_datasets.items()]

    # 1. loop through each subtask and query the task with the BM25 model
    # 2. obtain the BM25 score for each subtask
    # 3. create a score/mapping to map back
    # 4. save the score/mapping to a file
    bm25_sub_task_scores = []
    for task in raw_val_datasets_processed:
        tokenized_sub_task = task.split(" ")
        sub_task_scores = bm25.get_scores(tokenized_sub_task)
        bm25_sub_task_scores.append(sub_task_scores)

    bm25_scores = np.mean(bm25_sub_task_scores, axis=0)
    output_dir = os.path.join(args.output_path, target_task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(
        args.output_path, target_task_name, f"{target_task_name}_bm25_score_100.pt")
    torch.save(bm25_scores, output_file)
    print("Saved bm25 score to {}".format(output_file))
