import argparse
import os

import torch

from ccds.training.data_selection.get_training_dataset import load_raw_dataset
from datasets import Dataset

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str, # this could be just a scoring from different data selection methods
                           nargs='+', help='The path to the score file')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')

    args = argparser.parse_args()

    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


if __name__ == "__main__":
    args = parse_args()
    assert len(args.train_file_names) == len(args.train_files)
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    lm_datasets = load_raw_dataset(args.train_files, sample_percentage=1.0)
    lm_datasets_dict = lm_datasets.to_dict()
    
    # for key in lm_datasets_dict.keys():
    #     lm_datasets_dict[key].extend(val_datasets_dict[key])

    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        score_paths = os.path.join(output_path, f"{target_task}_bm25_score.pt") 
        
        bm25_scores = torch.load(score_paths, map_location=device)
        bm25_scores_tensor = torch.from_numpy(bm25_scores)

        total_samples = bm25_scores.shape[0]

        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
            print(f"Selecting {args.max_samples} samples")
        else:
            data_amount_name = f"num{args.max_samples}"

        # sort the scores and output the corresponding data index
        topk_scores, topk_indices = torch.topk(bm25_scores_tensor, args.max_samples, largest=True)

        # Create a subset of lm_datasets based on the topk_indices
        selected_lm_datasets_dict = {key: [lm_datasets_dict[key][i] for i in topk_indices] for key in lm_datasets_dict.keys()}

        selected_lm_datasets = Dataset.from_dict(selected_lm_datasets_dict)

        # Save in JSON Lines format
        selected_lm_datasets.to_json(f"{output_path}/{target_task}-train-p{args.percentage}-bm25.jsonl")
        