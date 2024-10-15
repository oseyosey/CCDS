import argparse
import os
import torch
import random  # Import the random module for shuffling
from training.data_selection.get_training_dataset import load_raw_dataset
from datasets import Dataset

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str, 
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
    argparser.add_argument('--epochs_num', type=int, 
                           help='The number of epochs for training the model')
    argparser.add_argument('--data_seed', type=int, default=42, help='Random seed for shuffling the data')

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
    
    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        score_paths = os.path.join(output_path, f"{target_task}_ppl_score_epoch{args.epochs_num}.pt") 
        
        ppl_scores = torch.load(score_paths, map_location=device)
        ppl_scores_tensor = torch.from_numpy(ppl_scores)

        total_samples = ppl_scores.shape[0]

        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
            print(f"Selecting {args.max_samples} samples")
        else:
            data_amount_name = f"num{args.max_samples}"

        # Step 1: Select top-k scores based on PPL (smallest PPL scores)
        topk_scores, topk_indices = torch.topk(ppl_scores_tensor, args.max_samples, largest=False) 

        # Step 2: Use topk_indices to create the initial subset of the dataset
        selected_lm_datasets_dict = {key: [lm_datasets_dict[key][i] for i in topk_indices] for key in lm_datasets_dict.keys()}

        # At this point, selected_lm_datasets_dict is already sorted from smallest to largest PPL.

        # Step 3: Reorder the already selected data based on percentiles
        total_selected = len(topk_indices)

        # Calculate the 33.3% and 66.6% indices
        percentile_33_idx = int(total_selected * 0.333)
        percentile_66_idx = int(total_selected * 0.666)

        # Split into first third, middle third, and bottom third
        first_third_indices = list(range(0, percentile_33_idx))  # 0% - 33.3%
        middle_third_indices = list(range(percentile_33_idx, percentile_66_idx))  # 33.3% - 66.6%
        bottom_third_indices = list(range(percentile_66_idx, total_selected))  # 66.6% - 100%

        # Step 4: Shuffle the indices with a random seed
        random.seed(args.data_seed)  # Set random seed for reproducibility
        random.shuffle(middle_third_indices)  # Shuffle the middle third
        random.shuffle(first_third_indices)   # Shuffle the first third
        random.shuffle(bottom_third_indices)  # Shuffle the bottom third

        # Step 5: Reorder: middle third first, then first third, then bottom third
        reordered_indices = middle_third_indices + first_third_indices + bottom_third_indices

        # Step 6: Apply the reordering to the selected_lm_datasets_dict
        reordered_lm_datasets_dict = {key: [selected_lm_datasets_dict[key][i] for i in reordered_indices] for key in selected_lm_datasets_dict.keys()}

        # Step 7: Convert the reordered data to the appropriate format and save
        selected_lm_datasets = Dataset.from_dict(reordered_lm_datasets_dict)
        selected_lm_datasets.to_json(f"{output_path}/{target_task}-train-p{args.percentage}-ppl-epoch{args.epochs_num}-mid-ppl-{args.data_seed}.jsonl")

