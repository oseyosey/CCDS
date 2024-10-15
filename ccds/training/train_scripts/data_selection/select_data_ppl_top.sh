#!/bin/bash

train_file_names=$1
target_task_names=$2
output_path=$3
percentage=$4
epochs=$5
data_seed=$6

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

data_dir=data
train_files=("$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
    "$data_dir/train/processed/cot/cot_data.jsonl"
    "$data_dir/train/processed/dolly/dolly_data.jsonl"
"$data_dir/train/processed/oasst1/oasst1_data.jsonl")

python3 -m training.data_selection.ppl.write_selected_data_ppl_top \
--target_task_names ${target_task_names} \
--train_file_names ${train_file_names} \
--train_files ${train_files[@]} 2>&1 \
--output_path $output_path \
--percentage $percentage \
--epochs_num $epochs \
--data_seed $data_seed