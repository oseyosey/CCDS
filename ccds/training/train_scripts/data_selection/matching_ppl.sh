#!/bin/bash
train_file_names=$1
target_task_names=$2
output_path=$3
model_dir=$4
epochs=$5

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

data_dir=data
train_files=("$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
    "$data_dir/train/processed/cot/cot_data.jsonl"
    "$data_dir/train/processed/dolly/dolly_data.jsonl"
"$data_dir/train/processed/oasst1/oasst1_data.jsonl")

# python3 -m ccds.training.data_selection.ppl.prepare_training_dataset \
# --train_files ${train_files[@]} 2>&1 \
# --output_path $output_path

python3 -m ccds.training.data_selection.ppl.matching_ppl \
--model_dir $model_dir \
--train_files ${train_files[@]} 2>&1 \
--train_file_names $train_file_names \
--target_task_names $target_task_names \
--output_path $output_path \
--epochs_num $epochs
