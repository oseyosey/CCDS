#!/bin/bash
export HF_HOME=YOUR_HF_HOME

target_task=$1

# target_task=mmlu
# model_dir="meta-llama/Llama-2-7b-hf"

train_file_names="flan_v2 cot dolly oasst1"
target_task_names=$target_task
selected_data_output_path="data/selected_data"
select_percentage=1.0
epochs=25 #* how many epochs the model are trained

model_dir="" # finetuned PPL model directory

./ccds/training/train_scripts/data_selection/matching_ppl.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$model_dir"  "$epochs"

#TOP PPL 
for data_seed in 3 6 9 12 15; do
    ./ccds/training/train_scripts/data_selection/select_data_ppl_top.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage" "$epochs" "$data_seed"
done

# MID PPL 
for data_seed in 3 6 9 12 15; do
    ./ccds/training/train_scripts/data_selection/select_data_ppl_mid.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage" "$epochs" "$data_seed"
done

