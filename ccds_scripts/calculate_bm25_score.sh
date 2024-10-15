#!/bin/bash
export HF_HOME=YOUR_HF_HOME

train_file_names="flan_v2 cot dolly oasst1"
target_task_names="mmlu" # mmlu, bbh, ifeval
selected_data_output_path="data/selected_data"
select_percentage=1.0

./ccds/training/train_scripts/data_selection/matching_bm25.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" 

./ccds/training/train_scripts/data_selection/select_data_bm25.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage"

