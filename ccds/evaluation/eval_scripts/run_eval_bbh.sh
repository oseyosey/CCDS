#!/bin/bash
export HF_HOME="YOUR_HF_HOME"
export OMP_NUM_THREADS=12

# argument parsing
model_path=$1
#output_dir=$2 * for base model evaluation

data_path="YOUR_EVALUATION_DATA_PATH"

source ../evaluation/eval_bbh.sh $model_path $data_path #$output_dir

# run main evaluation function
eval_bbh
extract_bbh
