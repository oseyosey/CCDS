#!/bin/bash
export HF_HOME="YOUR_HF_HOME"
export OMP_NUM_THREADS=12

# argument parsing
model_path=$1

data_path="YOUR_EVALUATION_DATA_PATH"

source ../evaluation/eval_mmlu.sh $model_path $data_path

# run main evaluation function
eval_mmlu
extract_mmlu