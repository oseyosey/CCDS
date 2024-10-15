#!/bin/bash

### TRAINING SCRIPT FOR RANDOM DATA SELECTION ###

source training/train_scripts/train/base_training_args.sh

data_dir=$1
model_path=$2
percentage=$3
data_seed=$4
job_name=$5
validation_task=$6
include_validation=$7
number_of_gpus=$8
batch_size_per_device=$9
data_shuffle=${10}
learning_rate=${11}
num_train_epochs=${12}
save_steps_per_epoch=${13}
accelerate_launch=${14}
out_dir=${15}
deepspeed_config=${16}

ID=$(printf "%05d" $((10000 + RANDOM % 90000)))
if [[ $accelerate_launch == "true" ]]; then
    echo "Using accelerate launch"
    header="accelerate launch --config_file YOUR_DEEPSPEED_CONFIG_PATH/multi_gpu${number_of_gpus}_${deepspeed_config}.yaml --main_process_port=$ID training/train/train.py"
else
    header="torchrun --master_port=$ID --nproc_per_node $number_of_gpus --nnodes 1 \
    -m training.train.train"
fi


output_dir="../${out_dir}/${job_name}"
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_files=("$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
    "$data_dir/train/processed/cot/cot_data.jsonl"
    "$data_dir/train/processed/dolly/dolly_data.jsonl"
"$data_dir/train/processed/oasst1/oasst1_data.jsonl")

# use fsdp for large models (if not deepspeed)
if [[ $accelerate_launch == "false" ]]; then
    if [[ $number_of_gpus != "1" ]]; then
        if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
            base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
            elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
            base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
            elif [[ $model_path == "meta-llama/Llama-2-7b-hf" ]]; then
            base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_7b_finetune"
        fi
    fi
fi

training_args="$base_training_args \
--num_train_epochs $num_train_epochs \
--save_steps_per_epoch $save_steps_per_epoch \
--per_device_train_batch_size $batch_size_per_device \
--data_dir $data_dir \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--validation_task_name $validation_task \
--analysis_dataset $validation_task \
--include_validation $include_validation \
--learning_rate $learning_rate \
--data_shuffle $data_shuffle \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args" 