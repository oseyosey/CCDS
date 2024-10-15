export HF_HOME=YOUR_HF_HOME
export NCCL_P2P_DISABLE=1 # for FSDP training‘
export OMP_NUM_THREADS=12

PERCENTAGE=$1 # percentage of the full data to train, you can specify the training file you want to use in the script
number_of_gpus=$2
DATA_SEED=$3
VALIDATION_TASK=$4
include_validation=$5
pipeline_parallel=$6
batch_size_per_device=$7
METHOD=$8
data_shuffle=$9
learning_rate=${10}
num_train_epochs=${11}
save_steps_per_epoch=${12}
val_only=${13}
model_name=${14}
accelerate_launch=${15} #* use accelerate launch for multi-gpu training
out_dir=${16}
deepspeed_config=${17}

DATA_DIR=data

if [[ "$model_name" == "llama2-13b" ]]; then
    MODEL_PATH=meta-llama/Llama-2-13b-hf
elif [[ "$model_name" == "llama2-7b" ]]; then
    MODEL_PATH=meta-llama/Llama-2-7b-hf
elif [[ "$model_name" == "llama2-70b" ]]; then
    MODEL_PATH=meta-llama/Llama-2-70b-hf
elif [[ "$model_name" == "llama3-8b" ]]; then
    MODEL_PATH=meta-llama/Meta-Llama-3-8B
else
    echo "Model name not recognized"
    exit 1
fi

DS_PERCENTAGE=1.0 # percentage of the data that is selected using BM25.

if [[ "$val_only" == "true" ]]; then
    JOB_NAME="${model_name}-{$VALIDATION_TASK}-${num_train_epochs}-val-only"
else
    JOB_NAME="${model_name}-{$VALIDATION_TASK}-{$METHOD}-p${PERCENTAGE}-lora-seed${DATA_SEED}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
fi


if [[ "$val_only" == "true" ]]; then
    ./ccds/training/train_scripts/train/full_data_lora_train_val_only.sh \
    "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" \
    "$VALIDATION_TASK" "$include_validation" "$number_of_gpus" \
    "$batch_size_per_device" "$data_shuffle" "$learning_rate" \
    "$num_train_epochs" "$save_steps_per_epoch" "$accelerate_launch" \
    "$out_dir" "$deepspeed_config" #if disabled, use FSDP for DP.
else
    ./ccds/training/train_scripts/train/full_data_lora_train.sh \
    "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" \
    "$JOB_NAME" "$VALIDATION_TASK" "$include_validation" \
    "$number_of_gpus" "$batch_size_per_device" "$data_shuffle" \
    "$learning_rate" "$num_train_epochs" "$save_steps_per_epoch" \
    "$accelerate_launch" "$out_dir" "$deepspeed_config" #if disabled, use FSDP for DP.
fi

