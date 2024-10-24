export HF_HOME=YOUR_HF_HOME
export NCCL_P2P_DISABLE=1 # for FSDP training
export OMP_NUM_THREADS=12

DS_PERCENTAGE=$1 # percentage of the data that is selected using BM25.
number_of_gpus=$2
DATA_SEED=$3
VALIDATION_TASK=$4
include_validation=$5
pipeline_parallel=$6
batch_size_per_device=$7
data_shuffle=$8
learning_rate=$9
num_train_epochs=${10}
save_steps_per_epoch=${11}
accelerate_launch=${12} #* use accelerate launch for multi-gpu training
method=${13} #* method for selecting the data (top-k, mid-k)
model_name=${14}
out_dir=${15}
deepspeed_config=${16}
reference_number=${17}

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

PERCENTAGE=1.0 # percentage of the data to select from the selected data for training.
DS_METHOD=bm25
JOB_NAME="${model_name}-{$VALIDATION_TASK}-{$DS_METHOD}-p${DS_PERCENTAGE}-lora-seed${DATA_SEED}-lr$learning_rate-${num_train_epochs}epoch-$number_of_gpus"

TRAIN_FILES="data/selected_data/${VALIDATION_TASK}/${VALIDATION_TASK}-train-p${PERCENTAGE}-${DS_METHOD}.jsonl"

./ccds/training/train_scripts/train/generic_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$DS_PERCENTAGE" \
"$DATA_SEED" "$JOB_NAME" "$VALIDATION_TASK" \
"$include_validation" "$number_of_gpus" "$TRAIN_FILES" \
"$batch_size_per_device" "$data_shuffle" "$learning_rate" \
"$num_train_epochs" "$save_steps_per_epoch"  "$accelerate_launch" \
"$out_dir" "$deepspeed_config" 