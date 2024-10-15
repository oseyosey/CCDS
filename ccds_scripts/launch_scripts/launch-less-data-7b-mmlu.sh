nvidia-smi

model_name=llama2-7b
number_of_gpus=4
validation_task=mmlu
include_validation=false
pipeline_parallel=false
batch_size_per_device=1
data_shuffle=false 
num_train_epochs=1
save_steps_per_epoch=1 
use_accelerate_launch=true
out_dir=out_7b_mmlu
deepspeed_config=zero1


learning_rate=4e-05
for less_seed in 3; do
  for ds_percentage in 0.025 0.05; do
    data_seed=$less_seed

    #* Training
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    cd YOUR_REPO_PATH && \
    conda run -p YOUR_CONDA_ENV \
    bash ccds_scripts/train_less.sh \ 
    "$ds_percentage" "$number_of_gpus" "$data_seed" "$validation_task" "$include_validation" "$pipeline_parallel" \
    "$batch_size_per_device" "$data_shuffle" "$learning_rate" "$num_train_epochs" "$save_steps_per_epoch" \
    "$less_seed" "$model_name" "$use_accelerate_launch" "$out_dir" "$deepspeed_config"

    #* Evaluation
    export CUDA_VISIBLE_DEVICES=0

    model_path="YOUR_REPO_PATH/${out_dir}/${model_name}-{$validation_task}-{less}-p${ds_percentage}-lora-seed${less_seed}-$data_seed-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
    cd YOUR_REPO_PATH/ccds/evaluation && \
    conda run -p YOUR_CONDA_ENV \
    bash eval_scripts/run_eval_mmlu.sh $model_path

  done
done


learning_rate=2e-05
for less_seed in 3; do
  for ds_percentage in 0.10 0.25 0.50 1.0; do
    data_seed=$less_seed

    #* Training
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    cd YOUR_REPO_PATH && \
    conda run -p YOUR_CONDA_ENV \
    bash ccds_scripts/train_less.sh \ 
    "$ds_percentage" "$number_of_gpus" "$data_seed" "$validation_task" "$include_validation" "$pipeline_parallel" \
    "$batch_size_per_device" "$data_shuffle" "$learning_rate" "$num_train_epochs" "$save_steps_per_epoch" \
    "$less_seed" "$model_name" "$use_accelerate_launch" "$out_dir" "$deepspeed_config"

    #* Evaluation
    export CUDA_VISIBLE_DEVICES=0

    model_path="YOUR_REPO_PATH/${out_dir}/${model_name}-{$validation_task}-{less}-p${ds_percentage}-lora-seed${less_seed}-$data_seed-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
    cd YOUR_REPO_PATH/ccds/evaluation && \
    conda run -p YOUR_CONDA_ENV \
    bash eval_scripts/run_eval_mmlu.sh $model_path

  done
done