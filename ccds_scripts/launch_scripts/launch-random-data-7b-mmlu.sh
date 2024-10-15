nvidia-smi

model_name=llama2-7b
percentage=0.05 # {0.025, 0.05, 0.1, 0.25, 0.50, 1.0}%
number_of_gpus=4
data_seed=3 # Seed for shuffling data: 3, 6, 9, ...
validation_task=mmlu
include_validation=false
pipeline_parallel=false
batch_size_per_device=1
method=random
data_shuffle=true 
learning_rate=2e-05
num_train_epochs=4
save_steps_per_epoch=1
val_only=false
accelerate_launch=false
out_dir=out_7b_mmlu
deepspeed_config=zero1


for learning_rate in 4e-05; do
    for percentage in 0.025 0.05; do
        for data_seed in 3; do

            #* Training 
            export CUDA_VISIBLE_DEVICES=0,1,2,3

            cd YOUR_REPO_PATH && \
            conda run -p YOUR_CONDA_ENV \
            bash ccds_scripts/train_random.sh "$percentage" "$number_of_gpus" "$data_seed" "$validation_task" \
            "$include_validation" "$pipeline_parallel" "$batch_size_per_device" \
            "$method" "$data_shuffle" "$learning_rate" "$num_train_epochs" \
            "$save_steps_per_epoch" "$val_only" "$model_name" "$accelerate_launch" "$out_dir" \
            "$deepspeed_config"

            #* Evaluation
            export CUDA_VISIBLE_DEVICES=0

            model_path="YOUR_REPO_PATH/${out_dir}/${model_name}-{$validation_task}-{$method}-p${percentage}-lora-seed${data_seed}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
            cd YOUR_REPO_PATH/ccds/evaluation && \
            conda run -p YOUR_CONDA_ENV \
            bash eval_scripts/run_eval_mmlu.sh $model_path
        
        done
    done
done


for learning_rate in 2e-05; do
    for percentage in 0.10 0.25 0.50; do
        for data_seed in 3; do

            #* Training 
            export CUDA_VISIBLE_DEVICES=0,1,2,3

            cd YOUR_REPO_PATH && \
            conda run -p YOUR_CONDA_ENV \
            bash ccds_scripts/train_random.sh "$percentage" "$number_of_gpus" "$data_seed" "$validation_task" \
            "$include_validation" "$pipeline_parallel" "$batch_size_per_device" \
            "$method" "$data_shuffle" "$learning_rate" "$num_train_epochs" \
            "$save_steps_per_epoch" "$val_only" "$model_name" "$accelerate_launch" "$out_dir" \
            "$deepspeed_config"

            #* Evaluation
            export CUDA_VISIBLE_DEVICES=0

            model_path="YOUR_REPO_PATH/${out_dir}/${model_name}-{$validation_task}-{$method}-p${percentage}-lora-seed${data_seed}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
            cd YOUR_REPO_PATH/ccds/evaluation && \
            conda run -p YOUR_CONDA_ENV \
            bash eval_scripts/run_eval_mmlu.sh $model_path
        
        done
    done
done