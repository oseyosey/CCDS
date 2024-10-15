nvidia-smi

model_name=llama2-7b
ds_percentage=0.05 # {0.025, 0.05, 0.1, 0.25, 0.50, 1.0}%
number_of_gpus=2
data_seed=3 # Seed for shuffling data after data selection: 3, 6, 9, ...
validation_task=mmlu
include_validation=false
pipeline_parallel=false
batch_size_per_device=1
data_shuffle=false 
learning_rate=4e-05
num_train_epochs=1
save_steps_per_epoch=1 
accelerate_launch=true
method=bm25
out_dir=out_7b_mmlu
deepspeed_config=zero1


for learning_rate in 4e-05; do
    for ds_percentage in 0.025 0.05; do
        for data_seed in 3; do

        #* Training 
        export CUDA_VISIBLE_DEVICES=0,1,2,3

        cd YOUR_REPO_PATH && \
        conda run -p YOUR_CONDA_ENV \
        bash ccds_scripts/train_bm25.sh \
        "$ds_percentage" "$number_of_gpus" "$data_seed" "$validation_task" \
        "$include_validation" "$pipeline_parallel" "$batch_size_per_device" "$data_shuffle" \
        "$learning_rate" "$num_train_epochs" "$save_steps_per_epoch" "$accelerate_launch" \
        "$method" "$model_name" "$out_dir" "$deepspeed_config"
        
        #* Evaluation
        export CUDA_VISIBLE_DEVICES=0

        model_path="YOUR_REPO_PATH/${out_dir}/${model_name}-{$validation_task}-{bm25}-p${ds_percentage}-lora-seed${data_seed}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
        cd YOUR_REPO_PATH/ccds/evaluation && \
        conda run -p YOUR_CONDA_ENV \
        bash eval_scripts/run_eval_mmlu.sh $model_path

        done
    done
done

