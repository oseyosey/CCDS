nvidia-smi

model_name=llama2-7b
number_of_gpus=4
validation_task=mmlu # mmlu, bbh, ifeval
include_validation=false
pipeline_parallel=false
batch_size_per_device=2
data_shuffle=false 
num_train_epochs=1
save_steps_per_epoch=1 
ppl_epoch=25 # number of epochs to train the PPL model (25)
use_accelerate_launch=true # use accelerate launch for multi-gpu training
method=mid-ppl # method for selecting the data (Mid-PPL, TOP-PPL)
out_dir=out_7b_mmlu
deepspeed_config=zero1


for learning_rate in 4e-05; do
    for ds_percentage in 0.025 0.05; do
        for dataset_seed in 3; do
        data_seed=$dataset_seed

        #* TRAINING
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        
        cd YOUR_REPO_PATH && \
        conda run -p YOUR_CONDA_ENV \
        bash ccds_scripts/train_ppl.sh "$ds_percentage" "$number_of_gpus" "$data_seed" "$validation_task" \
        "$include_validation" "$pipeline_parallel" "$batch_size_per_device" "$data_shuffle" \
        "$learning_rate" "$num_train_epochs" "$save_steps_per_epoch" "$ppl_epoch" \
        "$dataset_seed" "$use_accelerate_launch" "$method" "$model_name" "$out_dir" \
        "$deepspeed_config"

        #* EVALUATION
        export CUDA_VISIBLE_DEVICES=0

        model_path="YOUR_REPO_PATH/${out_dir}/${model_name}-{$validation_task}-{ppl-$method-$ppl_epoch}-p${ds_percentage}-lora-seed${data_seed}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus-dataseed$dataset_seed"
        cd YOUR_REPO_PATH/ccds/evaluation && \
        conda run -p YOUR_CONDA_ENV \
        bash eval_scripts/run_eval_mmlu.sh $model_path

        done
    done
done