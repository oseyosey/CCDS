nvidia-smi

MODEL_NAME=llama2-7b
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


for learning_rate in 4e-05; do
    for ds_percentage in 0.025 0.05; do
        for DATA_SEED in 3 6 9 12 15; do

            #* Training 
            export CUDA_VISIBLE_DEVICES=0,1

            cd YOUR_REPO_PATH && \
            conda run -p YOUR_CONDA_ENV \
            bash ccds_scripts/train_embed.sh \
            "$ds_percentage" "$number_of_gpus" "$DATA_SEED" "$validation_task" "$include_validation" "$pipeline_parallel" \
            "$batch_size_per_device" "$data_shuffle" "$learning_rate" "$num_train_epochs" "$save_steps_per_epoch" \
            "$MODEL_NAME" "$use_accelerate_launch" "$out_dir" "$deepspeed_config"

            #* Evaluation 
            export CUDA_VISIBLE_DEVICES=0

            model_path="YOUR_REPO_PATH/${out_dir}/${MODEL_NAME}-{$validation_task}-{embedding}-p${ds_percentage}-lora-seed${DATA_SEED}-lr${learning_rate}-${num_train_epochs}epoch-$number_of_gpus"
            cd YOUR_REPO_PATH/ccds/evaluation && \
            conda run -p YOUR_CONDA_ENV \
            bash eval_scripts/run_eval_mmlu.sh $model_path

        done
    done
done


