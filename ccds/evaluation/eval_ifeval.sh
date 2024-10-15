source eval.sh

DIR=$1
DATA_DIR=$2
OUTPUT_DIR=$3 # output directory (if needed)


# main evaluation function
eval_ifeval() {
    mdir=$DIR
    set_save_dir $mdir ifeval $OUTPUT_DIR
    mkdir -p $save_dir
    cmd="python -m eval.ifeval.run_eval \
    --data_dir $DATA_DIR/ifeval/ \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format" 
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

export -f eval_ifeval


# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
# export CUDA_VISIBLE_DEVICES=0

# Evaluating tulu 7B model using chat format
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-7B-sft \
#     --model ../checkpoints/tulu2/7B-sft \
#     --tokenizer ../checkpoints/tulu2/7B-sft \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
