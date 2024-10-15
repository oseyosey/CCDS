source eval.sh

DIR=$1 # model directory
DATA_DIR=$2 # evaluation data directory
output_dir=$3 # output directory (if needed)
TYPE=$4

# main evaluation function
eval_mmlu() {
    mdir=$DIR
    set_save_dir $mdir mmlu $output_dir
    mkdir -p $save_dir
    cmd="python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir $DATA_DIR/mmlu \
    --save_dir $save_dir \
    --model_name_or_path $mdir \
    --tokenizer_name_or_path $mdir \
    --eval_batch_size 1 \
    --convert_to_bf16"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# evaluate the validation set, which is not supported yet
valid_mmlu() {
    mdir=$DIR
    type=$TYPE
    set_valid_dir $mdir mmlu
    mkdir -p $save_dir
    cmd="python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --eval_valid \
    --data_dir $DATA_DIR/mmlu \
    --save_dir $save_dir \
    --model_name_or_path $mdir \
    --tokenizer_name_or_path $mdir \
    --eval_batch_size 1 \
    --convert_to_bf16"
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# # extract the results
# extract_mmlu() {
#     mdir=$DIR
#     set_save_dir $mdir mmlu
#     result=$(jq .average_acc $save_dir/metrics.json)
#     result=$(echo "$result * 100" | bc)
#     echo $result
# }

# extract the results (not using jq but python)
extract_mmlu() {
    mdir=$DIR
    set_save_dir $mdir mmlu
    result=$(python3 -c "
import json
with open('$save_dir/metrics.json') as f:
    data = json.load(f)
    result = data.get('average_acc', 0) * 100
print(result)
")
    echo $result
}

# extract the results for the validation set
extract_valid_mmlu() {
    mdir=$DIR
    set_valid_dir $mdir mmlu
    result=$(jq .average_acc $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

export -f eval_mmlu
export -f valid_mmlu
export -f extract_mmlu
export -f extract_valid_mmlu
