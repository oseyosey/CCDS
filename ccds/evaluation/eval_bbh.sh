source eval.sh

DIR=$1 # model directory
DATA_DIR=$2 # evaluation data directory
OUTPUT_DIR=$3 # output directory (if needed)


# main evaluation function
eval_bbh() {
    mdir=$DIR
    set_save_dir $mdir bbh $OUTPUT_DIR
    mkdir -p $save_dir
    cmd="python -m eval.bbh.run_eval \
    --data_dir $DATA_DIR/bbh/ \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 1 \
    --convert_to_bf16 \
    --max_num_examples_per_task 40 " 
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# evaluate the validation set, which is not supported yet
valid_bbh() {
    mdir=$DIR
    set_valid_dir $mdir bbh
    echo $save_dir
    mkdir -p $save_dir
    cmd="python -m eval.bbh.run_eval \
    --data_dir $DATA_DIR/bbh-valid \
    --save_dir $save_dir \
    --model $mdir \
    --tokenizer $mdir \
    --eval_batch_size 1 \
    --convert_to_bf16 \
    --eval_valid \
    --max_num_examples_per_task 3 "
    eval "$cmd" 2>&1 | tee $save_dir/log.txt
}

# extract the results
# extract_bbh() {
#     mdir=$DIR
#     set_save_dir $mdir bbh
#     result=$(jq .average_exact_match $save_dir/metrics.json)
#     result=$(echo "$result * 100" | bc)
#     echo $result
# }

# extract the results
extract_bbh() {
    mdir=$DIR
    set_save_dir $mdir bbh
    result=$(python3 -c "
import json
with open('$save_dir/metrics.json') as f:
    data = json.load(f)
    result = data.get('average_exact_match', 0) * 100
print(result)
")
    echo $result
}

# extract the results for the validation set
extract_valid_bbh() {
    mdir=$DIR
    set_valid_dir $mdir bbh
    result=$(jq .average_exact_match $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}


export -f eval_bbh
export -f valid_bbh
export -f extract_bbh
export -f extract_valid_bbh
