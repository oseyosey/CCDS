set_save_dir() {
    mdir=$1
    if [[ -d $mdir ]]; then
        save_dir=${mdir}/eval/$2
    else
        # save_dir=$n/space10/out/$(basename $mdir)/eval/$2 # path needs to be modified.
        save_dir=$3
        # save_dir="output_dir="/share/kuleshov/jy928/compute_optimal_data_selection/out/llama2-7b-bbh-base"
    fi
}

set_valid_dir() {
    mdir=$1
    if [[ -d $mdir ]]; then
        save_dir=${mdir}/valid/$2
    else
        save_dir=$3
    fi
}

export DATA_DIR=$n/../data/eval # assume the directory is at LESS/evaluation
export set_save_dir
export set_valid_dir

