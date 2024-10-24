# Compute-Constrained Data Selection

![](plots/pdfs/Simulation_of_Performance_under_Constraints.png)

This repository offers an overview of the components discussed in the paper **Compute-constrained Data Selection**.

Data selection can reduce the amount of training data needed to finetune LLMs; however, the efficacy of data selection scales directly with its compute. Motivated by the practical challenge of compute-constrained finetuning, we consider the setting in which both the cost of selecting data and training are budgeted for. We first formalize the problem of data selection with a cost-aware utility function, and model the data selection problem as trading off initial-selection cost for training gain. We run a comprehensive sweep of experiments across multiple tasks, varying compute budget by scaling finetuning tokens, model sizes, and data selection compute. These experiments show the validity of this model in real-world experiments. Interestingly we find that many powerful data selection methods are almost never compute-optimal, and that cheaper data selection alternatives dominate both from a theoretical and empirical perspective. Models and datasets from our training runs are available via this repository.

## Table of Contents

<!-- TOC -->

- [Data Selection Methods](#data-selection-methods)
  - [Lexicon-Based](#Lexicon-Based)
  - [Embedding-Based](#Embedding-Based)
  - [Perplexity-Based](#Perplexity-Based)
  - [Gradient-Based](#Gradient-Based)
- [Models](#models)
  - [Overview](#Overview)
  - [Setup](#Setup)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Parametric Fit](#parametric-fit)
- [Plots](#Plots)
- [Citation](#citation)
- [Acknowledgement](#Acknowledgement)

<!-- /TOC -->

## Data Selection Methods

 We consider four classes of data selection in this work, that represent four different levels of compute.

### Lexicon-Based

Lexicon data selection methods utilize statistical properties of text to evaluate the relevance of data points without relying on deep learning models. We choose BM25 as it's one of the most effecitve and commonly used. Since the algorithm can be run efficiently with a single-core cpu, the data selection FLOPs are almost 0.

To run BM25 data selection for a specific task, refer to `ccds_scripts/calculate_bm25_score.sh`, which generates utility scores for each data entry and produces the selected (ordered) data:

```
train_file_names="flan_v2 cot dolly oasst1"
target_task_names="mmlu" # mmlu, bbh, ifeval
selected_data_output_path="data/selected_data"
select_percentage=1.0

./ccds/training/train_scripts/data_selection/matching_bm25.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" 

./ccds/training/train_scripts/data_selection/select_data_bm25.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage"
```

### Embedding-based 

These methods utilize embedding models to select data points that are most similar to the target data. Since we are using a very small model (BERT-size) and the embedding requires only one-time transformation of data points into dense vectors, the data selection FLOPs are quite small. 

To run our Embed data selection for a specific task, refer to ccds_scripts/calculate_embed_score.sh:

```
train_file_names="flan_v2 cot dolly oasst1"
target_task_names="mmlu" # mmlu, bbh, ifeval
selected_data_output_path="data/selected_data"
select_percentage=1.0

./ccds/training/train_scripts/data_selection/matching_embed.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" 

./ccds/training/train_scripts/data_selection/select_data_embed.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage"
```

### Perplexity-Based

Perplexity-based data selection utilizes language models to evaluate the utility of data points based on model loss. *Top-PPL* and *Mid-PPL* have both shown improved performance and training efficiency, where *Top-PPL* ranks data points with the highest perplexity scores, and *Mid-PPL* does the same for points in the middle of the score distribution. The data selection FLOPs is exactly one forward pass across the entire datasets. 

To perform PPL data selection, you must first fine-tune the model on the reference datasets. Afterward, you can run the selection by referencing `ccds_scripts/calculate_ppl_score.sh`:

```
train_file_names="flan_v2 cot dolly oasst1"
target_task_names=$target_task
selected_data_output_path="data/selected_data"
select_percentage=1.0
epochs=25 #* how many epochs the model are trained

model_dir="" # finetuned PPL model directory

./ccds/training/train_scripts/data_selection/matching_ppl.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$model_dir"  "$epochs"

#TOP PPL 
for data_seed in 3 6 9 12 15; do
    ./ccds/training/train_scripts/data_selection/select_data_ppl_top.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage" "$epochs" "$data_seed"
done

# MID PPL 
for data_seed in 3 6 9 12 15; do
    ./ccds/training/train_scripts/data_selection/select_data_ppl_mid.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage" "$epochs" "$data_seed"
done
```

### Gradient-Based

Gradinet-based methods generally evaluate the utility of data points based on their influence on the model’s loss with respect to the target data. Computing gradients involves both forward and backward passes, totaling approximately three times the cost of a forward pass (the actually implementation may requires more). [Low-rank sgradiEnt Similarity Search ](https://arxiv.org/pdf/2402.04333)(*LESS*) shows superior performance gain over cheaper methods and random selection.

Please refer to [LESS](https://github.com/princeton-nlp/LESS) to see how to perform gradient-based data selection (we also provide our implementation of LESS in the ccds codebase). To summarize, you need to:

- **Step 1:** Warmup training for 4 epochs

- **Step 2:** Building the gradient datastore on training dataset
- **Step 3:** Building the gradient datastore on reference/validation dataset
- **Step 4:** Calculate the influence score for each training data point w.r.t the validaiton data point

Once you obtain the influence scores for each data with LESS, you can use ```ccds_scripts/calculate_less_score.sh``` to select data:

```
train_file_names="flan_v2 cot dolly oasst1"
target_task_names="mmlu"
selected_data_output_path="data/selected_data"
select_percentage=1.0
less_seed=3

./ccds/training/train_scripts/data_selection/select_data_less.sh "$train_file_names" "$target_task_names" "$selected_data_output_path" "$select_percentage" "$less_seed"
```

## 

## Models

### Overview

Our training runs can be downloaded at https://huggingface.co/CCDS-Scaling.

Models follow the naming convention: {model}-{parameters}-{task}-{data_selection_method}-{tokens_percentage}-{SFT_method}-{seed}. For example, llama2-7b-ifeval-less-p0.10-lora-seed2-linear-1epoch represents a Llama2 model with 7 billion parameters, trained on IFEval, using LESS for data selection, with 10% of unique training tokens, on linear decay learning schedule, and trained for 1 epoch. 

Datasets are also available on the same site.

Dataset names follow the format: {task}-train-p1.0-{data_selection_method}. Deterministic methods like BM25 and Embed are named accordingly, e.g., ifeval-train-p1.0-embedding. Stochastic methods like PPL and LESS include a seed number, e.g., ifeval-train-p1.0-less-seed3.

Note that perplexity-based and gradient-based methods use a Small-size model (Llama2 7B) for PPL and LESS.

### Setup

To setup the CCDS environment to reproduce our training runs, you can follow these installation steps.

**Step 1**: Ensure you have [PyTorch](https://pytorch.org/get-started/previous-versions/) installed:

```
conda create -n ccds python=3.10

pip3 install torch==2.1.2 torchvision torchaudio
```

**Step 2**: Install the required packages:

```
cd CCDS
pip install -r requirement.txt
```

**Step 3**:  Install the less package in editable mode for development:

```
pip install -e .
```

You will also need to download the processed [ccds_data](https://huggingface.co/datasets/CCDS-Scaling/ccds_data). Once unzipped, You have to put the folder in the directory `CCDS/ccds`. You will also need to download the necessary datasets you want to have that is processed by different data selection methods. As an example, you can dwnoload data selected by [LESS on IFEVAL](https://huggingface.co/datasets/CCDS-Scaling/ifeval-train-p1.0-less-seed3) and put inside the unzipped data folder. 

### Training

We include the shell running script that was used to train the model in `ccds_scripts`. Once you complete the set up, you can use these as a reference to train the models we run in our experiments.

You need to adapt the ncessary enviornment variables and paths for successful running. Our setup is provided in ccds_scripts/launch_scripts.

For example, to train a 7B model on 5% of the data selected by perplexity, you can follow the provided script. For further details on our experimental setup, please refer to our paper.

```
for learning_rate in 4e-05; do
    for ds_percentage in 0.05; do
        for dataset_seed in 3; do
        data_seed=$dataset_seed

        #* TRAINING
        cd YOUR_REPO_PATH && \
        conda run -p YOUR_CONDA_ENV \
        bash ccds_scripts/train_ppl.sh "$ds_percentage" "$number_of_gpus" "$data_seed" "$validation_task" \
        "$include_validation" "$pipeline_parallel" "$batch_size_per_device" "$data_shuffle" \
        "$learning_rate" "$num_train_epochs" "$save_steps_per_epoch" "$ppl_epoch" \
        "$dataset_seed" "$use_accelerate_launch" "$method" "$model_name" "$out_dir" \
        "$deepspeed_config"

        done
    done
done
```

### Evaluation

Follow the instructions in `ccds/evaluation/README.md` to set up the evaluation environment. Use `ccds_scripts/launch_scripts/launch-random-data-7b-mmlu.sh` for both training and evaluation. Be sure to modify the necessary variables and configure your Hugging Face tokens and paths in the script.

For evaluation only, use the following script (MMLU example):

```
model_path="YOUR_MODEL_PATH"
cd YOUR_REPO_PATH/ccds/evaluation && \
conda run -p YOUR_CONDA_ENV \
bash eval_scripts/run_eval_mmlu.sh $model_path
```

## Parametric Fit

To analyze the trade-off between the compute of data selection methods and the expected gain in model performance, we define a simplified parametric form for expected peformance as the following:

$$
P(k) = (\bar{P}  - P_0)\times \left( 1 - \exp\left( -\lambda \frac{C(k)}{C(|\mathcal{D}|)} \right) \right) + P_0.
$$

We fit the compute-constrained parametric fit performance function using the code in `plots/MMLU_BBH_Combined_Plot_Fit.ipynb`, which can also be found in this [colab notebook](https://colab.research.google.com/drive/1TakWDNYYLQqcYOCK43xk2s5wcbhw9GIA?usp=sharing). 

Note that the actual fitted parameters is unlikely to be useful: the performance gain rate is model and dataset dependent. However, the close fit of the curves indicates that the underlying model — significant performance gain at the cost of exponentially more compute investment — holds up in practical settings.

To perform fitting, we use the curve_fit from scipy with Levenberg-Marquardt algorithm:

```
# Performance function in terms of data selection FLOPs
def gain_function(flops, P0, lambda_, flops_total, P):
    return P0 + (P - P0) * (1 - np.exp(-lambda_ * (flops / flops_total)))

P = P_values_mmlu[model_size]

flops = flops_data_mmlu[model_size][method]
mmlu = mmlu_data[model_size][method]

P0_initial = mmlu[0]
lambda_initial = 1.0

popt, _ = curve_fit(
    lambda flops, P0, lambda_: gain_function(flops, P0, lambda_, flops_total, P),
    flops, mmlu,
    p0=[E0_initial, lambda_initial],
    bounds=([0, 0], [H, np.inf]),
    maxfev=10000
)
E0_opt, lambda_opt = popt
```

## Plots

Code to produce the plots in our main paper can be found in these respective files / links.

- Figure 1: `plots/Simulation_of_Performance_under_Constraints.pdf`, [colab](https://colab.research.google.com/drive/1XNqmNU_SHYE1q-8c6OCXCV8KVUVfp_3a?usp=sharing)
- Figure 2: `plots/MMLU_BBH_Combined_Plot_With_Annotations.pdf`, [colab](https://colab.research.google.com/drive/1V7GpBa7lTn3YfPeZW5tqhK8WWA4Qg_Qg?usp=sharing)
- Figure 3: `plots/MMLU_BBH_Combined_Plot_Fit.pdf`, [colab](https://colab.research.google.com/drive/1TakWDNYYLQqcYOCK43xk2s5wcbhw9GIA?usp=sharing)
- Figure 4: `plots/Break-Even_Analysis_MMLU_LESS_7B.pdf`,  [colab](https://colab.research.google.com/drive/1tnZo2Y1fW8tQKdSlTkGL0biuOeI0hKJJ?usp=sharing)
- Figure 5 (a): `plots/Fixed_Training_Budget_MMLU_7B.pdf`, [colab](https://colab.research.google.com/drive/1k72sZ4D3qvRE_G8hnpPOiMT-ANpnaNR-?usp=sharing)
- Figure 5 (b): `plots/IFEval_7B_Comparison.pdf`, [colab](https://colab.research.google.com/drive/1nTSr2HeEUTtQQJU7LZrwbRL9aUjVptAT?usp=sharing)
- Figure 6: `plots/MMLU_Dataset_Sim_Heatmap.pdf`, [colab](https://colab.research.google.com/drive/1rpHMUJOk06GGZwwFRelwJ2GOIrYkwvHs?usp=sharing)

## Citation

```bibtex
@article{yin2024compute-constrained,
  title={Compute-Constrained Data Selection},
  author={Yin, Junjie Oscar and Rush, Alexander M.},
  journal={arXiv preprint arXiv:2410.16208},
  year={2024}
}
```

## Acknowledgement

- The training code is built upon [LESS](https://github.com/princeton-nlp/LESS) by Xia et al. 
- The evaluation code is modified from [OpenInstruct](https://github.com/allenai/open-instruct) by AI2. 
- This work is based on the [LLaMA 2/3](https://ai.meta.com/llama) as the pre-trained models.
- We use [LoRA](https://arxiv.org/pdf/2106.09685), [PEFT](https://huggingface.co/docs/peft/en/index) for supervised fine-tuning.

