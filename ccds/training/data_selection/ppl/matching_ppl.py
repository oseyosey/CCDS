import argparse
import os
import re
import string
import numpy as np
from tqdm import tqdm

import torch

from ccds.training.data_selection.get_training_dataset import load_raw_dataset
from ccds.training.data_selection.get_validation_dataset import get_raw_val_dataset

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training w. BM25 (Word Frequency)')
argparser.add_argument('--model_dir', type=str, nargs='+',
                           help='"The path to the model (used for calculating ppl & nll)')
argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks (BBH, TYDIQA, MMLU)")
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')
argparser.add_argument('--start_index', type=int, default=0,
                       help='The start index of the data')
argparser.add_argument('--end_index', type=int, default=270679,
                          help='The end index of the data')
argparser.add_argument('--epochs_num', type=int, 
                          help='The number of epochs for training the model')


args = argparser.parse_args()
target_task_name = args.target_task_names[0]
model_dir = args.model_dir[0]
start_index = args.start_index
end_index = args.end_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# edit the function to support lora loading
def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        convert_to_bf16=True,
        gptq_model=False,
        use_fast_tokenizer=True,
        padding_side="left",
    ):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
    
    is_peft = "lora" in model_name_or_path #* bit hacky as it requires our path name to contain the string "lora"
    from peft import PeftConfig, PeftModel
    
    if is_peft:
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        peft_dir = model_name_or_path
        model_name_or_path = peft_config.base_model_name_or_path
        
    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=None)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        
        if is_peft:
            model = PeftModel.from_pretrained(model, peft_dir, device_map="auto").merge_and_unload()
            print(f"loaded the peft model") 
         
        if convert_to_half:
            model = model.half()
            print("Convert model to half precision.")
            assert next(model.parameters()).dtype == torch.float16, "model parameters should be in half precision"
        elif convert_to_bf16:
            model = model.bfloat16()
            print("Convert model to bfloat16 precision.")
            assert next(model.parameters()).dtype == torch.bfloat16, "model parameters should be in bfloat16 precision"
        else:
            print("Model is not converted to half or bfloat16 precision.")
            assert next(model.parameters()).dtype == torch.float32, "model parameters should be in float32 precision"

    model.eval()

    if not tokenizer_name_or_path:
        if is_peft:
            tokenizer_name_or_path = peft_dir
        else:
            tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # replace with new embeddings 
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings 
    # to avoid wrong embedding index.    
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        
    return model, tokenizer


#* Obtain Model's Perplexity *#
# Modified from https://huggingface.co/docs/transformers/en/perplexity

#? currently assume batch size of 1
def compute_nll_ppl(model, encodings):
    max_length = 2048
    stride = 512
    seq_len = encodings.shape[1]

    device = "cuda"
    nlls = []
    prev_end_loc = 0
    
    token_probs = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

            # # Compute the probabilities for each token
            # # Apply softmax to the logits to get probabilities
            # probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
            # # probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # all_prob = []
            # input_ids_processed = input_ids[0][1:]
            # for i, token_id in enumerate(input_ids_processed):
            #     probability = probabilities[0, i, token_id].item()
            #     all_prob.append(probability)

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = torch.stack(nlls).mean()
    ppl = torch.exp(avg_nll)
    return ppl.item(), avg_nll.item() #, all_prob

#* Load in Model
model, tokenizer = load_hf_lm_and_tokenizer(model_dir)

#* load in training datasets
raw_datasets = load_raw_dataset(
        args.train_files, sample_percentage=1.0)

raw_datasets_dict = raw_datasets.to_dict() # have to first turn it into dict to be efficient

raw_messages = []
for idx in tqdm(range(len(raw_datasets['messages']))):
    message = raw_datasets_dict['messages'][idx][0]['content']+ " A:" + raw_datasets_dict['messages'][idx][1]['content']
    raw_messages.append(message)

#* calculate the perplexity of each instance in training datasets
ppl_scores = []
# for idx in tqdm(range(start_index, end_index)):
for idx in tqdm(range(len(raw_datasets['messages']))):
    input_text_encodings = tokenizer.encode(raw_messages[idx], return_tensors="pt")
    ppl, nll = compute_nll_ppl(model, input_text_encodings)
    ppl_scores.append(ppl)

import numpy
ppl_scores_numpy = numpy.array(ppl_scores)
    
output_dir = os.path.join(args.output_path, target_task_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(
    args.output_path, target_task_name, f"{target_task_name}_ppl_score_epoch{args.epochs_num}.pt")
torch.save(ppl_scores_numpy, output_file)
print("Saved less score to {}".format(output_file))
