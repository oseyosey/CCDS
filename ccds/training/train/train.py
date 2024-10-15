#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time

import datasets
import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          set_seed)
from transformers.optimization import AdamW

from huggingface_hub import login

from accelerate import Accelerator
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from training.data_selection.get_training_dataset import get_training_dataset, get_training_dataset_with_validation
from training.train.data_arguments import DataArguments, get_data_statistics
from training.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from training.train.training_arguments import TrainingArguments


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "300"

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #* Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    #* Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")
    logger.info(f"Total Number of GPUS (Cuda) {torch.cuda.device_count()}")

    #* login set up
    login(token = "YOUR_HF_TOKEN")

    #* Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    #* Load training dataset
    if training_args.val_only:
        #* (Training with Validation Dataset Only)
        from training.data_selection.get_validation_dataset import get_dataset
        train_dataset = get_dataset(training_args.analysis_dataset,
                                    data_dir=data_args.data_dir,
                                    tokenizer=tokenizer,
                                    max_length=data_args.max_seq_length)
    else:
        #* (Training without Validation Dataset)
        train_dataset = get_training_dataset(data_args.train_files,
                                            tokenizer=tokenizer,
                                            max_seq_length=data_args.max_seq_length,
                                            sample_percentage=data_args.percentage,
                                            # seed=data_args.sample_data_seed,
                                            seed = training_args.data_seed,
                                            data_shuffle=data_args.data_shuffle)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])
            
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    
    analysis_dataset = None
    if training_args.analysis_mode:
        from training.data_selection.get_validation_dataset import get_dataset
        analysis_dataset = get_dataset(training_args.analysis_dataset,
                                       data_dir=data_args.data_dir,
                                       tokenizer=tokenizer,
                                       max_length=data_args.max_seq_length)

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    # Adjust Gradient Accumulation Steps based on number of gpus
    training_args.gradient_accumulation_steps = training_args.batch_size // ( torch.cuda.device_count() * training_args.per_device_train_batch_size)
    logger.info(f"Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
    
    # enabling automatic fitting of batch size
    # training_args.auto_find_batch_size = True

    #* Train & Eval Steps Calculation
    total_steps = len(train_dataset) * training_args.num_train_epochs // training_args.batch_size
    train_steps = total_steps // ( training_args.save_steps_per_epoch * training_args.num_train_epochs) 
    training_args.eval_steps = train_steps * training_args.save_steps_per_epoch # perform evaluation per epoch
    training_args.save_steps = train_steps # save checkpoints every 0.2 epochs
    if training_args.eval_steps == 0: #? In the case train_steps < num_train_epochs, eval_steps will be 0.
        training_args.eval_steps = total_steps
    
    logger.info(f"Number of Training Epochs: {training_args.num_train_epochs}")
    logger.info(f"Total Steps: {total_steps}")
    logger.info(f"Train Steps: {train_steps}")
    logger.info(f"Eval steps: {training_args.eval_steps}")
    logger.info(f"Save steps: {training_args.save_steps}")

    training_args.gradient_checkpointing = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None, #analysis_dataset
        tokenizer=tokenizer,
        # optimizers = (optimizer, lr_scheduler),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest")
    )

    def contains_checkpoint_directories(output_dir):
        entries = os.listdir(output_dir)
        
        has_subdirs = False
        for entry in entries:
            entry_path = os.path.join(output_dir, entry)
            if os.path.isdir(entry_path):
                has_subdirs = True
                break

        if not has_subdirs:
            return False
        
        for entry in entries:
            entry_path = os.path.join(output_dir, entry)
            if os.path.isdir(entry_path) and "checkpoint" in entry:
                return True
        return False
    

    #* Training
    if os.path.exists(training_args.output_dir) and contains_checkpoint_directories(training_args.output_dir):
        logger.info("Resuming training from checkpoint...")
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()
    
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()
