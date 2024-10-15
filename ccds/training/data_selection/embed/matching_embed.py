import argparse
import os
import string
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity #* this one is normalized. 
from datasets import Dataset

from training.data_selection.get_training_dataset import load_raw_dataset
from training.data_selection.get_validation_dataset import get_raw_val_dataset

from training.data_selection.embed.embed import DenseEncoder  # Assuming you have this in a file or module

# Argument parsing
argparser = argparse.ArgumentParser(description='Script for ranking training data using cosine similarity of embeddings')
argparser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
argparser.add_argument('--train_files', type=str, nargs='+', required=True, help='Path to training files')
argparser.add_argument('--train_file_names', type=str, nargs='+', required=True, help='Names of the training files')
argparser.add_argument('--target_task_names', type=str, nargs='+', required=True, help='Names of the target tasks (e.g., BBH, TYDIQA, MMLU)')
argparser.add_argument('--output_path', type=str, default="selected_data", help='Directory for saving output')

args = argparser.parse_args()

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load and preprocess datasets
raw_datasets = load_raw_dataset(args.train_files, sample_percentage=1.0)
messages_corpus = [msg[0]['content'] for msg in raw_datasets['messages']]
messages_corpus_processed = [preprocess_text(doc) for doc in messages_corpus]

# Initialize the embedding model (Dense Encoder)
model_name = "sentence-transformers/gtr-t5-base"
model = DenseEncoder(model_name, max_seq_length=256)

# Encode the training data
train_dataset = Dataset.from_dict({'text': messages_corpus_processed})
train_embeddings = model.encode(train_dataset, "text", batch_size=512)

# Function to process each target task and calculate cosine similarity
def process_task(target_task_name):
    # Load validation dataset for the task
    raw_val_datasets = get_raw_val_dataset(task_name=target_task_name, data_dir=args.data_dir)
    raw_val_datasets_processed = [preprocess_text(doc) for _, doc in raw_val_datasets.items()]

    # Encode the validation dataset
    val_dataset = Dataset.from_dict({'text': raw_val_datasets_processed})
    val_embeddings = model.encode(val_dataset, "text", batch_size=128)

    # Calculate cosine similarity between each validation embedding and all training embeddings
    similarity_scores = cosine_similarity(val_embeddings, train_embeddings)

    # Average similarity scores across validation set
    avg_similarity_scores = np.mean(similarity_scores, axis=0)

    # Save the cosine similarity scores to a file
    output_dir = os.path.join(args.output_path, target_task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{target_task_name}_embedding_scores.pt")
    torch.save(avg_similarity_scores, output_file)
    print(f"Saved {model_name} embedding cosine similarity scores to {output_file}")

# Process each target task
for target_task_name in args.target_task_names:
    process_task(target_task_name)
