"""
Data loading and preprocessing for emotion classification.
"""

import os
import sys
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab')


def load_emotions_dataset():
    """
    Load the emotions dataset from Hugging Face Hub.
    
    Returns:
        DatasetDict containing train, validation, and test splits
    """
    print("\nLoading emotions dataset from Hugging Face Hub...")
    dataset = load_dataset("dair-ai/emotion")
    
    print(f"\nDataset Overview:")
    print(f"  Training samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['validation']):,}")
    print(f"  Test samples: {len(dataset['test']):,}")
    print(f"  Features: {dataset['train'].features}")
    
    return dataset


def explore_dataset(dataset, save_plots=True):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        dataset: DatasetDict from Hugging Face
        save_plots: Whether to save plots to files
    """
    print("\n" + "-" * 60)
    print("Exploratory Data Analysis")
    print("-" * 60)
    
    # Convert to pandas
    dataset.set_format(type="pandas")
    df = dataset["train"][:]
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nSample Data:")
    print(df.head())
    
    # Add label names
    def label_int2str(row):
        return dataset["train"].features["label"].int2str(row)
    
    df["label_name"] = df["label"].apply(label_int2str)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    ax = df["label_name"].value_counts(ascending=True).plot.barh(color='steelblue')
    plt.title("Emotion Distribution in Training Data", fontsize=14, fontweight='bold')
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Emotion", fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(df["label_name"].value_counts(ascending=True)):
        ax.text(v + 50, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    if save_plots:
        filepath = os.path.join(RESULTS_DIR, "class_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {filepath}")
    plt.show()
    
    # Analyze text lengths
    df["words_per_text"] = df["text"].str.split().apply(len)
    
    print(f"\nText Length Statistics:")
    print(df["words_per_text"].describe())
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='label_name', y='words_per_text', palette='Set2')
    plt.title("Text Length Distribution by Emotion", fontsize=14, fontweight='bold')
    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Words per Text", fontsize=12)
    plt.tight_layout()
    if save_plots:
        filepath = os.path.join(RESULTS_DIR, "text_length_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.show()
    
    # Reset format for further processing
    dataset.reset_format()
    
    return dataset


def tokenize_dataset(dataset, model_checkpoint, max_length=128):
    """
    Tokenize the entire dataset using the specified tokenizer.
    
    Args:
        dataset: DatasetDict from Hugging Face
        model_checkpoint: Hugging Face model checkpoint name
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (tokenized DatasetDict, tokenizer)
    """
    print(f"\nTokenizing dataset with {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    def tokenize_function(examples):
        """Tokenize a batch of examples."""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    print(f"\nTokenization complete:")
    print(f"  Columns: {tokenized_dataset['train'].column_names}")
    print(f"  Max length: {max_length}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    
    return tokenized_dataset, tokenizer


def get_labels(dataset):
    """
    Get emotion labels from dataset.
    
    Args:
        dataset: DatasetDict from Hugging Face
        
    Returns:
        List of label names
    """
    return dataset["train"].features["label"].names


def demonstrate_tokenization(model_checkpoint):
    """
    Demonstrate different tokenization approaches for educational purposes.
    
    Args:
        model_checkpoint: Hugging Face model checkpoint name
    """
    print("\n" + "-" * 60)
    print("Tokenization Demonstration")
    print("-" * 60)
    
    # Download NLTK resources
    download_nltk_resources()
    
    text = "I'm feeling absolutely wonderful today! 😊"
    print(f"\nOriginal text: '{text}'")
    
    # Character tokenization
    char_tokens = list(text)
    print(f"\n1. Character Tokenization:")
    print(f"   Tokens: {char_tokens[:20]}... ({len(char_tokens)} total)")
    
    # Word tokenization (NLTK)
    from nltk import word_tokenize
    word_tokens = word_tokenize(text)
    print(f"\n2. Word Tokenization (NLTK):")
    print(f"   Tokens: {word_tokens}")
    
    # Subword tokenization (DistilBERT)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    encoded = tokenizer(text)
    subword_tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids)
    print(f"\n3. Subword Tokenization ({model_checkpoint}):")
    print(f"   Tokens: {subword_tokens}")
    print(f"   Input IDs: {encoded.input_ids}")
    
    # Special tokens
    print(f"\n4. Special Tokens:")
    for token, token_id in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids):
        print(f"   {token}: {token_id}")
