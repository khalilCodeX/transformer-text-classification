"""
Feature extraction using pretrained model hidden states.

This approach uses the pretrained transformer as a fixed feature extractor,
training only a simple classifier on top of the extracted representations.
"""

import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from transformers import AutoModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import compute_metrics, plot_confusion_matrix, print_model_info, print_section_header


def extract_hidden_states(dataset, model_checkpoint, device):
    """
    Extract hidden states from pretrained model for all examples.
    
    This function passes each example through the transformer and extracts
    the [CLS] token representation from the last hidden layer.
    
    Args:
        dataset: Tokenized DatasetDict
        model_checkpoint: Hugging Face model checkpoint
        device: torch device (cuda or cpu)
        
    Returns:
        Dataset with extracted hidden states
    """
    print_section_header("FEATURE EXTRACTION")
    print("Using pretrained model as fixed feature extractor")
    
    # Load pretrained model
    print(f"\nLoading pretrained model: {model_checkpoint}")
    model = AutoModel.from_pretrained(model_checkpoint).to(device)
    model.eval()  # Set to evaluation mode
    print_model_info(model)
    
    # Set format for PyTorch
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    def extract_hidden_states_batch(batch):
        """Extract CLS token hidden state for a batch."""
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            outputs = model(**inputs)
            # Get [CLS] token representation (first token)
            last_hidden_state = outputs.last_hidden_state
            cls_hidden_state = last_hidden_state[:, 0]
        return {"hidden_state": cls_hidden_state.cpu().numpy()}
    
    # Extract hidden states
    print("\nExtracting hidden states from all samples...")
    dataset_with_hidden = dataset.map(
        extract_hidden_states_batch, 
        batched=True,
        batch_size=64,
        desc="Extracting features"
    )
    
    print(f"Hidden state dimension: {dataset_with_hidden['train'][0]['hidden_state'].shape}")
    
    return dataset_with_hidden


def train_classifier(dataset_with_hidden, labels):
    """
    Train logistic regression classifier on extracted features.
    
    Args:
        dataset_with_hidden: Dataset with extracted hidden states
        labels: List of emotion labels
        
    Returns:
        Tuple of (trained classifier, metrics dictionary)
    """
    print("\n" + "-" * 60)
    print("Training Logistic Regression Classifier")
    print("-" * 60)
    
    # Prepare features
    X_train = np.array(dataset_with_hidden["train"]["hidden_state"])
    X_valid = np.array(dataset_with_hidden["validation"]["hidden_state"])
    y_train = np.array(dataset_with_hidden["train"]["label"])
    y_valid = np.array(dataset_with_hidden["validation"]["label"])
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_valid: {X_valid.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_valid: {y_valid.shape}")
    
    # Baseline comparison (most frequent class)
    print("\n1. Baseline Model (Most Frequent Class):")
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    dummy_score = dummy_clf.score(X_valid, y_valid)
    print(f"   Accuracy: {dummy_score:.4f}")
    
    # Train logistic regression
    print("\n2. Logistic Regression:")
    lr_clf = LogisticRegression(
        max_iter=500,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    lr_clf.fit(X_train, y_train)
    
    # Evaluate
    lr_score = lr_clf.score(X_valid, y_valid)
    print(f"   Accuracy: {lr_score:.4f}")
    
    # Get predictions
    y_preds = lr_clf.predict(X_valid)
    metrics = compute_metrics(y_preds, y_valid)
    print(f"   F1 Score: {metrics['f1']:.4f}")
    
    # Improvement over baseline
    improvement = (lr_score - dummy_score) / dummy_score * 100
    print(f"\n   Improvement over baseline: {improvement:.1f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_preds, y_valid, labels,
        title="Feature Extraction Confusion Matrix"
    )
    
    return lr_clf, metrics


def run_feature_extraction(emotions_encoded, model_checkpoint, labels, device):
    """
    Run the complete feature extraction pipeline.
    
    Args:
        emotions_encoded: Tokenized dataset
        model_checkpoint: Hugging Face model checkpoint
        labels: List of emotion labels
        device: torch device
        
    Returns:
        Tuple of (classifier, metrics)
    """
    # Extract hidden states
    dataset_with_hidden = extract_hidden_states(
        emotions_encoded, model_checkpoint, device
    )
    
    # Train classifier
    classifier, metrics = train_classifier(dataset_with_hidden, labels)
    
    return classifier, metrics
