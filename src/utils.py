"""
Utility functions for emotion classification.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


def get_device():
    """
    Get the available device (GPU or CPU).
    
    Returns:
        torch.device: Available device for computation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


def create_directories(dirs):
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")


def plot_confusion_matrix(y_preds, y_true, labels, title="Confusion Matrix", save=True):
    """
    Plot a normalized confusion matrix.
    
    Args:
        y_preds: Predicted labels
        y_true: True labels
        labels: Label names
        title: Title for the plot
        save: Whether to save the plot to file
    """
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=True)
    plt.title(title)
    plt.tight_layout()
    
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename = f"{title.lower().replace(' ', '_')}.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {filepath}")
    
    plt.show()


def compute_metrics(predictions, references=None):
    """
    Compute accuracy and F1 score.
    
    Args:
        predictions: Predicted labels or HuggingFace EvalPrediction object
        references: True labels (optional if predictions is EvalPrediction)
        
    Returns:
        Dictionary with accuracy and F1 score
    """
    if hasattr(predictions, 'predictions'):
        # HuggingFace Trainer output
        preds = predictions.predictions.argmax(-1)
        refs = predictions.label_ids
    else:
        preds = predictions
        refs = references
    
    accuracy = accuracy_score(refs, preds)
    f1 = f1_score(refs, preds, average="weighted")
    
    return {"accuracy": accuracy, "f1": f1}


def compute_metrics_for_trainer(eval_pred):
    """
    Compute metrics function for HuggingFace Trainer.
    
    Args:
        eval_pred: EvalPrediction object from Trainer
        
    Returns:
        Dictionary with accuracy and F1 score
    """
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


def print_model_info(model):
    """
    Print model parameter information.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (float32)")


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
