"""
Full fine-tuning of transformer model for emotion classification.

This approach updates all model parameters during training, allowing
the model to fully adapt to the emotion classification task.
"""

import os
import sys
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_LABELS, MODELS_DIR
)
from src.utils import (
    plot_confusion_matrix, 
    print_model_info, 
    print_section_header,
    compute_metrics_for_trainer
)


def fine_tune_model(emotions_encoded, model_checkpoint, labels, device):
    """
    Fine-tune a pretrained transformer model for sequence classification.
    
    This function:
    1. Loads a pretrained model with a classification head
    2. Trains all parameters on the emotion dataset
    3. Evaluates on validation set
    4. Saves the best model
    
    Args:
        emotions_encoded: Tokenized dataset
        model_checkpoint: Hugging Face model checkpoint
        labels: List of emotion labels
        device: torch device
        
    Returns:
        Tuple of (trained model, trainer)
    """
    print_section_header("FULL FINE-TUNING")
    print("Training all model parameters")
    
    # Load model for sequence classification
    print(f"\nLoading model: {model_checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=NUM_LABELS,
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)}
    ).to(device)
    
    print_model_info(model)
    
    # Define output directory
    output_dir = os.path.join(MODELS_DIR, "distilbert-finetuned-emotion")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate logging steps
    num_training_steps = len(emotions_encoded["train"]) // BATCH_SIZE
    logging_steps = max(num_training_steps // 4, 1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        disable_tqdm=False,
        logging_steps=logging_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        push_to_hub=False,
        log_level="warning",
        report_to="none",
        seed=42,
        # GPU optimizations
        fp16=device.type == "cuda",  # Mixed precision on GPU
        dataloader_num_workers=4 if device.type == "cuda" else 0,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Mixed precision (fp16): {training_args.fp16}")
    
    # Create trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics_for_trainer,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["validation"],
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 40)
    train_result = trainer.train()
    
    # Print training results
    print("\nTraining completed!")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Training time: {train_result.metrics['train_runtime']:.2f}s")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"  Validation loss: {eval_results['eval_loss']:.4f}")
    print(f"  Validation accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Validation F1: {eval_results['eval_f1']:.4f}")
    
    # Save the model
    trainer.save_model(output_dir)
    print(f"\nModel saved to: {output_dir}")
    
    # Get predictions for confusion matrix
    predictions = trainer.predict(emotions_encoded["validation"])
    y_preds = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    plot_confusion_matrix(
        y_preds, y_true, labels,
        title="Fine-Tuning Confusion Matrix"
    )
    
    return model, trainer


def load_fine_tuned_model(model_path, device):
    """
    Load a fine-tuned model from disk.
    
    Args:
        model_path: Path to saved model
        device: torch device
        
    Returns:
        Loaded model
    """
    print(f"Loading fine-tuned model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    return model
