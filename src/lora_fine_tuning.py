"""
LoRA fine-tuning of transformer model for efficient training.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique
that adds small trainable matrices to the model while keeping the original
weights frozen. This dramatically reduces the number of trainable parameters.
"""

import os
import sys
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_LABELS, LORA_CONFIG, MODELS_DIR
)
from src.utils import (
    plot_confusion_matrix, 
    print_model_info, 
    print_section_header,
    compute_metrics_for_trainer
)


def create_lora_model(model_checkpoint, labels, device):
    """
    Create a model with LoRA adapters applied.
    
    LoRA works by:
    1. Freezing all pretrained weights
    2. Adding low-rank decomposition matrices (A and B) to attention layers
    3. Only training these small adapter matrices
    
    Args:
        model_checkpoint: Hugging Face model checkpoint
        labels: List of emotion labels
        device: torch device
        
    Returns:
        Model with LoRA adapters
    """
    print_section_header("LoRA FINE-TUNING")
    print("Parameter-Efficient Fine-Tuning with Low-Rank Adaptation")
    
    # Load base model
    print(f"\nLoading base model: {model_checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=NUM_LABELS,
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)}
    ).to(device)
    
    print("\nBase model before LoRA:")
    print_model_info(model)
    
    # Configure LoRA
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {LORA_CONFIG['r']}")
    print(f"  Alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"  Dropout: {LORA_CONFIG['lora_dropout']}")
    print(f"  Target modules: {LORA_CONFIG['target_modules']}")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias="none"
    )
    
    # Apply LoRA adapters
    print("\nApplying LoRA adapters...")
    lora_model = get_peft_model(model, lora_config)
    
    print("\nModel after LoRA:")
    print_model_info(lora_model)
    
    # Print LoRA layers
    print("\nLoRA layers added:")
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
    
    return lora_model


def fine_tune_lora(emotions_encoded, model_checkpoint, labels, device):
    """
    Fine-tune a transformer model using LoRA for efficient training.
    
    Benefits of LoRA:
    - 99% fewer trainable parameters
    - Faster training
    - Lower memory requirements
    - Comparable performance to full fine-tuning
    
    Args:
        emotions_encoded: Tokenized dataset
        model_checkpoint: Hugging Face model checkpoint
        labels: List of emotion labels
        device: torch device
        
    Returns:
        Tuple of (LoRA model, trainer)
    """
    # Create LoRA model
    lora_model = create_lora_model(model_checkpoint, labels, device)
    
    # Define output directory
    output_dir = os.path.join(MODELS_DIR, "lora-distilbert-finetuned-emotion")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate logging steps
    num_training_steps = len(emotions_encoded["train"]) // BATCH_SIZE
    logging_steps = max(num_training_steps // 4, 1)
    
    # Training arguments (similar to full fine-tuning)
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
        fp16=device.type == "cuda",
        dataloader_num_workers=4 if device.type == "cuda" else 0,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Mixed precision (fp16): {training_args.fp16}")
    
    # Create trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        compute_metrics=compute_metrics_for_trainer,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["validation"],
    )
    
    # Train
    print("\nStarting LoRA training...")
    print("-" * 40)
    train_result = trainer.train()
    
    # Print training results
    print("\nLoRA training completed!")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Training time: {train_result.metrics['train_runtime']:.2f}s")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"  Validation loss: {eval_results['eval_loss']:.4f}")
    print(f"  Validation accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Validation F1: {eval_results['eval_f1']:.4f}")
    
    # Save the LoRA model
    lora_model.save_pretrained(output_dir)
    print(f"\nLoRA model saved to: {output_dir}")
    
    # Get predictions for confusion matrix
    predictions = trainer.predict(emotions_encoded["validation"])
    y_preds = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    plot_confusion_matrix(
        y_preds, y_true, labels,
        title="LoRA Fine-Tuning Confusion Matrix"
    )
    
    return lora_model, trainer


def load_lora_model(model_path, base_model_checkpoint, device):
    """
    Load a LoRA model from disk.
    
    Args:
        model_path: Path to saved LoRA adapters
        base_model_checkpoint: Original base model checkpoint
        device: torch device
        
    Returns:
        Loaded model with LoRA adapters
    """
    from peft import PeftModel
    
    print(f"Loading LoRA model from: {model_path}")
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_checkpoint,
        num_labels=NUM_LABELS
    ).to(device)
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model
