#!/usr/bin/env python3
"""
Emotion Classification with Transformers
=========================================

Main entry point for the emotion classification pipeline.

This script demonstrates three approaches to text classification:
1. Feature Extraction: Using pretrained hidden states with Logistic Regression
2. Full Fine-Tuning: Training all transformer parameters
3. LoRA Fine-Tuning: Parameter-efficient fine-tuning

Usage:
    python main.py                    # Run all approaches
    python main.py --approach fe      # Feature extraction only
    python main.py --approach ft      # Fine-tuning only
    python main.py --approach lora    # LoRA fine-tuning only
    python main.py --skip-eda         # Skip exploratory data analysis

Author: Your Name
Date: 2024
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CHECKPOINT, MODELS_DIR, RESULTS_DIR, DATA_CACHE_DIR
from src.utils import get_device, create_directories, print_section_header
from src.data_loader import (
    load_emotions_dataset,
    explore_dataset,
    tokenize_dataset,
    get_labels,
    demonstrate_tokenization
)
from src.feature_extraction import run_feature_extraction
from src.fine_tuning import fine_tune_model
from src.lora_fine_tuning import fine_tune_lora


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Emotion Classification with Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run complete pipeline
  python main.py --approach fe      Feature extraction only
  python main.py --approach ft      Fine-tuning only  
  python main.py --approach lora    LoRA fine-tuning only
  python main.py --skip-eda         Skip data exploration
        """
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["all", "fe", "ft", "lora"],
        default="all",
        help="Which approach to run (default: all)"
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis"
    )
    parser.add_argument(
        "--demo-tokenization",
        action="store_true",
        help="Run tokenization demonstration"
    )
    return parser.parse_args()


def print_banner():
    """Print project banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     EMOTION CLASSIFICATION WITH TRANSFORMERS                  ║
    ║                                                               ║
    ║     Detecting emotions in text using DistilBERT               ║
    ║     • Feature Extraction  • Fine-Tuning  • LoRA               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(results):
    """Print summary of results."""
    print_section_header("RESULTS SUMMARY")
    
    print("\nApproach Comparison:")
    print("-" * 50)
    print(f"{'Approach':<25} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 50)
    
    for approach, metrics in results.items():
        if metrics:
            acc = metrics.get('accuracy', 'N/A')
            f1 = metrics.get('f1', 'N/A')
            if isinstance(acc, float):
                print(f"{approach:<25} {acc:<12.4f} {f1:<12.4f}")
            else:
                print(f"{approach:<25} {acc:<12} {f1:<12}")
    
    print("-" * 50)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print_banner()
    
    # Setup
    print_section_header("SETUP")
    device = get_device()
    create_directories([MODELS_DIR, RESULTS_DIR, DATA_CACHE_DIR])
    
    # Store results
    results = {}
    
    # Optional: Demonstrate tokenization
    if args.demo_tokenization:
        demonstrate_tokenization(MODEL_CHECKPOINT)
    
    # Step 1: Load data
    print_section_header("STEP 1: LOADING DATA")
    dataset = load_emotions_dataset()
    
    # Step 2: Exploratory Data Analysis (optional)
    if not args.skip_eda:
        print_section_header("STEP 2: EXPLORATORY DATA ANALYSIS")
        dataset = explore_dataset(dataset)
    else:
        print("\nSkipping EDA (--skip-eda flag set)")
    
    # Step 3: Tokenize dataset
    print_section_header("STEP 3: TOKENIZING DATA")
    emotions_encoded, tokenizer = tokenize_dataset(dataset, MODEL_CHECKPOINT)
    labels = get_labels(dataset)
    print(f"\nEmotion labels: {labels}")
    
    # Step 4: Run selected approach(es)
    
    # Feature Extraction
    if args.approach in ["all", "fe"]:
        print_section_header("STEP 4A: FEATURE EXTRACTION")
        try:
            _, fe_metrics = run_feature_extraction(
                emotions_encoded, MODEL_CHECKPOINT, labels, device
            )
            results["Feature Extraction"] = fe_metrics
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            results["Feature Extraction"] = None
    
    # Full Fine-Tuning
    if args.approach in ["all", "ft"]:
        print_section_header("STEP 4B: FULL FINE-TUNING")
        try:
            _, ft_trainer = fine_tune_model(
                emotions_encoded, MODEL_CHECKPOINT, labels, device
            )
            ft_eval = ft_trainer.evaluate()
            results["Full Fine-Tuning"] = {
                "accuracy": ft_eval["eval_accuracy"],
                "f1": ft_eval["eval_f1"]
            }
        except Exception as e:
            print(f"Error in fine-tuning: {e}")
            results["Full Fine-Tuning"] = None
    
    # LoRA Fine-Tuning
    if args.approach in ["all", "lora"]:
        print_section_header("STEP 4C: LoRA FINE-TUNING")
        try:
            _, lora_trainer = fine_tune_lora(
                emotions_encoded, MODEL_CHECKPOINT, labels, device
            )
            lora_eval = lora_trainer.evaluate()
            results["LoRA Fine-Tuning"] = {
                "accuracy": lora_eval["eval_accuracy"],
                "f1": lora_eval["eval_f1"]
            }
        except Exception as e:
            print(f"Error in LoRA fine-tuning: {e}")
            results["LoRA Fine-Tuning"] = None
    
    # Print summary
    print_summary(results)
    
    print("\n✓ Pipeline completed successfully!")
    return results


if __name__ == "__main__":
    main()
