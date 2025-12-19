"""
Configuration settings for the emotion classification project.
"""

import os

# Model configuration
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS = 6
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# LoRA configuration
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 1,
    "lora_dropout": 0.1,
    "target_modules": ["q_lin", "k_lin", "v_lin"],
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

# Device configuration
USE_GPU = True
