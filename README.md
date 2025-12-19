# Emotion Classification with Transformers

A professional implementation of emotion classification using state-of-the-art transformer models. This project demonstrates three distinct approaches to text classification: feature extraction, full fine-tuning, and parameter-efficient LoRA fine-tuning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This repository contains a complete pipeline for classifying emotions in text using DistilBERT. The project showcases modern NLP best practices including:

- **Data Processing**: Loading, exploring, and tokenizing datasets
- **Feature Extraction**: Leveraging pretrained model representations
- **Model Fine-Tuning**: Full and parameter-efficient training approaches
- **Evaluation**: Comprehensive metrics and visualizations

## Features

### 📊 Three Training Approaches

| Approach | Accuracy | F1 Score | Trainable Params | Training Time |
|----------|----------|----------|------------------|---------------|
| Feature Extraction | ~63% | ~0.63 | 0% (frozen) | ~5 min |
| Full Fine-Tuning | ~92% | ~0.92 | 100% | ~15 min (GPU) |
| LoRA Fine-Tuning | ~92% | ~0.92 | ~2% | ~12 min (GPU) |

### 🎯 Emotion Classes

The model classifies text into six emotion categories:
- 😢 Sadness
- 😊 Joy
- ❤️ Love
- 😠 Anger
- 😨 Fear
- 😮 Surprise

## Project Structure

```
emotion-classification/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading and exploration
│   ├── feature_extraction.py   # Feature extraction approach
│   ├── fine_tuning.py          # Full fine-tuning approach
│   ├── lora_fine_tuning.py     # LoRA fine-tuning approach
│   └── utils.py                # Utility functions
├── config.py                   # Configuration settings
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── gpu_setup.sh               # GPU setup script
├── README.md                  # This file
├── ARCHITECTURE.md            # Technical documentation
└── .gitignore                 # Git ignore rules
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-classification.git
cd emotion-classification

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab')"
```

### GPU Setup (Optional but Recommended)

For NVIDIA GPUs, run the setup script:

```bash
chmod +x gpu_setup.sh
./gpu_setup.sh
```

Or manually install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Running the Pipeline

```bash
# Run complete pipeline (all approaches)
python main.py

# Run specific approach
python main.py --approach fe      # Feature extraction only
python main.py --approach ft      # Fine-tuning only
python main.py --approach lora    # LoRA fine-tuning only

# Skip exploratory data analysis
python main.py --skip-eda

# Show tokenization demonstration
python main.py --demo-tokenization
```

## Configuration

Edit `config.py` to customize training parameters:

```python
# Model
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS = 6

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5

# LoRA
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 1,
    "lora_dropout": 0.1,
    "target_modules": ["q_lin", "k_lin", "v_lin"],
}
```

## Usage Examples

### Using a Trained Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(
    "models/distilbert-finetuned-emotion"
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Predict emotion
text = "I'm so happy today!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()

emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
print(f"Predicted emotion: {emotions[prediction]}")  # Output: joy
```

### Using LoRA Model

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

# Load base model + LoRA adapters
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6
)
model = PeftModel.from_pretrained(base_model, "models/lora-distilbert-finetuned-emotion")
```

## Results

### Class Distribution
![Class Distribution](results/class_distribution.png)

### Confusion Matrices

| Feature Extraction | Fine-Tuning | LoRA |
|-------------------|-------------|------|
| ![FE](results/feature_extraction_confusion_matrix.png) | ![FT](results/fine-tuning_confusion_matrix.png) | ![LoRA](results/lora_fine-tuning_confusion_matrix.png) |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | - | NVIDIA 8GB+ VRAM |
| Storage | 5 GB | 10 GB |

### GPU Memory Usage

| Approach | GPU Memory |
|----------|------------|
| Feature Extraction | ~4 GB |
| Fine-Tuning (batch=64) | ~8 GB |
| LoRA (batch=64) | ~6 GB |

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in config.py
BATCH_SIZE = 32  # or 16

# Or enable gradient accumulation in fine_tuning.py
gradient_accumulation_steps = 2
```

### Slow Training on CPU

- Use `--approach fe` for feature extraction (fastest)
- Reduce `NUM_EPOCHS` to 1
- Use a smaller model like `distilbert-base-uncased`

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify installation
python -c "import torch; import transformers; import peft; print('OK')"
```

## Technical Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed explanations of:
- LoRA (Low-Rank Adaptation)
- NLTK Tokenization
- Transformer Architecture
- System Design

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install dev dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ main.py config.py

# Lint
flake8 src/ main.py config.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers and Datasets libraries
- [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) for the dataset
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation

## Citation

```bibtex
@software{emotion_classification_2024,
  title={Emotion Classification with Transformers},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/emotion-classification}
}
```

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

⭐ If you find this project useful, please consider giving it a star!
