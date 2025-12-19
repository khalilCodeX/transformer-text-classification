# Technical Architecture

This document provides an in-depth explanation of the emotion classification system, including key concepts, algorithms, and implementation details.

## Table of Contents

1. [System Overview](#system-overview)
2. [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
3. [NLTK Tokenization](#nltk-tokenization)
4. [Transformer Architecture](#transformer-architecture)
5. [Training Approaches](#training-approaches)
6. [Performance Analysis](#performance-analysis)

---

## System Overview

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Load Data   │───▶│   Explore    │───▶│   Tokenize   │          │
│  │  (HF Hub)    │    │   (EDA)      │    │  (DistilBERT)│          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                 │                   │
└─────────────────────────────────────────────────┼───────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
         ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
         │ Feature Extract  │         │  Fine-Tuning     │         │  LoRA Tuning     │
         │                  │         │                  │         │                  │
         │ • Frozen model   │         │ • All params     │         │ • 2% params      │
         │ • LogReg head    │         │ • Full training  │         │ • Low-rank adapt │
         │ • ~63% F1        │         │ • ~92% F1        │         │ • ~92% F1        │
         └──────────────────┘         └──────────────────┘         └──────────────────┘
```

### Module Dependencies

```
main.py
    ├── config.py (configuration)
    └── src/
        ├── utils.py (shared utilities)
        ├── data_loader.py (data handling)
        ├── feature_extraction.py (approach 1)
        ├── fine_tuning.py (approach 2)
        └── lora_fine_tuning.py (approach 3)
```

---

## LoRA: Low-Rank Adaptation

### Motivation

Traditional fine-tuning updates all model parameters, which is computationally expensive:

| Model | Parameters | Memory (FP32) |
|-------|------------|---------------|
| DistilBERT | 66M | 264 MB |
| BERT Base | 110M | 440 MB |
| BERT Large | 340M | 1.36 GB |
| GPT-3 | 175B | 700 GB |

**Challenge**: As models grow larger, fine-tuning becomes impractical due to:
- Memory constraints
- Training time
- Storage for multiple task-specific models

### Mathematical Foundation

LoRA decomposes weight updates into low-rank matrices:

```
Original:     h = Wx
Full FT:      h = (W + ΔW)x
LoRA:         h = Wx + BAx
```

Where:
- `W` ∈ ℝ^(d×k): Original pretrained weights (frozen)
- `ΔW` ∈ ℝ^(d×k): Full weight update (traditional fine-tuning)
- `B` ∈ ℝ^(d×r): Down-projection matrix (trainable)
- `A` ∈ ℝ^(r×k): Up-projection matrix (trainable)
- `r` << min(d, k): Rank (typically 8-64)

### Parameter Efficiency

For a weight matrix W with dimensions d × k:

```
Full fine-tuning parameters: d × k
LoRA parameters: r × (d + k)
Reduction ratio: r × (d + k) / (d × k) ≈ r/min(d,k)
```

**Example** (DistilBERT attention layer, d=k=768, r=32):
- Full: 768 × 768 = 589,824 parameters
- LoRA: 32 × (768 + 768) = 49,152 parameters
- **Reduction: 91.7%**

### Implementation Details

```python
# LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,                    # Rank of decomposition
    lora_alpha=1,            # Scaling factor
    lora_dropout=0.1,        # Dropout for regularization
    target_modules=[         # Which layers to adapt
        "q_lin",             # Query projection
        "k_lin",             # Key projection
        "v_lin"              # Value projection
    ],
    bias="none"              # Don't train bias terms
)
```

### Initialization Strategy

```
A: Random Gaussian initialization ~ N(0, σ²)
B: Zero initialization
```

This ensures `ΔW = BA ≈ 0` at initialization, preserving pretrained behavior.

### Scaling Factor (α/r)

The forward pass applies scaling:

```
h = Wx + (α/r) × BAx
```

- Default: α = r (scaling = 1)
- Higher α: Stronger adaptation
- Lower α: Preserve pretrained behavior

### Benefits Summary

| Aspect | Full Fine-Tuning | LoRA |
|--------|-----------------|------|
| Trainable params | 100% | 1-5% |
| Memory footprint | High | Low |
| Training speed | Baseline | ~20% faster |
| Multi-task storage | N copies | 1 base + N adapters |
| Performance | Best | Comparable |

---

## NLTK Tokenization

### Overview

Natural Language Toolkit (NLTK) provides classical tokenization approaches used for comparison and understanding.

### Tokenization Levels

#### 1. Character-Level

```python
text = "Hello World"
tokens = list(text)
# ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd']
```

**Properties**:
- Vocabulary: ~100 characters
- Sequence length: Very long
- No OOV (out-of-vocabulary) issues
- Poor semantic granularity

#### 2. Word-Level (NLTK)

```python
from nltk import word_tokenize

text = "I'm feeling great! 😊"
tokens = word_tokenize(text)
# ['I', "'m", 'feeling', 'great', '!', '😊']
```

**Properties**:
- Vocabulary: 50,000-100,000 words
- Handles contractions
- OOV issues with rare words
- Language-specific rules

#### 3. Subword-Level (Transformers)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text = "unbelievable"
tokens = tokenizer.tokenize(text)
# ['un', '##be', '##lie', '##va', '##ble']
```

**Properties**:
- Vocabulary: 30,000-50,000 subwords
- Handles OOV gracefully
- `##` indicates word continuation
- Learned from data (BPE/WordPiece)

### Comparison

| Aspect | Character | Word | Subword |
|--------|-----------|------|---------|
| Vocabulary | ~100 | 50K+ | 30K |
| OOV handling | None | Frequent | Rare |
| Sequence length | Very long | Medium | Medium |
| Semantic meaning | Poor | Good | Good |
| Morphology | Implicit | Lost | Preserved |

### Special Tokens in Transformers

```
[CLS]   - Classification token (start)
[SEP]   - Separator (between segments)
[PAD]   - Padding (for batching)
[UNK]   - Unknown token
[MASK]  - Masked token (for MLM training)
```

---

## Transformer Architecture

### DistilBERT Overview

DistilBERT is a distilled version of BERT:

| Property | BERT Base | DistilBERT |
|----------|-----------|------------|
| Layers | 12 | 6 |
| Hidden size | 768 | 768 |
| Attention heads | 12 | 12 |
| Parameters | 110M | 66M |
| Speed | 1x | 1.6x |
| Performance | 100% | 97% |

### Architecture Diagram

```
Input: "I love this movie!"
        │
        ▼
┌─────────────────────────────────────┐
│         Tokenization                │
│   [CLS] i love this movie ! [SEP]   │
│   [101, 1045, 2293, 2023, ...]     │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│      Token + Position Embedding     │
│         (768-dim vectors)           │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│     Transformer Layer × 6           │
│  ┌─────────────────────────────┐   │
│  │  Multi-Head Self-Attention  │   │
│  │  (12 heads, 64 dim each)    │   │
│  └─────────────────────────────┘   │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐   │
│  │     Add & Layer Norm        │   │
│  └─────────────────────────────┘   │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐   │
│  │    Feed-Forward Network     │   │
│  │    (768 → 3072 → 768)       │   │
│  └─────────────────────────────┘   │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐   │
│  │     Add & Layer Norm        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│    [CLS] Token Representation       │
│         (768-dim vector)            │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│      Classification Head            │
│    Linear(768 → 6) + Softmax        │
└─────────────────────────────────────┘
        │
        ▼
    Output: [0.01, 0.95, 0.02, 0.01, 0.00, 0.01]
            (probabilities for each emotion)
```

### Self-Attention Mechanism

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q = XW_Q (queries)
- K = XW_K (keys)
- V = XW_V (values)
- d_k = 64 (key dimension)
```

### Multi-Head Attention

```python
# 12 parallel attention heads
heads = [Attention(Q_i, K_i, V_i) for i in range(12)]
output = Concat(heads) @ W_O
```

### Feed-Forward Network

```python
FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2
# W_1: (768, 3072)
# W_2: (3072, 768)
```

---

## Training Approaches

### Approach 1: Feature Extraction

```
┌──────────────────────────────┐
│     Pretrained DistilBERT    │
│         (FROZEN)             │
│                              │
│  Input → ... → [CLS] hidden  │
└──────────────────────────────┘
              │
              ▼
       768-dim vector
              │
              ▼
┌──────────────────────────────┐
│    Logistic Regression       │
│       (TRAINABLE)            │
│                              │
│   768 → 6 (emotions)         │
└──────────────────────────────┘
```

**Advantages**:
- Fast training (minutes)
- Minimal compute requirements
- Good baseline

**Disadvantages**:
- Limited performance (~63% F1)
- No task-specific adaptation

### Approach 2: Full Fine-Tuning

```
┌──────────────────────────────┐
│     Pretrained DistilBERT    │
│       (ALL TRAINABLE)        │
│                              │
│  66M parameters updated      │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│    Classification Head       │
│       (TRAINABLE)            │
│                              │
│   768 → 6 (emotions)         │
└──────────────────────────────┘
```

**Advantages**:
- Best performance (~92% F1)
- Full model adaptation

**Disadvantages**:
- High compute cost
- Risk of catastrophic forgetting
- Large model storage

### Approach 3: LoRA Fine-Tuning

```
┌──────────────────────────────────────────┐
│         Pretrained DistilBERT            │
│              (FROZEN)                    │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │   Attention Layers with LoRA       │  │
│  │                                    │  │
│  │   W_q ────────────────────────     │  │
│  │      └─► B_q × A_q (trainable)     │  │
│  │                                    │  │
│  │   W_k ────────────────────────     │  │
│  │      └─► B_k × A_k (trainable)     │  │
│  │                                    │  │
│  │   W_v ────────────────────────     │  │
│  │      └─► B_v × A_v (trainable)     │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│    Classification Head       │
│       (TRAINABLE)            │
└──────────────────────────────┘
```

**Advantages**:
- Near full fine-tuning performance (~92% F1)
- 98% fewer trainable parameters
- Multiple task adapters for one model

**Disadvantages**:
- Slightly more complex setup
- Marginal performance gap

---

## Performance Analysis

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Forward pass | O(n² × d) | O(n × d) |
| Attention | O(n² × d) | O(n²) |
| FFN | O(n × d²) | O(d²) |
| LoRA forward | O(n × r × d) | O(r × d) |

Where: n = sequence length, d = hidden dim, r = LoRA rank

### Memory Requirements

```
Feature Extraction:
  Model:      268 MB (frozen)
  Embeddings: 16k × 768 × 4 = 49 MB
  Total:      ~320 MB

Full Fine-Tuning (batch=64):
  Model:      268 MB
  Gradients:  268 MB
  Optimizer:  536 MB (Adam: 2× params)
  Activations: 64 × 128 × 768 × 4 = 25 MB
  Total:      ~1.1 GB

LoRA (batch=64):
  Base model: 268 MB (frozen)
  LoRA params: 10 MB
  Gradients:  10 MB
  Optimizer:  20 MB
  Activations: 25 MB
  Total:      ~333 MB
```

### Training Time Comparison

| Approach | CPU Time | GPU Time | Speedup |
|----------|----------|----------|---------|
| Feature Extraction | 5 min | 1 min | 5× |
| Fine-Tuning | 3 hours | 15 min | 12× |
| LoRA | 2.5 hours | 12 min | 12.5× |

### Accuracy vs Efficiency Trade-off

```
F1 Score
   │
0.95│                    ● Full FT
   │               ● LoRA
0.90│
   │
0.85│
   │
0.80│
   │
0.65│ ● Feature Extraction
   │
0.60└─────────────────────────────────
     1K    10K   100K   1M   66M
              Trainable Parameters
```

---

## Conclusion

This architecture demonstrates:

1. **LoRA** enables efficient fine-tuning with minimal performance loss
2. **Subword tokenization** (BPE/WordPiece) handles OOV better than word tokenization
3. **Transformer attention** captures contextual relationships effectively
4. **Trade-offs exist** between compute, memory, and performance

The modular design allows easy experimentation with different approaches while maintaining production-quality code.

---

## References

1. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
2. Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT"
3. Vaswani, A., et al. (2017). "Attention Is All You Need"
4. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
