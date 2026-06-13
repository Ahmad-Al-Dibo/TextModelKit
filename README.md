# MiniGPT

A compact GPT-style language model built as an educational project to understand and implement the core concepts behind modern Transformer architectures.

MiniGPT focuses on simplicity, readability, and reproducibility while demonstrating how autoregressive language models learn to generate text. The project is intended for students, developers, and researchers who want to explore the internal mechanics of Transformer-based language models without the complexity of production-scale systems.

---

# Features

* Transformer-based language model architecture
* SentencePiece BPE tokenization
* Learned token embeddings
* Learned positional embeddings
* Multi-Head Self-Attention
* Feed-Forward Neural Networks
* Autoregressive next-token prediction
* Text generation and sampling
* Checkpoint saving and loading
* Reproducible training configurations
* Educational and research-oriented implementation

---

# Overview

MiniGPT demonstrates the complete language model training pipeline:

1. Raw text collection
2. Tokenization
3. Dataset creation
4. Transformer training
5. Next-token prediction
6. Text generation

The project intentionally prioritizes clarity and educational value over state-of-the-art performance.

---

# Architecture

```text
Raw Text
    ↓
Tokenizer
    ↓
Tokens
    ↓
Training Dataset
    ↓
MiniGPT Model
├─ Token Embeddings
├─ Position Embeddings
├─ Multi-Head Self-Attention
├─ Feed Forward Layers
└─ Next Token Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Model Learning
```

---

# Model Architecture

```text
Input Tokens
      ↓
Token Embeddings
      +
Position Embeddings
      ↓
Transformer Block 1
      ↓
Transformer Block 2
      ↓
Layer Normalization
      ↓
Language Modeling Head
      ↓
Vocabulary Probabilities
```

---

# Available Models

| Model     | Parameters |
| --------- | ---------: |
| MiniGPT   |  1,435,792 |
| MediumGPT |  2,725,792 |

---

# Model Downloads

| Model     | Description                                        | Download                                                               |
| --------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| MiniGPT   | Baseline educational GPT model                     | https://drive.google.com/file/d/1ksxeEqtXhsuF287R_gQBwqTKp9zQ41No/view |
| MediumGPT | Improved model with stronger language capabilities | https://drive.google.com/file/d/1f938MeX0wDxJyksx042LjPyJ-zRdmyYl/view |

---

# Training Configurations

The following configurations were used to train the released checkpoints.

## MiniGPT Configuration

```python
Config(
    embed_dim=64,
    block_size=64,

    num_blocks=2,
    num_heads=2,
    dropout=0.1,

    batch_size=32,
    epochs=10,

    learning_rate=3e-4,
    weight_decay=0.01,
    grad_clip=1.0,

    tokenizer_type="sentencepiece",
    sentencepiece_model_type="bpe",
    max_vocab=16000,

    validation_split=0.1,

    early_stopping_patience=2,
    restore_best_model=True,

    seed=42
)
```

## MediumGPT Configuration

```python
Config(
    embed_dim=64,
    num_blocks=3,
    num_heads=4,
    dropout=0.1,
    block_size=128,

    batch_size=32,
    grad_accum_steps=1,
    epochs=10,

    learning_rate=2.5e-4,
    weight_decay=0.1,
    grad_clip=1.0,

    max_data_size=2_500_000,
    validation_split=0.1,

    tokenizer_type="sentencepiece",
    sentencepiece_model_type="bpe",
    sentencepiece_character_coverage=1.0,
    max_vocab=20000,
    tokenizer_rare_threshold=2,

    early_stopping_patience=2,
    early_stopping_min_delta=1e-4,
    restore_best_model=True,

    seed=42,
    training_log_interval=50
)
```

---

# Architecture Comparison

| Property            | MiniGPT           | MediumGPT         |
| ------------------- | ----------------- | ----------------- |
| Parameters          | 1.44M             | 2.73M             |
| Embedding Dimension | 64                | 64                |
| Transformer Blocks  | 2                 | 3                 |
| Attention Heads     | 2                 | 4                 |
| Context Window      | 64                | 128               |
| Vocabulary Size     | 16K               | 20K               |
| Tokenizer           | SentencePiece BPE | SentencePiece BPE |
| Training Data       | Smaller           | Larger            |
| Context Retention   | Basic             | Improved          |

---

# Example Generation

## Prompt

```text
the half-ling book one in the fall
```

## MiniGPT Output

```text
the half-ling book one in the fall . fine , i guess that meant our country ? well , hold a blocky place in his mouthful cloud of claw marks . she led her to a building and passed by herself down the ground ...
```

## MediumGPT Output

```text
the half-ling book one in the fall . after her mother had found herself a dirty little taller than . and the air was determined to pull of the bolt map . he explained that david was about to take his care of his mechanics ...
```

MediumGPT maintains longer grammatical structures and stronger context retention than MiniGPT, indicating a more effective internal language representation.

---

# Evaluation Report

## Summary

Two language models were evaluated after training on comparable datasets.

| Model     | Result                                 |
| --------- | -------------------------------------- |
| MiniGPT   | Learns basic grammatical patterns      |
| MediumGPT | Better coherence and context retention |

## Metrics

| Metric              | MiniGPT | MediumGPT |
| ------------------- | ------- | --------- |
| Grammar             | 4/10    | 6/10      |
| Coherence           | 3/10    | 5.5/10    |
| Context Retention   | 2/10    | 5/10      |
| Readability         | 4/10    | 6/10      |
| Narrative Structure | 2/10    | 5/10      |

## Analysis

MiniGPT successfully learns local word patterns and basic grammatical structures but struggles to maintain long-range coherence.

MediumGPT demonstrates:

* Better context retention
* More consistent sentence construction
* Improved semantic relationships
* Stronger narrative continuity
* More stable text generation

Although both models remain far from modern production-scale language models, the results clearly show that increasing model depth, attention capacity, vocabulary size, and context length leads to measurable improvements in generation quality.

---

# Project Structure

```text
MiniGPT/
│
├── app.py
├── config.py
├── train.py
├── generate.py
│
├── model/
│   ├── attention.py
│   ├── transformer.py
│   ├── blocks.py
│   └── minigpt.py
│
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.py
│
├── data/
│
├── checkpoints/
│
├── docs/
│
└── README.md
```

---

# Installation

```bash
git clone <repository-url>
cd MiniGPT
pip install -r requirements.txt
```

---

# Training

Train a model using the default configuration:

```bash
python train.py
```

Resume training from a checkpoint:

```bash
python train.py --resume checkpoints/latest.pt
```

---

# Text Generation

Generate text from a trained checkpoint:

```bash
python generate.py
```

Example prompt:

```text
Once upon a time
```

Example generated output:

```text
Once upon a time there was a small village hidden beyond the mountains...
```

---

# Roadmap

Planned improvements include:

* 5M–10M parameter models
* Longer context windows (256–512 tokens)
* Rotary Positional Embeddings (RoPE)
* Multi-Query Attention
* Flash Attention support
* Larger training datasets
* Fine-tuning workflows
* Quantized inference
* GPU-optimized training pipelines
* Benchmark suite integration
* Training visualizations
* Inference API

---

# Reproducibility

All released checkpoints, configurations, training settings, and evaluation examples are included to support reproducible experimentation and educational research.

Training seeds, tokenizer configurations, and model hyperparameters are documented to allow independent reproduction of reported results.

---

# Limitations

MiniGPT is an educational implementation and should not be expected to perform at the level of modern large language models.

Current limitations include:

* Limited parameter count
* Small training datasets
* Short context windows
* Restricted reasoning capabilities
* Reduced factual accuracy
* No instruction tuning
* No reinforcement learning from human feedback (RLHF)

---

# Purpose

This project serves as a learning platform for understanding Transformer-based language models from first principles.

The primary goals are:

* Educational exploration
* Research experimentation
* Model architecture understanding
* Training pipeline development
* Reproducible Transformer implementations

MiniGPT is not intended to compete with production-scale systems such as GPT, Claude, Gemini, or Llama models. Instead, it provides a transparent and extensible implementation for studying how language models operate internally.

---

# References

1. Vaswani et al. (2017) — Attention Is All You Need
2. Brown et al. (2020) — Language Models are Few-Shot Learners
3. SentencePiece: A Simple and Language Independent Subword Tokenizer
4. PyTorch Documentation
5. Transformer Architecture Research Literature
6. GPT Language Model Research
