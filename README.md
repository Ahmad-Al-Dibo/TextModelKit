# MiniGPT

A compact GPT-based language model built as an educational project to understand and implement the core concepts behind modern Transformer architectures.

---

# Overview

MiniGPT is designed to demonstrate the fundamental building blocks of large language models, including:

* Tokenization
* Embeddings
* Self-Attention
* Transformer Blocks
* Next-Token Prediction
* Text Generation

The project prioritizes simplicity, readability, and educational value over production-level performance.

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
├─ Self-Attention
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

# Example Generation

### MiniGPT

Prompt:

```text
the half-ling book one in the fall
```

Output:

```text
the half-ling book one in the fall . fine , i guess that meant our country ? well , hold a blocky place in his mouthful cloud of claw marks ...
```

### MediumGPT

Prompt:

```text
the half-ling book one in the fall
```

Output:

```text
the half-ling book one in the fall . after her mother had found herself a dirty little taller than . and the air was determined to pull ...
```

MediumGPT maintains longer grammatical structures and stronger context retention than MiniGPT, indicating a more effective internal language representation.

---

# Evaluation Report

## Summary

Two language models were evaluated:

| Model     | Result                                 |
| --------- | -------------------------------------- |
| MiniGPT   | Learns basic grammatical patterns      |
| MediumGPT | Better coherence and context retention |

## Comparison

| Metric              | MiniGPT | MediumGPT |
| ------------------- | ------- | --------- |
| Grammar             | 4/10    | 6/10      |
| Coherence           | 3/10    | 5.5/10    |
| Context Retention   | 2/10    | 5/10      |
| Readability         | 4/10    | 6/10      |
| Narrative Structure | 2/10    | 5/10      |

## Analysis

MiniGPT primarily learns local word patterns and simple grammatical relationships.

MediumGPT demonstrates:

* Improved context retention
* Longer coherent sentence structures
* Stronger semantic relationships
* More stable text generation

Although both models remain far from modern production-scale language models, MediumGPT clearly shows that increased model capacity and training significantly improve generation quality.

---

# Project Structure

```text
MiniGPT/
│
├── app.py
├── model/
├── tokenizer/
├── data/
├── docs/
├── checkpoints/
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

# Usage

```bash
python app.py
```

---

# Roadmap

Planned improvements include:

* Larger context windows
* Multi-Head Attention
* SentencePiece BPE tokenization
* Larger training datasets
* Improved sampling strategies
* GPU optimization
* Inference API
* Model benchmarking
* Training visualization tools

---

# Purpose

This project is intended as a learning and experimentation platform for understanding how Transformer-based language models work internally. It is not designed to compete with large-scale commercial language models but to provide a transparent and extensible implementation for study and research.

---

# References

* Attention Is All You Need (Vaswani et al., 2017)
* GPT Language Models (OpenAI)
* PyTorch Documentation
* Transformer Architecture Research
