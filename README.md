***

# MiniGPT

MiniGPT is a GPT-style language modeling project focused on building, analyzing, and evolving Transformer-based systems.

This repository presents the **project report, results, configurations, and design decisions** behind MiniGPT.

> Note: This repository currently contains documentation and results.  
> The implementation is being developed into a **full reusable library** and will be added here.

***

# Project Goal

The goal of MiniGPT is to evolve from an experimental project into a **real, modular language model library**.

This includes:

* Building a clean and extensible GPT-like framework
* Supporting training, fine-tuning, and evaluation workflows
* Enabling instruction-tuning and conversational models
* Providing a structured and reusable codebase

This repository documents the current progress and results of that process.

***

# Current Repository Contents

This repository contains:

* Model architecture descriptions
* Training configurations
* Evaluation results and comparisons
* Generated text examples
* Project analysis and insights

It is currently focused on **documentation and reporting**, while the codebase is being prepared as a standalone library.

***

# Planned Library Features

The upcoming MiniGPT library will include:

* Transformer-based model architecture
* Pretraining and fine-tuning support
* Instruction-tuning pipeline
* Modular trainer system
* Dataset and tokenization utilities
* Evaluation and diagnostics tools
* Config-driven training workflows
* Reusable API for experimentation

***

# Overview

MiniGPT follows a standard autoregressive language modeling pipeline:

```text
Raw Text
    ↓
Tokenization (BPE)
    ↓
Dataset Creation
    ↓
Transformer Training
    ↓
Next Token Prediction
    ↓
Text Generation
```

***

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
├─ Feed Forward Networks
└─ Output Layer
```

***

# Model Variants

| Model     | Parameters |
| --------- | ---------: |
| MiniGPT   |      1.44M |
| MediumGPT |      2.73M |

***

# Model Downloads

| Model     | Description                          | Link                                                                     |
| --------- | ------------------------------------ | ------------------------------------------------------------------------ |
| MiniGPT   | Baseline small GPT model             | <https://drive.google.com/file/d/1ksxeEqtXhsuF287R_gQBwqTKp9zQ41No/view> |
| MediumGPT | Improved model with better coherence | <https://drive.google.com/file/d/1f938MeX0wDxJyksx042LjPyJ-zRdmyYl/view> |

***

# Training Configurations

Two configurations were used during experimentation:

## MiniGPT

* Small model for initial experiments
* Short context window
* Focus on basic language patterns

## MediumGPT

* Increased model capacity
* Larger context window
* Improved text generation quality

***

# Evaluation Results

## Performance Comparison

| Metric              | MiniGPT | MediumGPT |
| ------------------- | ------: | --------: |
| Grammar             |    4/10 |      6/10 |
| Coherence           |    3/10 |    5.5/10 |
| Context Retention   |    2/10 |      5/10 |
| Readability         |    4/10 |      6/10 |
| Narrative Structure |    2/10 |      5/10 |

***

## Key Findings

MiniGPT:

* Learns basic grammatical patterns
* Limited long-range context understanding

MediumGPT:

* Better sentence structure
* Improved context retention
* More coherent outputs

Increasing model size and context improves performance but remains far from large-scale systems.

***

# Example Output

**Prompt**

```
the half-ling book one in the fall
```

**MiniGPT**

```
the half-ling book one in the fall . fine , i guess that meant our country ...
```

**MediumGPT**

```
the half-ling book one in the fall . after her mother had found herself ...
```

***

# Observations

* Small models learn patterns quickly
* Context length strongly affects quality
* Model depth improves coherence
* Scaling increases performance but does not solve all limitations

***

# Limitations

* Small parameter sizes
* Limited datasets
* Short context windows
* No instruction tuning implemented yet
* Limited reasoning capabilities

***

# Future Development

MiniGPT is actively being developed into a full library. Planned next steps:

* Modular codebase release
* Complete training and fine-tuning pipeline
* Instruction-tuning support
* Improved training stability
* Better evaluation tooling
* API interface for experimentation

***

# Summary

MiniGPT is a developing project aimed at building a structured GPT-style library.

This repository provides:

* Design and architecture decisions
* Experimental results
* Model evaluations

The next stage of the project is focused on delivering a **complete, reusable MiniGPT library**.

***

# References

1. Vaswani et al. — Attention Is All You Need
2. Brown et al. — Language Models are Few-Shot Learners
3. SentencePiece tokenizer
4. PyTorch documentation

***
