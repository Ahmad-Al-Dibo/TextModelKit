# MiniGPT

MiniGPT is an educational project focused on understanding and analyzing how GPT-style Transformer language models work in practice.

> Note: This repository contains the **project report, results, configurations, and documentation only**.  
> The source code is not included.

***

# Project Purpose

MiniGPT was created to explore the internal mechanics of small-scale Transformer-based language models.

The main objectives are:

* Understanding Transformer architecture from first principles
* Experimenting with GPT-style training pipelines
* Evaluating model performance and limitations
* Comparing different model configurations

This project is intended for learning and experimentation rather than production use.

***

# Repository Contents

This repository serves as a documentation and results showcase. It includes:

* Model architecture descriptions
* Training configurations
* Evaluation results and analysis
* Generated text examples
* Model comparisons
* Project insights and conclusions

It does not contain runnable code.

***

# Overview

MiniGPT follows a simplified language modeling pipeline:

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

The design prioritizes clarity and interpretability over complexity or performance.

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

| Model     | Description                               | Link                                                                     |
| --------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| MiniGPT   | Baseline educational GPT model            | <https://drive.google.com/file/d/1ksxeEqtXhsuF287R_gQBwqTKp9zQ41No/view> |
| MediumGPT | Improved model with better text coherence | <https://drive.google.com/file/d/1f938MeX0wDxJyksx042LjPyJ-zRdmyYl/view> |

***

# Training Configurations

Two configurations were used for experimentation:

## MiniGPT

* Small model focused on learning basic language patterns
* Short context window
* Faster training

## MediumGPT

* Increased model capacity
* Larger context window
* Improved generation quality

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

* Learns basic grammar and token relationships
* Struggles with maintaining long-range context

MediumGPT:

* Produces more coherent sentences
* Maintains context more effectively
* Shows improved semantic structure

Increasing model size and context leads to measurable improvements, though both models remain limited compared to large-scale systems.

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

* Small language models can learn grammatical patterns quickly
* They struggle with long-term coherence
* Context length strongly influences output quality
* Model depth and attention capacity improve performance

***

# Limitations

The project has several constraints:

* Small parameter sizes
* Limited training data
* Short context windows
* No instruction tuning
* No alignment techniques (e.g. RLHF)
* Limited reasoning capabilities

***

# Future Work

Potential improvements include:

* Larger models (5M–10M+ parameters)
* Longer context windows
* Instruction tuning support
* Improved attention mechanisms
* Better datasets
* Expanded evaluation benchmarks

***

# Summary

MiniGPT is a learning-focused project that demonstrates:

* How GPT-style models are structured
* How training affects model behavior
* What limitations small models have

This repository provides a clear and structured view of the project, making it suitable for students, developers, and researchers exploring language models.

***

# References

1. Vaswani et al. — Attention Is All You Need
2. Brown et al. — Language Models are Few-Shot Learners
3. SentencePiece tokenizer
4. PyTorch documentation
