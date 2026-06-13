"""
Model construction and training pipeline helpers.
"""

import torch
import torch.nn as nn

from .model import MiniGPT
from .trainer import Trainer


def build_model(config, vocab_size):
    """Build a MiniGPT model from config and move it to the configured device."""
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        block_size=config.block_size,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    )
    return model.to(torch.device(config.device))


def build_trainer(model, config):
    """Create the optimizer, criterion, and Trainer for a model."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    return Trainer(model, optimizer, criterion, config)


def checkpoint_is_compatible(checkpoint, config, vocab_size, tokenizer=None):
    """Check whether a checkpoint can be reused for the current run."""
    checkpoint_config = checkpoint.get("config", {})
    same_vocab = checkpoint.get("vocab_size") == vocab_size
    same_shape = (
        checkpoint_config.get("embed_dim", config.embed_dim) == config.embed_dim
        and checkpoint_config.get("block_size", config.block_size) == config.block_size
        and checkpoint_config.get("num_blocks", config.num_blocks) == config.num_blocks
    )
    same_tokenizer = (
        checkpoint_config.get("tokenizer_type", "word")
        == getattr(config, "tokenizer_type", "word")
        and checkpoint_config.get("sentencepiece_model_type", "bpe")
        == getattr(config, "sentencepiece_model_type", "bpe")
    )
    if tokenizer is not None:
        checkpoint_stoi = checkpoint.get("stoi")
        checkpoint_vocab = checkpoint.get("vocab")
        if checkpoint_stoi is not None:
            same_tokenizer = same_tokenizer and checkpoint_stoi == tokenizer.stoi
        elif checkpoint_vocab is not None:
            same_tokenizer = same_tokenizer and checkpoint_vocab == tokenizer.vocab
        else:
            same_tokenizer = False

    same_training_strategy = (
        checkpoint_config.get("learning_rate", config.learning_rate) == config.learning_rate
        and checkpoint_config.get("weight_decay", config.weight_decay) == config.weight_decay
        and checkpoint_config.get("dropout", 0.0) == config.dropout
        and checkpoint_config.get("validation_split", config.validation_split)
        == config.validation_split
    )
    # return same_vocab and same_shape and same_tokenizer and same_training_strategy
    return same_vocab and same_shape and same_tokenizer and same_training_strategy


def load_compatible_checkpoint(trainer, config, vocab_size, tokenizer=None):
    """Load an existing checkpoint when it matches the current model setup."""
    if not trainer.exists(config.model_path):
        return False

    checkpoint = torch.load(config.model_path, map_location=torch.device(config.device))
    if checkpoint_is_compatible(checkpoint, config, vocab_size, tokenizer=tokenizer):
        trainer.load(config.model_path)
        return True

    print("[WARNING] Existing checkpoint is incompatible; retraining.")
    print(f"  Checkpoint vocab: {checkpoint.get('vocab_size')}, current vocab: {vocab_size}")
    print(f"  Checkpoint config: {checkpoint.get('config', {})}")
    if tokenizer is not None:
        print("  Tokenizer mapping differs or is missing; retraining is required.")
    return False


def create_epoch_checkpoint_callback(config, tokenizer, vocab_size):
    """Create a callback that saves the latest resumable checkpoint each epoch."""
    def checkpoint_callback(trainer):
        save_training_checkpoint(trainer, config, tokenizer, vocab_size)

    return checkpoint_callback


def save_training_checkpoint(trainer, config, tokenizer, vocab_size):
    """Save a trainer checkpoint with tokenizer metadata."""
    tokenizer_metadata = None
    if hasattr(tokenizer, "to_checkpoint"):
        tokenizer_metadata = tokenizer.to_checkpoint()

    trainer.save(
        config.model_path,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        block_size=config.block_size,
        vocab_size=vocab_size,
        tokenizer_metadata=tokenizer_metadata,
    )
