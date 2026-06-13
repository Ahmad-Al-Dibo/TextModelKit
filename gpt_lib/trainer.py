"""
Trainer Module - Training loop and model management
"""

import time
import copy
import math
import torch
from pathlib import Path


class Trainer:
    """Model trainer with save/load functionality and generalization monitoring"""
    
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = torch.device(config.device)
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []
        self.val_token_accuracies = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state_dict = None
        self.current_epoch = 0
    
    def train(
        self,
        loader,
        epochs,
        val_loader=None,
        early_stopping_patience=None,
        min_delta=None,
        checkpoint_callback=None,
    ):
        """Train the model with optional validation and early stopping
        
        Args:
            loader: Training DataLoader
            epochs: Total target number of completed epochs
            val_loader: Validation DataLoader (optional)
            early_stopping_patience: Stop if val loss doesn't improve (optional)
            min_delta: Minimum validation loss improvement (optional)
            checkpoint_callback: Optional callable run after each completed epoch
        """
        if self.current_epoch >= epochs:
            print(
                f"\n[OK] Training already complete "
                f"({self.current_epoch}/{epochs} epochs)."
            )
            if val_loader is not None and getattr(self.config, "restore_best_model", True):
                self.restore_best_model()
            return

        self.model.train()
        train_start = time.time()
        if min_delta is None:
            min_delta = getattr(self.config, "early_stopping_min_delta", 0.0)
        
        if self.current_epoch > 0:
            print(f"\n[OK] Resuming training from epoch {self.current_epoch + 1}/{epochs}.")

        for epoch in range(self.current_epoch, epochs):
            epoch_start = time.time()
            total_loss = 0
            log_interval = getattr(self.config, "training_log_interval", 50)
            total_batches = len(loader)
            
            for batch_idx, (x, y) in enumerate(loader, start=1):
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(x)
                B, T, V = logits.shape
                
                loss = self.criterion(
                    logits.view(B * T, V),
                    y.view(B * T)
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                grad_clip = getattr(self.config, "grad_clip", None)
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                total_loss += loss.item()

                if log_interval and batch_idx % log_interval == 0:
                    elapsed = time.time() - epoch_start
                    batches_per_second = batch_idx / elapsed if elapsed else 0.0
                    remaining_batches = total_batches - batch_idx
                    eta_seconds = (
                        remaining_batches / batches_per_second
                        if batches_per_second
                        else 0.0
                    )
                    print(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx}/{total_batches} | "
                        f"Loss = {loss.item():.4f} | "
                        f"Elapsed = {elapsed:.1f}s | "
                        f"ETA = {eta_seconds:.1f}s",
                        flush=True,
                    )

            
            # Average training loss
            avg_train_loss = total_loss / len(loader)
            self.train_losses.append(avg_train_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Validation
            val_loss = None
            should_stop = False
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                val_loss = val_metrics["loss"]
                self.val_losses.append(val_loss)
                self.val_perplexities.append(val_metrics["perplexity"])
                self.val_token_accuracies.append(val_metrics["token_accuracy"])
                
                # Berechne generalisatie gap
                gen_gap = val_loss - avg_train_loss
                
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss = {avg_train_loss:.4f} | "
                    f"Val Loss = {val_loss:.4f} | "
                    f"Val PPL = {val_metrics['perplexity']:.2f} | "
                    f"Val Acc = {val_metrics['token_accuracy']*100:.2f}% | "
                    f"Gen Gap = {gen_gap:.4f} | "
                    f"Time = {epoch_time:.2f}s"
                )
                
                # Early stopping
                if early_stopping_patience is not None:
                    if self._check_early_stopping(val_loss, early_stopping_patience, min_delta):
                        print(f"\n[WARNING]  Early stopping at epoch {epoch+1}")
                        print(f"   Best validation loss: {self.best_val_loss:.4f}")
                        should_stop = True
                else:
                    self._update_best_validation(val_loss, min_delta)
            else:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss = {avg_train_loss:.4f} | "
                    f"Time = {epoch_time:.2f}s"
                )

            self.current_epoch = epoch + 1
            if checkpoint_callback is not None:
                checkpoint_callback(self)

            if should_stop:
                break
        
        total_train_time = time.time() - train_start
        hours = int(total_train_time // 3600)
        minutes = int((total_train_time % 3600) // 60)
        seconds = int(total_train_time % 60)
        if val_loader is not None and getattr(self.config, "restore_best_model", True):
            self.restore_best_model()
        print(f"\n[OK] Training completed in {hours}h {minutes}m {seconds}s")
    
    def _validate(self, val_loader):
        """Bereken validation loss, perplexity, en token accuracy."""
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        log_interval = getattr(self.config, "training_log_interval", 50)
        total_batches = len(val_loader)
        validation_start = time.time()
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader, start=1):
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(x)
                B, T, V = logits.shape
                
                loss = self.criterion(
                    logits.view(B * T, V),
                    y.view(B * T)
                )
                
                total_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                correct_tokens += (predictions == y).sum().item()
                total_tokens += y.numel()

                if log_interval and batch_idx % log_interval == 0:
                    elapsed = time.time() - validation_start
                    batches_per_second = batch_idx / elapsed if elapsed else 0.0
                    remaining_batches = total_batches - batch_idx
                    eta_seconds = (
                        remaining_batches / batches_per_second
                        if batches_per_second
                        else 0.0
                    )
                    print(
                        f"Validation | "
                        f"Batch {batch_idx}/{total_batches} | "
                        f"Elapsed = {elapsed:.1f}s | "
                        f"ETA = {eta_seconds:.1f}s",
                        flush=True,
                    )
         
        self.model.train()
        avg_loss = total_loss / len(val_loader)
        return {
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20)),
            "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        }
    
    def _check_early_stopping(self, val_loss, patience, min_delta=0.0):
        """Check if we should stop early based on validation loss
        
        Generalisatie eigenschap: Stop training als model niet meer verbetert
        """
        if self._update_best_validation(val_loss, min_delta):
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
        return False

    def _update_best_validation(self, val_loss, min_delta=0.0):
        """Track best validation loss and copy current weights."""
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            return True
        return False

    def restore_best_model(self):
        """Restore weights from the best validation loss, if available."""
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            print(f"[OK] Restored best validation model ({self.best_val_loss:.4f})")
    
    def get_generalization_gap(self):
        """Get generalization gap (val_loss - train_loss)"""
        if not self.train_losses or not self.val_losses:
            return None
        return self.val_losses[-1] - self.train_losses[-1]
    
    def get_train_history(self):
        """Get training history"""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_perplexities": self.val_perplexities,
            "val_token_accuracies": self.val_token_accuracies,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "current_epoch": self.current_epoch,
        }
    
    def save(
        self,
        model_path,
        vocab=None,
        stoi=None,
        itos=None,
        block_size=None,
        vocab_size=None,
        tokenizer_metadata=None,
    ):
        """Save model checkpoint"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "vocab": vocab,
            "stoi": stoi,
            "itos": itos,
            "tokenizer_metadata": tokenizer_metadata,
            "block_size": block_size,
            "vocab_size": vocab_size,
            "current_epoch": self.current_epoch,
            "patience_counter": self.patience_counter,
            "best_val_loss": self.best_val_loss,
            "best_model_state_dict": self.best_model_state_dict,
            "train_history": self.get_train_history(),
        }
        torch.save(checkpoint, model_path)
        print(f"[OK] Model saved to {model_path}")
    
    def load(self, model_path):
        """Load model checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore training history
        if "train_history" in checkpoint:
            history = checkpoint["train_history"]
            self.train_losses = history.get("train_losses", [])
            self.val_losses = history.get("val_losses", [])
            self.val_perplexities = history.get("val_perplexities", [])
            self.val_token_accuracies = history.get("val_token_accuracies", [])
            self.best_val_loss = history.get("best_val_loss", float('inf'))
            self.patience_counter = history.get("patience_counter", 0)
            self.current_epoch = history.get("current_epoch", len(self.train_losses))

        self.current_epoch = checkpoint.get("current_epoch", self.current_epoch)
        self.patience_counter = checkpoint.get("patience_counter", self.patience_counter)
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        self.best_model_state_dict = checkpoint.get("best_model_state_dict")
        
        print(f"[OK] Model loaded from {model_path} at epoch {self.current_epoch}")
        return checkpoint
    
    def exists(self, model_path):
        """Check if model checkpoint exists"""
        return Path(model_path).exists()
