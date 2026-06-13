"""
Mini GPT Library - Modular Language Model Framework
Herbruikbare componenten voor LLM training en inference
"""

from .tokenizer import SentencePieceTokenizer, Tokenizer
from .dataset import TextDataset, create_dataloader
from .data import DataBundle, load_tokens, prepare_data, read_tokens, split_train_val
from .model import MiniGPT
from .trainer import Trainer
from .config import Config
from .generator import Generator
from .inference import DEFAULT_MODEL_PATH, LoadedModel, load_model, predict
from .pipeline import (
    build_model,
    build_trainer,
    checkpoint_is_compatible,
    create_epoch_checkpoint_callback,
    load_compatible_checkpoint,
    save_training_checkpoint,
)
from .diagnostics import (
    build_training_diagnostics_report,
    ConceptBenchmarkCase,
    ConceptBenchmarkResult,
    DEFAULT_CONCEPT_BENCHMARKS,
    LongContextResult,
    TopKPrediction,
    format_concept_benchmark_report,
    format_long_context_results,
    format_tokenizer_coverage_report,
    format_top_k_predictions,
    predict_top_k,
    run_long_context_evaluation,
    score_concept_relationships,
)
from .utils import format_duration
from .regularization import (
    L1Regularization,
    L2Regularization,
    EarlyStopping,
    LearningRateScheduler,
    GeneralizationMonitor,
    MixupAugmentation,
    LabelSmoothing,
)

__version__ = "1.0.0"
__all__ = [
    "Tokenizer",
    "SentencePieceTokenizer",
    "TextDataset",
    "create_dataloader",
    "DataBundle",
    "load_tokens",
    "prepare_data",
    "read_tokens",
    "split_train_val",
    "MiniGPT",
    "Trainer",
    "Config",
    "Generator",
    "DEFAULT_MODEL_PATH",
    "LoadedModel",
    "load_model",
    "predict",
    "build_model",
    "build_trainer",
    "checkpoint_is_compatible",
    "create_epoch_checkpoint_callback",
    "load_compatible_checkpoint",
    "save_training_checkpoint",
    "build_training_diagnostics_report",
    "ConceptBenchmarkCase",
    "ConceptBenchmarkResult",
    "DEFAULT_CONCEPT_BENCHMARKS",
    "LongContextResult",
    "TopKPrediction",
    "format_concept_benchmark_report",
    "format_long_context_results",
    "format_tokenizer_coverage_report",
    "format_top_k_predictions",
    "predict_top_k",
    "run_long_context_evaluation",
    "score_concept_relationships",
    "format_duration",
    # Regularization & Generalization
    "L1Regularization",
    "L2Regularization",
    "EarlyStopping",
    "LearningRateScheduler",
    "GeneralizationMonitor",
    "MixupAugmentation",
    "LabelSmoothing",
]
