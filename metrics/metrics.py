import torch
from typing import Tuple
import logging
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


@inject
@log_function_entry_exit
@debug_log_data
def calculate_perplexity(loss: torch.Tensor, logger: logging.Logger = Provide[Container.logger]) -> torch.Tensor:
    """Calculates perplexity from a loss tensor.

    Perplexity is a common metric for language models and can be a useful
    indicator of how well the model is predicting the next token/word.
    It's calculated as exp(loss).

    Args:
        loss: The loss tensor (typically cross-entropy loss).
        logger: Injected logger instance.

    Returns:
        Perplexity tensor (scalar). Returns torch.tensor(float('inf')) if loss is very high.
    """
    logger.debug(f"Input loss value for perplexity calculation: {loss.item()}")
    try:
        perplexity: torch.Tensor = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float('inf')) # Handle potential overflow for very high loss
        logger.warning("Perplexity calculation resulted in OverflowError, returning infinity.")
    logger.debug(f"Calculated perplexity: {perplexity.item()}")
    return perplexity


@inject
@log_function_entry_exit
@debug_log_data
def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor, logger: logging.Logger = Provide[Container.logger]) -> torch.Tensor:
    """Calculates accuracy for classification tasks.

    Args:
        predictions: Model predictions (logits or probabilities, shape: (batch_size, num_classes)).
        labels: Ground truth labels (shape: (batch_size,)).
        logger: Injected logger instance.

    Returns:
        Accuracy tensor (scalar percentage).
    """
    logger.debug(f"Input predictions shape: {predictions.shape}, labels shape: {labels.shape}")
    predicted_classes: torch.Tensor = torch.argmax(predictions, dim=-1) # Get predicted class indices
    correct_predictions: torch.Tensor = (predicted_classes == labels).sum() # Count correct predictions
    accuracy: torch.Tensor = (correct_predictions / labels.size(0)) * 100.0 # Calculate accuracy percentage
    logger.debug(f"Calculated accuracy: {accuracy.item():.2f}%")
    return accuracy


# You can add more metric functions here as needed,
# e.g., for regression tasks (MSE, MAE), or other classification metrics (F1-score, etc.)