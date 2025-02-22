from typing import Dict, Any
import torch.nn as nn
from dependency_injector.wiring import inject, Provide
import logging

from ..architectures.transformer import TransformerEmbeddingModel
from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    @log_function_entry_exit
    @debug_log_data
    @inject
    def create_model(model_config: Dict[str, Any], logger: logging.Logger = Provide[Container.logger]) -> nn.Module:
        """Creates a model instance based on the configuration.

        Args:
            model_config: Configuration dictionary for the model.
            logger: Injected logger instance.

        Returns:
            An instance of torch.nn.Module (the model).

        Raises:
            ValueError: If the model name in the configuration is unknown.
        """
        model_name = model_config.get("name")
        logger.info(f"Creating model: {model_name}")

        if model_name == "transformer_embedding":
            logger.debug("Creating TransformerEmbeddingModel...")
            model = TransformerEmbeddingModel(model_config)
            logger.debug("TransformerEmbeddingModel created successfully.")
            return model
        else:
            logger.warning(f"Unknown model name requested: {model_name}")
            raise ValueError(f"Unknown model name: {model_name}")