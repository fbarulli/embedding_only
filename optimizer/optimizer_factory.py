import torch.optim as optim
from typing import Dict, Any
import logging
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


class OptimizerFactory:
    """Factory for creating optimizer instances."""

    @staticmethod
    @log_function_entry_exit
    @debug_log_data
    @inject
    def create_optimizer(model_params, optimizer_config: Dict[str, Any], logger: logging.Logger = Provide[Container.logger]) -> optim.Optimizer:
        """Creates an optimizer instance based on the configuration.

        Args:
            model_params: Model parameters to be optimized (e.g., model.parameters()).
            optimizer_config: Configuration dictionary for the optimizer from YAML.
            logger: Injected logger instance.

        Returns:
            An instance of torch.optim.Optimizer.

        Raises:
            ValueError: If the optimizer name in the configuration is unknown.
        """
        optimizer_name: str = str(optimizer_config["name"]) # Direct access, casting to string
        learning_rate: float = float(optimizer_config["learning_rate"]) # Direct access, casting to float
        weight_decay: float = float(optimizer_config["weight_decay"]) # Direct access, casting to float

        logger.info(f"Creating optimizer: {optimizer_name}")
        logger.debug(f"  Learning Rate: {learning_rate}, Weight Decay: {weight_decay}")

        if optimizer_name == "adamw":
            optimizer = optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)
            logger.debug("AdamW optimizer created.")
            return optimizer
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9) # Example with momentum
            logger.debug("SGD optimizer created.")
            return optimizer
        else:
            logger.warning(f"Unknown optimizer name requested: {optimizer_name}")
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")