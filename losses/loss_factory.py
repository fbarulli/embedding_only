from typing import Dict, Any, Callable
import torch.nn as nn
import logging
from dependency_injector.wiring import inject, Provide

from .loss_functions import InfoNCELoss
from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


class LossFunctionFactory:
    """Factory for creating loss function instances."""

    @staticmethod
    @log_function_entry_exit
    @debug_log_data
    @inject
    def create_loss_function(loss_config: Dict[str, Any], logger: logging.Logger = Provide[Container.logger]) -> nn.Module:
        """Creates a loss function instance based on the configuration.

        Args:
            loss_config: Configuration dictionary for the loss function from YAML.
            logger: Injected logger instance.

        Returns:
            An instance of torch.nn.Module representing the loss function.

        Raises:
            ValueError: If the loss function name in the configuration is unknown.
        """
        loss_name: str = str(loss_config["name"]) # Direct access, casting to string
        logger.info(f"Creating loss function: {loss_name}")

        if loss_name == "info_nce":
            temperature: float = float(loss_config["temperature"]) # Direct access, casting to float
            logger.debug(f"Creating InfoNCELoss with temperature: {temperature}")
            loss_fn = InfoNCELoss(temperature=temperature, logger=logger) # Pass logger
            logger.debug("InfoNCELoss created successfully.")
            return loss_fn
        else:
            logger.warning(f"Unknown loss function name requested: {loss_name}")
            raise ValueError(f"Unknown loss function name: {loss_name}")