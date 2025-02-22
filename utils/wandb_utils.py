import wandb
from typing import Dict, Any
import logging
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


@inject
@log_function_entry_exit
@debug_log_data
def initialize_wandb_run(config: Dict[str, Any], logger: logging.Logger = Provide[Container.logger]) -> None:
    """Initializes a Wandb run.

    Args:
        config: Configuration dictionary from YAML.
        logger: Injected logger instance.
    """
    logger.info("Initializing Wandb run...")
    wandb_config: Dict[str, Any] = config["wandb"] # Get wandb specific config

    wandb.init(
        project=str(wandb_config["project_name"]), # Ensure string type
        entity=str(wandb_config["entity"]),     # Ensure string type
        config=config,                         # Log the entire config to Wandb
        name=str(config["training"]["experiment_name"]), # Experiment name from config
    )
    logger.info(f"Wandb run initialized: project='{wandb_config['project_name']}', experiment='{config['training']['experiment_name']}'")


@inject
@log_function_entry_exit
@debug_log_data
def log_metrics_to_wandb(metrics: Dict[str, Any], commit: bool = True, logger: logging.Logger = Provide[Container.logger]) -> None:
    """Logs metrics to the current Wandb run.

    Args:
        metrics: Dictionary of metrics to log (e.g., {"train_loss": 0.5, "val_accuracy": 85.0}).
        commit: Whether to commit the logged metrics to Wandb immediately (default: True).
        logger: Injected logger instance.
    """
    logger.debug(f"Logging metrics to Wandb: {metrics}")
    wandb.log(metrics, commit=commit)
    logger.debug("Metrics logged to Wandb.")


@inject
@log_function_entry_exit
@debug_log_data
def finish_wandb_run(logger: logging.Logger = Provide[Container.logger]) -> None:
    """Finishes the current Wandb run.

    Args:
        logger: Injected logger instance.
    """
    logger.info("Finishing Wandb run...")
    wandb.finish()
    logger.info("Wandb run finished.")