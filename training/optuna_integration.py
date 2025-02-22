import optuna
from typing import Dict, Any, Callable
import logging
from utils.dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data
from training.trainer import Trainer # Import Trainer class


@inject
@log_function_entry_exit
@debug_log_data
def create_objective_function(container: Container, config: Dict[str, Any], logger: logging.Logger = Provide[Container.logger]) -> Callable[[optuna.Trial], float]:
    """Creates the Optuna objective function.

    This function will be optimized by Optuna. It takes an Optuna trial,
    suggests hyperparameters to the trial, trains and validates the model
    with those hyperparameters, and returns the metric to be optimized
    (e.g., validation accuracy).

    Args:
        container: Dependency injection container.
        config: Base configuration dictionary from YAML.
        logger: Injected logger instance.

    Returns:
        An Optuna objective function.
    """
    logger.info("Creating Optuna objective function...")

    def objective(trial: optuna.Trial) -> float:
        """The Optuna objective function to minimize/maximize."""
        trial_logger = logger.getChild(f"trial_{trial.number}") # Create child logger for each trial
        trial_logger.info(f"Starting Optuna trial {trial.number}...")

        # --- Suggest Hyperparameters to Optuna Trial ---
        lr = trial.suggest_float("learning_rate", low=1e-6, high=1e-4, log=True) # Example hyperparameter suggestion

        # --- Update Configuration with Suggested Hyperparameters ---
        trial_config = config.copy() # Create a copy to avoid modifying the base config
        trial_config["optimizer"]["learning_rate"] = lr # Update learning rate in trial config

        # --- Create Trainer Instance with Trial-Specific Configuration ---
        trial_container = container.clone() # Clone the base container for each trial
        trial_container.config.override(trial_config) # Override config in cloned container
        trainer = Trainer(config=trial_container.config(), container=trial_container, logger=trial_logger) # Pass trial logger

        # --- Run Training and Validation ---
        val_metrics = trainer.run_training() # Run training and get validation metrics (accuracy, loss)
        trial_logger.info(f"Trial {trial.number} finished. Validation metrics: {val_metrics}")

        # --- Return Metric to Optimize ---
        metric_to_optimize_str = str(config["optuna"]["metric_to_optimize"]) # Get metric name from config
        direction = str(config["optuna"]["direction"]) # Get direction (maximize/minimize)
        metric_value = val_metrics[metric_to_optimize_str] # Extract metric value from dict

        if direction == "maximize": # Optuna maximizes by default, so for maximize, return metric directly
            return metric_value
        elif direction == "minimize": # For minimize, return negative of the metric (Optuna will still maximize the negative)
            return -metric_value
        else:
            raise ValueError(f"Invalid optimization direction: {direction}. Must be 'maximize' or 'minimize'.")

    logger.info("Optuna objective function created.")
    return objective


# You can add helper functions related to Optuna integration here if needed,
# e.g., for defining the hyperparameter search space more programmatically,
# or for setting up pruning, samplers, etc.