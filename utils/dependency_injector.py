from dependency_injector import containers, providers
import logging
import yaml
from typing import Dict, Any

from models.factories.model_factory import ModelFactory
from datasets.dataloader_factory import DataLoaderFactory
from losses.loss_factory import LossFunctionFactory
from optimizers.optimizer_factory import OptimizerFactory
from utils.logging_utils import create_logger, LOG_LEVEL_MAP, DEFAULT_LOG_LEVEL


class Container(containers.DeclarativeContainer):
    """Dependency injection container."""

    config = providers.Configuration()
    logger = providers.Singleton(
        create_logger,
        name=config.project.name,
        level_str=config.training.logging_level,
    )
    model_factory = providers.Factory(
        ModelFactory,
    )
    dataloader_factory = providers.Factory(
        DataLoaderFactory,
    )
    loss_factory = providers.Factory(
        LossFunctionFactory,
    )
    optimizer_factory = providers.Factory(
        OptimizerFactory,
    )


def load_config_from_yaml(config_path: str) -> providers.Configuration:
    """Loads configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dependency_injector Configuration provider loaded from the YAML file.
    """
    config_provider = providers.Configuration()
    with open(config_path, 'r') as config_file:
        yaml_config: Dict[str, Any] = yaml.safe_load(config_file) # Type hint yaml_config
    config_provider.from_dict(yaml_config)
    return config_provider