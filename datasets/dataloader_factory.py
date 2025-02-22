import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging
from dependency_injector.wiring import inject, Provide

from .dataset import TextDataset
from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data

# --- Hardcoded Defaults --- (DEFINED HERE - These are now ACTIVE defaults in the code)
DEFAULT_DATASET_TYPE = "text_dataset"
DEFAULT_SHUFFLE = True
DEFAULT_PIN_MEMORY = True # Hardcoded default for pin_memory
# --- End Hardcoded Defaults ---


class DataLoaderFactory:
    """Factory for creating PyTorch DataLoaders."""

    @staticmethod
    @log_function_entry_exit
    @debug_log_data
    @inject
    def create_dataloader(dataloader_config: Dict[str, Any], logger: logging.Logger = Provide[Container.logger]) -> DataLoader:
        """Creates a DataLoader instance based on the configuration."""
        dataset_type: str = str(dataloader_config.get("dataset_type", DEFAULT_DATASET_TYPE)) # Default dataset_type - using .get() with default
        data_path: str = str(dataloader_config["data_path"]) # Direct access - data_path MUST be in config
        batch_size: int = int(dataloader_config["batch_size"]) # Direct access - batch_size MUST be in config
        shuffle: bool = bool(dataloader_config.get("shuffle", DEFAULT_SHUFFLE)) # Default shuffle - using .get() with default
        num_workers: int = int(dataloader_config["num_workers"]) # Direct access - num_workers MUST be in config
        pin_memory: bool = bool(dataloader_config.get("pin_memory", DEFAULT_PIN_MEMORY)) # Default pin_memory - using .get() with default
        tokenizer_name: str = str(dataloader_config["tokenizer"]["tokenizer_name"]) # Nested access - tokenizer_name MUST be in config
        max_length: int = int(dataloader_config["tokenizer"]["max_length"]) # Nested access - max_length MUST be in config
        masking_probability: float = float(dataloader_config["tokenizer"]["masking_probability"]) # Nested access - masking_probability MUST be in config


        logger.info(f"Creating DataLoader for dataset type: {dataset_type}")

        if dataset_type == "text_dataset":
            logger.debug(f"Creating TextDataset...")
            dataset = TextDataset(data_path, tokenizer_name, max_length, masking_probability, logger=logger)
            logger.debug(f"TextDataset created successfully.")
        else:
            logger.warning(f"Unknown dataset type requested: {dataset_type}")
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        logger.debug(f"DataLoader created with batch_size: {batch_size}, shuffle: {shuffle}, num_workers: {num_workers}, pin_memory: {pin_memory}")
        return dataloader