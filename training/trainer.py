import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import logging
import yaml
import os
from tqdm.autonotebook import tqdm
import wandb
import optuna
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import create_logger, log_function_entry_exit, debug_log_data
from utils.tqdm_utils import get_tqdm
from utils.wandb_utils import initialize_wandb_run, log_metrics_to_wandb, finish_wandb_run
from metrics.metrics import calculate_perplexity, calculate_accuracy
from models.factories.model_factory import ModelFactory
from datasets.dataloader_factory import DataLoaderFactory
from losses.loss_factory import LossFunctionFactory
from optimizers.optimizer_factory import OptimizerFactory
from training.optuna_integration import create_objective_function

DEFAULT_MIXED_PRECISION = True
DEFAULT_LOG_PERPLEXITY_WANDB = True
DEFAULT_RUN_OPTUNA_OPTIMIZATION = True
DEFAULT_OPTUNA_SAMPLER = "tpe"
DEFAULT_OPTUNA_PRUNER = "median"
DEFAULT_METRIC_TO_OPTIMIZE = "val_accuracy"
DEFAULT_OPTUNA_DIRECTION = "maximize"


class Trainer:
    @inject
    def __init__(self, config: Dict[str, Any], container: Container, logger: logging.Logger = Provide[Container.logger]) -> None:
        self.config = config
        self.container = container
        self.logger = logger
        logger.info("Initializing Trainer...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model_factory: ModelFactory = container.model_factory()
        self.dataloader_factory: DataLoaderFactory = container.dataloader_factory()
        self.loss_factory: LossFunctionFactory = container.loss_factory()
        self.optimizer_factory: OptimizerFactory = container.optimizer_factory()

        self.model: nn.Module = self._create_model()
        self.train_dataloader: DataLoader = self._create_dataloader(split="train")
        self.val_dataloader: DataLoader = self._create_dataloader(split="val")
        self.loss_fn: nn.Module = self._create_loss_function()
        self.optimizer: optim.Optimizer = self._create_optimizer()

        self.epochs: int = int(config["training"]["epochs"])
        self.experiment_name: str = str(config["training"]["experiment_name"])
        self.mixed_precision_enabled: bool = bool(config.get("training", {}).get("mixed_precision", DEFAULT_MIXED_PRECISION))
        self.gradient_checkpointing_enabled: bool = bool(config["training"]["gradient_checkpointing"])
        self.gradient_clip_value: float = float(config["regularization"]["gradient_clip_value"])
        self.log_perplexity_wandb = bool(config.get("wandb", {}).get("log_perplexity", DEFAULT_LOG_PERPLEXITY_WANDB))

        logger.debug(f"  Epochs: {self.epochs}, Experiment Name: {self.experiment_name}, Mixed Precision: {self.mixed_precision_enabled}, Gradient Checkpointing: {self.gradient_checkpointing_enabled}, Gradient Clip Value: {self.gradient_clip_value}")
        logger.info("Trainer initialized.")


    @log_function_entry_exit
    @debug_log_data
    def _create_model(self) -> nn.Module:
        model_config: Dict[str, Any] = self.config["model"]
        model: nn.Module = self.model_factory.create_model(model_config)
        model.to(self.device)
        self.logger.info(f"Model '{model_config['name']}' created and moved to device: {self.device}")
        return model


    @log_function_entry_exit
    @debug_log_data
    def _create_dataloader(self, split: str) -> DataLoader:
        dataloader_config: Dict[str, Any] = self.config["data_loading"]
        dataloader_config["tokenizer_name"] = self.config["data_loading"]["tokenizer"]["tokenizer_name"]
        dataloader_config["max_length"] = self.config["data_loading"]["tokenizer"]["max_length"]
        dataloader_config["masking_probability"] = self.config["data_loading"]["tokenizer"]["masking_probability"]

        if split == "val":
            dataloader_config["data_path"] = dataloader_config["data_path"].replace("train", "val")
        dataloader: DataLoader = self.dataloader_factory.create_dataloader(dataloader_config)
        self.logger.info(f"DataLoader created for '{split}' split.")
        return dataloader


    @log_function_entry_exit
    @debug_log_data
    def _create_loss_function(self) -> nn.Module:
        loss_config: Dict[str, Any] = self.config["loss_function"]
        loss_fn: nn.Module = self.loss_factory.create_loss_function(loss_config)
        self.logger.info(f"Loss function '{loss_config['name']}' created.")
        return loss_fn


    @log_function_entry_exit
    @debug_log_data
    def _create_optimizer(self) -> optim.Optimizer:
        optimizer_config: Dict[str, Any] = self.config["optimizer"]
        model_params = self.model.parameters()
        optimizer: optim.Optimizer = self.optimizer_factory.create_optimizer(model_params, optimizer_config)
        self.logger.info(f"Optimizer '{optimizer_config['name']}' created.")
        return optimizer


    @log_function_entry_exit
    def train(self) -> None:
        logger = self.logger
        logger.info("Starting training...")
        model = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        train_dataloader = self.train_dataloader
        device = self.device
        epochs = self.epochs
        gradient_clip_value = self.gradient_clip_value
        log_perplexity_wandb = self.log_perplexity_wandb

        model.train()

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            progress_bar = get_tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)

            for batch_idx, batch in enumerate(progress_bar):
                original_input_ids = batch['original_input_ids'].to(device)
                masked_input_ids = batch['masked_input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                original_outputs = model(original_input_ids, attention_mask=attention_mask)
                masked_outputs = model(masked_input_ids, attention_mask=attention_mask)

                loss = loss_fn(original_outputs, masked_outputs)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)

                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

                if logger.level <= logging.DEBUG:
                    if batch_idx % 10 == 0:
                        logger.debug(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}")

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            perplexity = calculate_perplexity(torch.tensor(avg_epoch_loss), logger=logger)

            log_metrics = {
                "train/epoch_loss": avg_epoch_loss,
                "train/perplexity": perplexity.item(),
                "epoch": epoch + 1
            }
            if log_perplexity_wandb:
                log_metrics["train/perplexity"] = perplexity.item()

            log_metrics_to_wandb(metrics=log_metrics, commit=True, logger=logger)

            logger.info(f"Epoch {epoch+1}/{epochs} Training - Average Loss: {avg_epoch_loss:.4f}, Perplexity: {perplexity.item():.2f}")

        logger.info("Training finished.")


    @log_function_entry_exit
    def validate(self) -> Dict[str, float]:
        logger = self.logger
        logger.info("Starting validation...")
        model = self.model
        val_dataloader = self.val_dataloader
        device = self.device
        loss_fn = self.loss_fn
        gradient_clip_value = self.gradient_clip_value

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            progress_bar = get_tqdm(val_dataloader, desc="Validation", leave=False)
            all_predictions = []
            all_labels = []

            for batch in progress_bar:
                original_input_ids = batch['original_input_ids'].to(device)
                masked_input_ids = batch['masked_input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(original_input_ids, attention_mask=attention_mask)

                loss = loss_fn(outputs, outputs)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = val_loss / len(val_dataloader)
        validation_accuracy = calculate_accuracy(torch.tensor(all_predictions), torch.tensor(all_labels), logger=logger)

        log_metrics = {
            "val/loss": avg_val_loss,
            "val/accuracy": validation_accuracy.item(),
        }
        log_metrics_to_wandb(metrics=log_metrics, commit=True, logger=logger)

        logger.info(f"Validation finished - Average Loss: {avg_val_loss:.4f}, Accuracy: {validation_accuracy:.2f}%")

        return {"val_loss": avg_val_loss, "val_accuracy": validation_accuracy.item()}


    @log_function_entry_exit
    def run_training(self) -> Dict[str, float]:
        logger = self.logger
        logger.info("Starting run_training orchestration...")

        run_optuna = bool(self.config.get("optuna", {}).get("run_optimization", DEFAULT_RUN_OPTUNA_OPTIMIZATION))

        if run_optuna:
            self.run_optuna_optimization()
            return {}
        else:
            val_metrics = self.validate()
            return val_metrics


    @log_function_entry_exit
    def run_optuna_optimization(self) -> None:
        logger = self.logger
        logger.info("Starting Optuna hyperparameter optimization...")

        optuna_config: Dict[str, Any] = self.config["optuna"]
        objective_function = create_objective_function(self.container, self.config, logger=logger)

        sampler_name: str = str(optuna_config.get("sampler", DEFAULT_OPTUNA_SAMPLER))
        pruner_name: str = str(optuna_config.get("pruner", DEFAULT_OPTUNA_PRUNER))
        n_trials: int = int(optuna_config["n_trials"])
        startup_trials: int = int(optuna_config["startup_trials"])
        metric_to_optimize_str: str = str(optuna_config.get("metric_to_optimize", DEFAULT_METRIC_TO_OPTIMIZE))
        direction_str: str = str(optuna_config.get("direction", DEFAULT_OPTUNA_DIRECTION))


        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(n_startup_trials=startup_trials, n_ei_candidates=24)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler name: {sampler_name}")

        if pruner_name == "median":
            pruner = optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=5)
        elif pruner_name == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner name: {pruner_name}")


        study = optuna.create_study(
            direction=direction_str,
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.experiment_name}-optuna",
            storage=f"sqlite:///optuna_{self.experiment_name}.db",
            load_if_exists=True,
        )

        logger.info(f"Optuna study created/loaded: name='{study.study_name}', storage='{study.storage_name}', direction='{study.direction}', sampler='{sampler_name}', pruner='{pruner_name}', n_trials={n_trials}, startup_trials={startup_trials}")

        study.optimize(
            objective_function,
            n_trials=n_trials,
            catch=(ValueError,),
        )

        best_trial = study.best_trial
        logger.info(f"Optuna optimization finished. Best trial:")
        logger.info(f"  Number: {best_trial.number}")
        logger.info(f"  Value (Validation {metric_to_optimize_str}): {best_trial.value:.4f}")
        logger.info(f"  Params: ")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        best_hyperparams = best_trial.params
        best_val_metric = best_trial.value
        self._save_best_hyperparameters_and_results(best_hyperparams, best_val_metric)

        logger.info("Optuna hyperparameter optimization finished.")


    @log_function_entry_exit
    @debug_log_data
    def _save_best_hyperparameters_and_results(self, best_hyperparams: Dict[str, Any], best_val_metric: float) -> None:
        output_dir = "experiments"
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"{self.experiment_name}_best_hparams.yaml")

        results_dict = {
            "best_hyperparameters": best_hyperparams,
            "best_validation_metric": best_val_metric,
            "metric_name": self.config["optuna"]["metric_to_optimize"],
            "direction": self.config["optuna"]["direction"],
            "experiment_name": self.experiment_name,
            "optuna_study_name": f"{self.experiment_name}-optuna",
            "optuna_storage": f"sqlite:///optuna_{self.experiment_name}.db",
        }

        with open(output_filepath, 'w') as outfile:
            yaml.dump(results_dict, outfile, indent=4)

        self.logger.info(f"Best hyperparameters and results saved to: {output_filepath}")


    @log_function_entry_exit
    def run_training(self) -> Dict[str, float]:
        logger = self.logger
        logger.info("Starting run_training orchestration...")

        run_optuna = bool(self.config.get("optuna", {}).get("run_optimization", DEFAULT_RUN_OPTUNA_OPTIMIZATION))

        if run_optuna:
            self.run_optuna_optimization()
            return {}
        else:
            val_metrics = self.validate()
            return val_metrics


def main():
    config_path = "config/config.yaml"
    container = Container()
    container.config.from_dict(yaml.safe_load(open(config_path)))
    container.wire([__name__])

    logger = container.logger()
    logger.info("Starting main training script.")

    trainer = Trainer(config=container.config(), container=container)

    run_optuna = bool(container.config().optuna.get("run_optimization", DEFAULT_RUN_OPTUNA_OPTIMIZATION))

    if run_optuna:
        trainer.run_optuna_optimization()
        logger.info("Optuna optimization run completed.")
    else:
        val_metrics = trainer.run_training()
        logger.info("Standard training run completed.")
        print("Validation Metrics:", val_metrics)

    logger.info("Training script finished.")


if __name__ == "__main__":
    main()