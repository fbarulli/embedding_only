import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


class InfoNCELoss(nn.Module):
    """InfoNCE loss function."""

    @inject
    def __init__(self, temperature: float = 0.1, logger: logging.Logger = Provide[Container.logger]):
        """Initializes InfoNCELoss.

        Args:
            temperature: Temperature parameter for InfoNCE loss. Defaults to 0.1.
            logger: Injected logger instance.
        """
        super().__init__()
        self.temperature = float(temperature) # Casting to float for robustness
        self.logger = logger # Store logger if needed for more granular logging later
        logger.debug(f"Initializing InfoNCELoss with temperature: {self.temperature}")


    @log_function_entry_exit
    @debug_log_data
    def forward(self, embeddings: torch.Tensor, augmented_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for InfoNCE loss.

        Args:
            embeddings: Embeddings of original samples (batch_size, seq_length, hidden_size).
            augmented_embeddings: Embeddings of augmented samples (batch_size, seq_length, hidden_size).

        Returns:
            The InfoNCE loss value (scalar).
        """
        self.logger.debug(f"  Input embeddings shape: {embeddings.shape}, augmented_embeddings shape: {augmented_embeddings.shape}")

        # Assuming embeddings and augmented_embeddings are from the same batch
        batch_size = embeddings.shape[0]
        embeddings = embeddings.view(batch_size, -1) # Flatten if needed
        augmented_embeddings = augmented_embeddings.view(batch_size, -1) # Flatten if needed

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        augmented_embeddings_norm = F.normalize(augmented_embeddings, p=2, dim=1)

        # Compute cosine similarity between each embedding and all augmented embeddings in the batch
        similarity_matrix = torch.matmul(embeddings_norm, augmented_embeddings_norm.T)

        # Create positive pairs (diagonal elements of similarity matrix)
        positives = torch.diag(similarity_matrix)

        # Negative pairs are all other elements in the similarity matrix
        negatives = similarity_matrix[~torch.eye(batch_size, dtype=bool)].view(batch_size, -1)

        # InfoNCE loss calculation
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        logits /= self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device) # Positive class is always the first

        loss: torch.Tensor = F.cross_entropy(logits, labels) # Type hinting loss
        self.logger.debug(f"  Output loss value: {loss.item()}") # Log scalar loss value
        return loss