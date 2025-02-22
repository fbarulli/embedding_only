import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple
import logging
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


class TransformerEmbeddingModel(nn.Module):
    """Transformer-based embedding model."""

    @inject
    def __init__(self, config: Dict[str, Any], logger: logging.Logger = Provide[Container.logger]) -> None:
        """Initializes TransformerEmbeddingModel.

        Args:
            config: Configuration dictionary for the transformer model from YAML.
            logger: Injected logger instance.
        """
        super().__init__()
        self.logger = logger # Store logger as instance attribute
        self.logger.info("Initializing TransformerEmbeddingModel") # Corrected to self.logger

        transformer_config: Dict[str, Any] = config["transformer_config"]

        self.hidden_size: int = int(transformer_config["hidden_size"])
        self.num_layers: int = int(transformer_config["num_layers"])
        self.num_attention_heads: int = int(transformer_config["num_attention_heads"])
        self.intermediate_size: int = int(transformer_config["intermediate_size"])
        self.dropout_prob: float = float(transformer_config["dropout_prob"])
        self.max_seq_length: int = int(transformer_config["max_seq_length"])
        self.vocab_size: int = int(transformer_config["vocab_size"])

        self.logger.debug(f"  Hidden Size: {self.hidden_size}, Layers: {self.num_layers}, Attention Heads: {self.num_attention_heads}, Intermediate Size: {self.intermediate_size}, Dropout: {self.dropout_prob}, Max Seq Length: {self.max_seq_length}, Vocab Size: {self.vocab_size}") # Corrected to self.logger

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.hidden_size, self.num_attention_heads, self.intermediate_size, self.dropout_prob, logger=logger) # Pass logger - this is correct
            for _ in range(self.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.config: Dict[str, Any] = config
        self.logger.info("TransformerEmbeddingModel initialized successfully.") # Corrected to self.logger


    @log_function_entry_exit
    @debug_log_data
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the TransformerEmbeddingModel.

        Args:
            input_ids: Input token IDs (batch_size, seq_length).
            attention_mask: Attention mask (batch_size, seq_length), optional.

        Returns:
            Token embeddings (batch_size, seq_length, hidden_size).
        """
        self.logger.debug(f"Input input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape if attention_mask is not None else None}") # Corrected to self.logger

        input_embeds: torch.Tensor = self.embedding(input_ids)
        hidden_states: torch.Tensor = input_embeds

        extended_attention_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for i, layer in enumerate(self.transformer_layers):
            self.logger.debug(f"  Processing Transformer Layer {i+1}/{len(self.transformer_layers)}") # Corrected to self.logger
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)

        pooled_output: torch.Tensor = self.layer_norm(hidden_states)
        pooled_output = self.dropout(pooled_output)

        self.logger.debug(f"Output embeddings shape: {pooled_output.shape}") # Corrected to self.logger
        return pooled_output


class TransformerLayer(nn.Module):
    """Single Transformer layer."""

    @inject
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int, dropout_prob: float, logger: logging.Logger = Provide[Container.logger]) -> None:
        """Initializes TransformerLayer.

        Args:
            hidden_size: Hidden size of the transformer layer.
            num_attention_heads: Number of attention heads.
            intermediate_size: Intermediate size of the feedforward network.
            dropout_prob: Dropout probability.
            logger: Injected logger instance.
        """
        super().__init__()
        self.logger = logger # Store logger as instance attribute
        self.logger.debug("Initializing TransformerLayer") # Corrected to self.logger
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_prob, logger=logger) # Pass logger - correct
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.logger.debug("TransformerLayer initialized.") # Corrected to self.logger


    @log_function_entry_exit
    @debug_log_data
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the TransformerLayer.

        Args:
            hidden_states: Input hidden states (batch_size, seq_length, hidden_size).
            attention_mask: Attention mask, optional.

        Returns:
            Output hidden states (batch_size, seq_length, hidden_size).
        """
        self.logger.debug(f"  Input hidden_states shape: {hidden_states.shape}, attention_mask shape: {attention_mask.shape if attention_mask is not None else None}") # Corrected to self.logger
        attention_output: torch.Tensor = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        intermediate_output: torch.Tensor = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        layer_output: torch.Tensor = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.layer_norm2(hidden_states + layer_output)
        self.logger.debug(f"  Output hidden_states shape: {hidden_states.shape}") # Corrected to self.logger
        return hidden_states



class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    @inject
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float, logger: logging.Logger = Provide[Container.logger]) -> None:
        """Initializes MultiHeadAttention.

        Args:
            hidden_size: Hidden size of the attention layer.
            num_attention_heads: Number of attention heads.
            dropout_prob: Dropout probability.
            logger: Injected logger instance.
        """
        super().__init__()
        self.logger = logger # Store logger as instance attribute
        self.logger.debug("Initializing MultiHeadAttention") # Corrected to self.logger
        self.num_attention_heads: int = int(num_attention_heads)
        self.attention_head_size: int = int(hidden_size / num_attention_heads)
        self.all_head_size: int = int(self.num_attention_heads * self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.output_dense = nn.Linear(self.all_head_size, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.logger.debug("MultiHeadAttention initialized.") # Corrected to self.logger


    @log_function_entry_exit
    @debug_log_data
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the MultiHeadAttention layer.

        Args:
            hidden_states: Input hidden states (batch_size, seq_length, hidden_size).
            attention_mask: Attention mask, optional.

        Returns:
            Attention output (batch_size, seq_length, hidden_size).
        """
        self.logger.debug(f"  Input hidden_states shape: {hidden_states.shape}, attention_mask shape: {attention_mask.shape if attention_mask is not None else None}") # Corrected to self.logger

        mixed_query_layer: torch.Tensor = self.query(hidden_states)
        mixed_key_layer: torch.Tensor = self.key(hidden_states)
        mixed_value_layer: torch.Tensor = self.value(hidden_states)

        query_layer: torch.Tensor = self.transpose_for_scores(mixed_query_layer)
        key_layer: torch.Tensor = self.transpose_for_scores(mixed_key_layer)
        value_layer: torch.Tensor = self.transpose_for_scores(mixed_value_layer)

        attention_scores: torch.Tensor = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs: torch.Tensor = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer: torch.Tensor = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output: torch.Tensor = self.output_dense(context_layer)
        output = self.output_layer_norm(output)

        self.logger.debug(f"  Output attention shape: {output.shape}") # Corrected to self.logger
        return output

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose to shape [batch_size, num_heads, seq_length, head_size]."""
        new_x_shape: Tuple[int, ...] = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)