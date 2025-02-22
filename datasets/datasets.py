import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizerFast
from typing import Dict, Any, Tuple, List
import random
import logging
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


class TextDataset(Dataset):
    """PyTorch Dataset for text and rating data with refined masking augmentation."""

    @inject
    def __init__(self, data_path: str, tokenizer_name: str, max_length: int, masking_probability: float, logger: logging.Logger = Provide[Container.logger]):
        """Initializes TextDataset with refined masking.

        Args:
            data_path: Path to the CSV data file.
            tokenizer_name: Name or path of the tokenizer to use (e.g., 'bert-base-uncased').
            max_length: Maximum sequence length for tokenization.
            masking_probability: Probability of masking tokens during augmentation.
            logger: Injected logger instance.
        """
        logger.info(f"Initializing TextDataset with refined masking: data_path={data_path}, tokenizer={tokenizer_name}, max_length={max_length}, masking_probability={masking_probability}")
        self.data = pd.read_csv(data_path)

        if 'rating' in self.data.columns:
            self.data['label'] = self.data['rating'] - 1
            self.data = self.data.drop(columns=['rating'])
            logger.debug("Rating column processed: subtracted 1 and renamed to 'label'.")
        else:
            logger.warning("Expected 'rating' column not found in data.")

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.masking_probability = float(masking_probability)
        self.mask_token_id = self.tokenizer.mask_token_id
        logger.debug(f"Dataset loaded. Samples: {len(self.data)}, Mask Token ID: {self.mask_token_id}")
        self.logger = logger

    @log_function_entry_exit
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    @log_function_entry_exit
    @debug_log_data
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieves a sample from the dataset at the given index, with refined masking.

        Refined masking includes:
        - More accurate complete word masking using token offsets.
        - Span masking.
        - Exclusive application of either span or complete word masking.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing original input_ids, masked input_ids, attention_mask, and label.
        """
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        self.logger.debug(f"Processing sample at index: {idx}, label: {label}")

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True, # Crucial for word masking
        )

        original_input_ids = encoding['input_ids'].squeeze()
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label_tensor = torch.tensor(label, dtype=torch.long)
        offset_mapping = encoding['offset_mapping'].squeeze() # Get offset mappings

        # --- Refined Masking Logic ---
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids) # Get tokens for masking decisions
        word_level_mask_indices = self._get_word_level_mask_indices(tokens, offset_mapping) # Get word-level indices to mask

        masked_input_ids = input_ids.clone() # Create a copy for masking

        for word_index in word_level_mask_indices:
            if random.random() < 0.5:
                # Complete Word Masking
                self._mask_complete_word(masked_input_ids, offset_mapping, word_index)
            else:
                # Span Masking
                self._mask_span(masked_input_ids, word_index)


        sample = {
            'original_input_ids': original_input_ids,
            'masked_input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor
        }
        self.logger.debug(f"Processed sample: original_input_ids shape: {sample['original_input_ids'].shape}, masked_input_ids shape: {sample['masked_input_ids'].shape}, attention_mask shape: {sample['attention_mask'].shape}, label: {sample['label']}")
        return sample


    def _get_word_level_mask_indices(self, tokens: List[str], offset_mapping: torch.Tensor) -> List[int]:
        """Gets word-level indices to mask, excluding special tokens."""
        word_level_indices = []
        current_word_index = None
        for index, token in enumerate(tokens):
            if offset_mapping[index] != (0, 0) and token not in self.tokenizer.special_tokens_map_extended.values(): # Check for non-special tokens
                if current_word_index is None or offset_mapping[index][0] != offset_mapping[index-1][1]: # Start of a new word
                    if random.random() < self.masking_probability:
                        word_level_indices.append(index) # Add index of the first token of the word to be masked
                    current_word_index = index
        return word_level_indices


    def _mask_complete_word(self, masked_input_ids: torch.Tensor, offset_mapping: torch.Tensor, word_start_index: int) -> None:
        """Masks all tokens corresponding to a complete word, starting from word_start_index."""
        start_offset = offset_mapping[word_start_index][0]
        for index in range(word_start_index, len(masked_input_ids)):
            if offset_mapping[index][0] >= start_offset and offset_mapping[index] != (0, 0): # Ensure it's part of the same word and not special token
                masked_input_ids[index] = self.mask_token_id
            else:
                break # Stop when we move to the next word


    def _mask_span(self, masked_input_ids: torch.Tensor, start_index: int, max_span_length: int = 5) -> None:
        """Masks a span of tokens starting from start_index."""
        span_length = random.randint(1, max_span_length)
        for index in range(start_index, min(start_index + span_length, len(masked_input_ids))):
            if offset_mapping[index] != (0, 0): # Avoid masking special tokens within span (though unlikely)
                 masked_input_ids[index] = self.mask_token_id