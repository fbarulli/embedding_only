# optimizers/optimizers.py
"""
This file can be used to define custom optimizers or extend existing ones if needed.
For now, it's empty as we are using standard optimizers from torch.optim.
"""
import logging
from dependency_injector.wiring import inject, Provide

from utils.dependency_injector import Container
from utils.logging_utils import log_function_entry_exit, debug_log_data


@inject
def example_optimizer_utility_function(logger: logging.Logger = Provide[Container.logger]):
    """Example utility function related to optimizers (currently does nothing useful)."""
    logger.info("Example optimizer utility function called.")
    # ... Add any utility functions related to optimizers here if needed ...
    pass


# You could define custom Optimizer classes here in the future if required.
# For example:
# class CustomAdamW(optim.AdamW):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, logger=None):
#         super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
#         self.logger = logger # Example of passing logger to a custom optimizer
#         if logger:
#             logger.debug("CustomAdamW optimizer initialized.")