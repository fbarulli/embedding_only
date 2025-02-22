from tqdm.autonotebook import tqdm as notebook_tqdm
from tqdm import tqdm as standard_tqdm  # Import standard tqdm as well, to avoid name clash
from typing import Iterable, Optional, Sized

def get_tqdm(iterable: Optional[Iterable] = None, desc: str = "Processing", total: Optional[int] = None, disable: bool = False):
    """
    Returns a tqdm progress bar, defaulting to notebook tqdm if in a notebook environment.

    Args:
        iterable:  Iterable to track progress over.
        desc: Description for the progress bar.
        total: Total number of iterations (if iterable length is not readily available).
        disable: Whether to disable the progress bar.

    Returns:
        A tqdm progress bar instance.
    """
    try:
        # Check if we are in a Jupyter Notebook environment
        get_ipython()
        return notebook_tqdm(iterable, desc=desc, total=total, disable=disable)
    except NameError:
        # Not in a notebook, use standard tqdm
        return standard_tqdm(iterable, desc=desc, total=total, disable=disable) # Use standard_tqdm here


# You can add more utility functions related to tqdm here,
# such as custom tqdm formatting, styles, etc. in the future.