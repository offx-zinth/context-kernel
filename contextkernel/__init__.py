"""
ContextKernel Package Root.

ContextKernel is a framework for building context-aware AI applications.
It provides tools for managing context, interacting with language models,
and exposing these capabilities via APIs and SDKs.
"""

__version__ = "0.1.0"

from .utils import get_settings
from .interfaces import ContextKernelClient, app as api_app

# Core components would be imported here as they are developed, e.g.:
# from .core_logic import ContextAgent

__all__ = [
    "__version__",
    "get_settings",
    "ContextKernelClient",
    "api_app", # The FastAPI application instance, aliased
    # "ContextAgent", # When available
]

# Optional: Basic logging configuration hint for library users,
# though applications using the kernel should ideally set up their own logging.
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
