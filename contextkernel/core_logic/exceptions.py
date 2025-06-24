
"""Custom exception classes for the Context Kernel's core logic."""

class CoreLogicError(Exception):
    """Base class for exceptions in the core logic modules."""
    pass

class SummarizationError(CoreLogicError):
    """Raised when an error occurs during text summarization."""
    pass

class EmbeddingError(CoreLogicError):
    """Raised when an error occurs during embedding generation or processing."""
    pass

class MemoryAccessError(CoreLogicError):
    """Raised when an error occurs while accessing a memory system (LTM, GraphDB, Cache, etc.)."""
    pass

class IntentDetectionError(CoreLogicError):
    """Raised when an error occurs during intent detection."""
    pass

class ConfigurationError(CoreLogicError):
    """Raised when there is a configuration problem for a core logic component."""
    pass

class ExternalServiceError(CoreLogicError):
    """Raised when an error occurs while interacting with an external service (e.g., an LLM API)."""
    pass

class PipelineError(CoreLogicError):
    """Raised when an error occurs within a Hugging Face pipeline or similar processing pipeline."""
    pass
