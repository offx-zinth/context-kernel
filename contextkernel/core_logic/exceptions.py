# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
