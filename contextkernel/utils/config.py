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

"""Configuration management for the Context Kernel application."""

import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional # Keep this for existing specific configs like RedisConfig

# Import individual module configurations
from contextkernel.core_logic.summarizer import SummarizerConfig
from contextkernel.core_logic.llm_retriever import LLMRetrieverConfig
from contextkernel.core_logic.llm_listener import LLMListenerConfig
from contextkernel.core_logic.context_agent import ContextAgentConfig
from contextkernel.core_logic.exceptions import ConfigurationError


logger = logging.getLogger(__name__)

# --- Existing Specific Configs (preserved) ---
class RedisConfig(BaseSettings):
    host: str = Field(default="localhost", validation_alias='REDIS_HOST')
    port: int = Field(default=6379, validation_alias='REDIS_PORT')
    password: Optional[str] = Field(default=None, validation_alias='REDIS_PASSWORD')
    db: int = Field(default=0, validation_alias='REDIS_DB')

    model_config = SettingsConfigDict(env_prefix='APP_REDIS_') # Example prefix for these specific settings


class Neo4jConfig(BaseSettings):
    uri: str = Field(default="neo4j://localhost:7687", validation_alias='NEO4J_URI')
    user: str = Field(default="neo4j", validation_alias='NEO4J_USER')
    password: str = Field(default="password", validation_alias='NEO4J_PASSWORD')

    model_config = SettingsConfigDict(env_prefix='APP_NEO4J_')


class VectorDBConfig(BaseSettings):
    type: str = Field(default="pinecone", validation_alias='VECTOR_DB_TYPE') # Example
    api_key: Optional[str] = Field(default=None, validation_alias='VECTOR_DB_API_KEY')
    environment: Optional[str] = Field(default=None, validation_alias='VECTOR_DB_ENVIRONMENT')

    model_config = SettingsConfigDict(env_prefix='APP_VECTOR_DB_')

# --- Main Application Configuration ---

class AppConfig(BaseSettings):
    """
    Master configuration class for the application.
    Nests individual component configurations and allows for environment variable overrides.
    """
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    retriever: LLMRetrieverConfig = Field(default_factory=LLMRetrieverConfig)
    listener: LLMListenerConfig = Field(default_factory=LLMListenerConfig)
    agent: ContextAgentConfig = Field(default_factory=ContextAgentConfig)

    # Keep other specific configs if they are meant to be part of the global app config
    # and not solely managed by other parts of a larger system.
    # For now, they are separate, but could be nested under AppConfig too if desired:
    # redis: RedisConfig = Field(default_factory=RedisConfig)
    # neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    # vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix='CK_APP_', # Main prefix for AppConfig settings
        env_nested_delimiter='__',
        env_file='.env',
        extra='ignore'
    )

def load_config() -> AppConfig:
    """Loads the application configuration."""
    try:
        config = AppConfig()
        logger.info("Application configuration loaded successfully.")
        logger.debug(f"AppConfig log_level: {config.log_level}")
        # Log a sample from each nested config to verify they are loaded
        logger.debug(f"Summarizer config via AppConfig: {config.summarizer.hf_abstractive_model_name}")
        logger.debug(f"Retriever config via AppConfig: {config.retriever.embedding_model_name}")
        logger.debug(f"Listener config via AppConfig: {config.listener.entity_extraction_model_name}")
        logger.debug(f"Agent config via AppConfig: {config.agent.spacy_model_name}")
        return config
    except Exception as e:
        logger.error(f"Failed to load AppConfig: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to load application configuration: {e}") from e

# Example usage for direct testing of this module
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Create a dummy .env file for testing
    with open(".env", "w") as f:
        f.write("CK_APP_LOG_LEVEL=DEBUG\n")
        # Example of overriding a nested config field:
        # AppConfig.env_prefix = 'CK_APP_'
        # AppConfig.env_nested_delimiter = '__'
        # SummarizerConfig.env_prefix = 'SUMMARIZER_' (this is used if SummarizerConfig is loaded standalone)
        # So, for AppConfig, the var is CK_APP_SUMMARIZER__HF_ABSTRACTIVE_MODEL_NAME
        f.write("CK_APP_SUMMARIZER__HF_ABSTRACTIVE_MODEL_NAME=google/pegasus-xsum\n")
        f.write("CK_APP_RETRIEVER__DEFAULT_TOP_K=7\n")
        f.write("CK_APP_AGENT__INTENT_CLASSIFIER_MODEL=typeform/distilbert-base-uncased-mnli\n")
        # For specific configs if they were nested under AppConfig:
        # f.write("CK_APP_REDIS__HOST=myredishost\n")

    try:
        app_config = load_config()
        print("\nAppConfig loaded successfully!")
        print(f"Log Level: {app_config.log_level}")
        print(f"Summarizer Model: {app_config.summarizer.hf_abstractive_model_name}")
        print(f"Retriever Top K: {app_config.retriever.default_top_k}")
        print(f"Agent Intent Model: {app_config.agent.intent_classifier_model}")

        # To load specific configs (if they are not nested under AppConfig)
        # print("\nLoading specific configs separately (if not nested):")
        # redis_settings = RedisConfig()
        # print(f"Separate Redis Host: {redis_settings.host}")

    except Exception as e:
        print(f"Error loading or testing config: {e}")
    finally:
        import os
        if os.path.exists(".env"):
            os.remove(".env")
