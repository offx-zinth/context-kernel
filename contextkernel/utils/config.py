"""Configuration management for the Context Kernel application."""

import logging
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Import individual module configurations
from contextkernel.core_logic.summarizer import SummarizerConfig
from contextkernel.core_logic.llm_retriever import LLMRetrieverConfig
from contextkernel.core_logic.llm_listener import LLMListenerConfig
from contextkernel.core_logic import NLPConfig # Updated import
from contextkernel.core_logic.exceptions import ConfigurationError


logger = logging.getLogger(__name__)

# --- Nested Configuration Models ---

class ServerConfig(BaseSettings):
    enabled: bool = Field(default=True, description="Enable or disable the API server.")
    host: str = Field(default="0.0.0.0", description="Server host address.")
    port: int = Field(default=8000, description="Server port.")
    log_level: str = Field(default="INFO", description="Logging level for the server.")

    model_config = SettingsConfigDict(env_prefix='CK_SERVER_')


class RedisConfig(BaseSettings):
    host: str = Field(default="localhost", validation_alias='CK_REDIS_HOST')
    port: int = Field(default=6379, validation_alias='CK_REDIS_PORT')
    password: Optional[SecretStr] = Field(default=None, validation_alias='CK_REDIS_PASSWORD')
    db: int = Field(default=0, validation_alias='CK_REDIS_DB')

    # Removed model_config = SettingsConfigDict(env_prefix='APP_REDIS_') as it will be nested


class Neo4jConfig(BaseSettings):
    uri: str = Field(default="neo4j://localhost:7687", validation_alias='CK_NEO4J_URI')
    user: str = Field(default="neo4j", validation_alias='CK_NEO4J_USER')
    password: Optional[SecretStr] = Field(default=None, validation_alias='CK_NEO4J_PASSWORD') # Changed to SecretStr and made Optional for consistency

    # Removed model_config = SettingsConfigDict(env_prefix='APP_NEO4J_')


class VectorDBConfig(BaseSettings):
    type: str = Field(default="pinecone", validation_alias='CK_VECTOR_DB_TYPE', description="Type of vector database (e.g., 'pinecone', 'chroma').")
    api_key: Optional[SecretStr] = Field(default=None, validation_alias='CK_VECTOR_DB_API_KEY')
    environment: Optional[str] = Field(default=None, validation_alias='CK_VECTOR_DB_ENVIRONMENT', description="Cloud environment for the vector database, if applicable.")

    # Removed model_config = SettingsConfigDict(env_prefix='APP_VECTOR_DB_')


class LLMConfig(BaseSettings):
    provider: str = Field(default="openai", description="LLM provider (e.g., 'openai', 'anthropic', 'google').")
    api_key: Optional[SecretStr] = Field(default=None, validation_alias='CK_LLM_API_KEY', description="API key for the LLM provider.")
    model: Optional[str] = Field(default=None, description="Specific model name to use (e.g., 'gpt-4', 'claude-2', 'gemini-pro').")

    # Removed model_config as it will be nested

# --- Main Application Settings ---

class AppSettings(BaseSettings):
    """
    Master settings class for the application.
    Nests individual component configurations and allows for environment variable overrides.
    """
    app_name: str = Field(default="contextkernel", description="Application name.")
    version: str = Field(default="0.1.0", description="Application version.")
    debug_mode: bool = Field(default=False, description="Enable debug mode.")
    state_manager_type: str = Field(default="in_memory", validation_alias="CK_STATE_MANAGER_TYPE", description="Type of state manager to use ('redis' or 'in_memory').")

    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig) # Added LLMConfig

    # Core logic modules
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    retriever: LLMRetrieverConfig = Field(default_factory=LLMRetrieverConfig)
    listener: LLMListenerConfig = Field(default_factory=LLMListenerConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig) # Renamed from agent to nlp

    # Data stores
    redis: RedisConfig = Field(default_factory=RedisConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)

    # log_level: str = "INFO" # This is now part of ServerConfig

    model_config = SettingsConfigDict(
        env_prefix='CK_', # Main prefix for AppSettings
        env_nested_delimiter='__',
        env_file='.env',
        extra='ignore'
    )

def get_settings() -> AppSettings:
    """Loads the application settings."""
    try:
        settings = AppSettings()
        logger.info("Application settings loaded successfully.")
        logger.debug(f"AppSettings loaded: {settings.model_dump_json(indent=2)}")
        return settings
    except Exception as e:
        logger.error(f"Failed to load AppSettings: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to load application settings: {e}") from e

# Example usage for direct testing of this module
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Set root logger to DEBUG for detailed output

    # Create a dummy .env file for testing
    with open(".env", "w") as f:
        f.write("CK_DEBUG_MODE=true\n")
        f.write("CK_SERVER__ENABLED=true\n") # Example for server enabled
        f.write("CK_SERVER__HOST=127.0.0.1\n")
        f.write("CK_SERVER__PORT=8080\n")
        f.write("CK_SERVER__LOG_LEVEL=DEBUG\n") # Example of overriding server log level
        f.write("CK_STATE_MANAGER_TYPE=redis\n") # Example for state manager type
        f.write("CK_LLM__PROVIDER=anthropic\n")
        f.write("CK_LLM__API_KEY=mysecretllmapikey\n") # Example LLM API Key
        f.write("CK_SUMMARIZER__HF_ABSTRACTIVE_MODEL_NAME=google/pegasus-xsum-test\n")
        f.write("CK_RETRIEVER__DEFAULT_TOP_K=8\n")
        f.write("CK_NLP__INTENT_CLASSIFIER_MODEL=another/test-model\n") # Updated from CK_AGENT_ to CK_NLP_
        f.write("CK_REDIS__HOST=testhost\n")
        f.write("CK_REDIS__PASSWORD=testredispassword\n")
        f.write("CK_NEO4J__USER=testuser\n")
        f.write("CK_NEO4J__PASSWORD=testneo4jpassword\n")
        f.write("CK_VECTOR_DB__TYPE=chroma\n")
        f.write("CK_VECTOR_DB__API_KEY=testvectordbapikey\n")


    logger.info("Attempting to load settings for __main__ example...")
    try:
        app_settings = get_settings()
        print("\nAppSettings loaded successfully!")
        print(f"App Name: {app_settings.app_name}")
        print(f"Version: {app_settings.version}")
        print(f"Debug Mode: {app_settings.debug_mode}")
        print(f"State Manager Type: {app_settings.state_manager_type}")
        print(f"Server Enabled: {app_settings.server.enabled}")
        print(f"Server Host: {app_settings.server.host}")
        print(f"Server Port: {app_settings.server.port}")
        print(f"Server Log Level: {app_settings.server.log_level}")
        print(f"LLM Provider: {app_settings.llm.provider}")
        if app_settings.llm.api_key:
            print(f"LLM API Key: {app_settings.llm.api_key.get_secret_value()[:5]}...") # Print only a part for security
        print(f"Summarizer Model: {app_settings.summarizer.hf_abstractive_model_name}")
        print(f"Retriever Top K: {app_settings.retriever.default_top_k}")
        print(f"NLP Intent Model: {app_settings.nlp.intent_classifier_model}") # Updated from agent to nlp
        print(f"Redis Host: {app_settings.redis.host}")
        if app_settings.redis.password:
            print(f"Redis Password: {app_settings.redis.password.get_secret_value()[:5]}...")
        print(f"Neo4j User: {app_settings.neo4j.user}")
        if app_settings.neo4j.password:
            print(f"Neo4j Password: {app_settings.neo4j.password.get_secret_value()[:5]}...")
        print(f"VectorDB Type: {app_settings.vector_db.type}")
        if app_settings.vector_db.api_key:
            print(f"VectorDB API Key: {app_settings.vector_db.api_key.get_secret_value()[:5]}...")


    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        import os
        if os.path.exists(".env"):
            os.remove(".env")
            logger.info(".env file removed.")
