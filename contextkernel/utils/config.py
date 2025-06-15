from typing import Optional, Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# For .env file support, ensure 'python-dotenv' is installed.
# Pydantic-settings will automatically load it if present.

class ServerSettings(BaseSettings):
    """Configuration for API server settings."""
    model_config = SettingsConfigDict(env_prefix='CK_SERVER_')

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enabled: bool = Field(True, description="Enable or disable the API server.")


class DatabaseSettings(BaseSettings):
    """Configuration for database connection."""
    model_config = SettingsConfigDict(env_prefix='CK_DATABASE_')

    url: str = "sqlite:///./context_kernel.db"  # Default to a local SQLite DB
    username: Optional[str] = None
    password: Optional[SecretStr] = None # Use SecretStr for sensitive fields
    pool_size: int = Field(5, gt=0, description="Database connection pool size.")
    connect_timeout: int = Field(10, gt=0, description="Connection timeout in seconds.")


class LLMSettings(BaseSettings):
    """Configuration for Language Model (LLM) interactions."""
    model_config = SettingsConfigDict(env_prefix='CK_LLM_')

    # api_key is sensitive and should be loaded from env or .env
    api_key: Optional[SecretStr] = Field(None, description="API key for the LLM service.")
    model: str = "default-model" # Example: "gpt-4", "claude-3-opus-20240229"
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature setting.")
    max_tokens: int = Field(1024, gt=0, description="Maximum tokens for LLM response.")
    # Example of a specific provider setting
    openai_organization_id: Optional[str] = Field(None, description="OpenAI Organization ID, if applicable.")


class AppSettings(BaseSettings):
    """
    Main application settings, incorporating nested configurations.
    Loads from a .env file and environment variables.
    Environment variables should be prefixed with 'CK_'.
    Nested settings use '__' as a delimiter (e.g., CK_SERVER__HOST).
    """
    model_config = SettingsConfigDict(
        env_file='.env',              # Load from .env file
        env_file_encoding='utf-8',
        env_prefix='CK_',             # Prefix for all environment variables
        env_nested_delimiter='__',    # Delimiter for nested models
        extra='ignore'                # Ignore extra fields from env/files
    )

    debug_mode: bool = Field(False, description="Enable debug mode for verbose logging and diagnostics.")
    app_name: str = Field("ContextKernel", description="Name of the application.")
    version: str = Field("0.1.0", description="Application version.")

    server: ServerSettings = ServerSettings()
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()

    # Example of a top-level sensitive setting
    global_secret_key: Optional[SecretStr] = Field(None, description="A global secret key for the application.")


# --- How to load and use the settings ---
# 1. Create a .env file in the root of your project (where you run the app from)
#    Example .env file content:
#    ```
#    CK_DEBUG_MODE=True
#    CK_SERVER__PORT=8080
#    CK_DATABASE__URL="postgresql://user:pass@host:port/dbname"
#    CK_LLM__API_KEY="your_llm_api_key_here"
#    CK_LLM__MODEL="gpt-4-turbo"
#    CK_GLOBAL_SECRET_KEY="a_very_secret_top_level_key"
#    ```
#    Ensure this .env file is in your .gitignore to avoid committing secrets.

# 2. Instantiate AppSettings:
#    The settings are automatically loaded upon instantiation.
_cached_settings: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    """
    Loads and returns the application settings.
    Settings are loaded once and cached for performance.
    """
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = AppSettings()
    return _cached_settings

# Example usage (typically you'd call get_settings() in your app modules):
if __name__ == "__main__":
    import os

    print("--- Example of loading and using AppSettings ---")

    # For demonstration, let's set some environment variables
    # In a real app, these would be set in your shell or .env file
    os.environ["CK_DEBUG_MODE"] = "true"
    os.environ["CK_SERVER__PORT"] = "9000"
    os.environ["CK_SERVER__LOG_LEVEL"] = "DEBUG"
    os.environ["CK_DATABASE__URL"] = "postgresql://demo_user:demo_pass@localhost/demodb"
    os.environ["CK_DATABASE__USERNAME"] = "demo_user_env" # Overrides if .env had it under CK_DATABASE__USERNAME
    os.environ["CK_LLM__API_KEY"] = "env_llm_api_key_12345"
    os.environ["CK_LLM__MODEL"] = "env-gpt-model"
    os.environ["CK_LLM__OPENAI_ORGANIZATION_ID"] = "org-env123"
    os.environ["CK_GLOBAL_SECRET_KEY"] = "env_global_secret"


    # Create a dummy .env file for this example run
    # In a real scenario, this file would exist in your project root.
    # Pydantic-settings will pick it up if `python-dotenv` is installed.
    try:
        with open(".env", "w") as f:
            f.write("CK_APP_NAME=MyContextKernelAppFromDotEnv\n")
            f.write("CK_DATABASE__URL=sqlite:///./dotenv.db\n") # Env var CK_DATABASE__URL will override this
            f.write("CK_DATABASE__USERNAME=dotenv_user\n") # Env var CK_DATABASE__USERNAME will override this
            f.write("CK_LLM__API_KEY=dotenv_api_key_should_be_overridden_by_env\n") # Env var will override
            f.write("CK_LLM__TEMPERATURE=0.9\n") # Not set in env, so this will be used

        print("\nCreated a dummy .env file for demonstration.")
        print("Content of .env:")
        with open(".env", "r") as f:
            print(f.read())

        # Instantiate settings (or use get_settings())
        # Forcing a reload for the demo by not using the cache directly here
        current_settings = AppSettings()

        print("\n--- Loaded Settings (Priority: Env Vars > .env > Defaults) ---")
        print(f"App Name: {current_settings.app_name}") # From .env
        print(f"Debug Mode: {current_settings.debug_mode}") # From env var
        print(f"Version: {current_settings.version}") # Default

        print("\nServer Settings:")
        print(f"  Host: {current_settings.server.host}") # Default
        print(f"  Port: {current_settings.server.port}") # From env var
        print(f"  Log Level: {current_settings.server.log_level}") # From env var

        print("\nDatabase Settings:")
        print(f"  URL: {current_settings.database.url}") # From env var (overrides .env)
        print(f"  Username: {current_settings.database.username}") # From env var (overrides .env)
        if current_settings.database.password:
            print(f"  Password: {current_settings.database.password.get_secret_value()}")
        else:
            print("  Password: Not set")
        print(f"  Pool Size: {current_settings.database.pool_size}") # Default
        print(f"  Connect Timeout: {current_settings.database.connect_timeout}") # Default


        print("\nLLM Settings:")
        if current_settings.llm.api_key:
            print(f"  API Key: {current_settings.llm.api_key.get_secret_value()}") # From env var (overrides .env)
        else:
            print("  API Key: Not set")
        print(f"  Model: {current_settings.llm.model}") # From env var
        print(f"  Temperature: {current_settings.llm.temperature}") # From .env
        print(f"  Max Tokens: {current_settings.llm.max_tokens}") # Default
        print(f"  OpenAI Org ID: {current_settings.llm.openai_organization_id}") # From env var

        if current_settings.global_secret_key:
            print(f"\nGlobal Secret Key: {current_settings.global_secret_key.get_secret_value()}") # From env var
        else:
            print("\nGlobal Secret Key: Not set")

    finally:
        # Clean up the dummy .env file and environment variables
        if os.path.exists(".env"):
            os.remove(".env")
            print("\nCleaned up dummy .env file.")
        del os.environ["CK_DEBUG_MODE"]
        del os.environ["CK_SERVER__PORT"]
        del os.environ["CK_SERVER__LOG_LEVEL"]
        del os.environ["CK_DATABASE__URL"]
        del os.environ["CK_DATABASE__USERNAME"]
        del os.environ["CK_LLM__API_KEY"]
        del os.environ["CK_LLM__MODEL"]
        del os.environ["CK_LLM__OPENAI_ORGANIZATION_ID"]
        del os.environ["CK_GLOBAL_SECRET_KEY"]

    print("\n--- Accessing via get_settings() function (cached) ---")
    # settings_instance will be the same as current_settings if not re-instantiated after env changes
    # For a true test of get_settings() caching, it should be called multiple times without
    # re-instantiating AppSettings() or changing environment variables in between.
    # Here, we demonstrate it gets *an* instance.
    # Re-setting one env var to show that calling AppSettings() again would pick it up,
    # but get_settings() would return the cached one if we didn't clear _cached_settings.
    _cached_settings = None # Clear cache for demo purposes
    os.environ["CK_APP_NAME"] = "CachedKernelApp"
    settings_instance = get_settings()
    print(f"App Name from get_settings(): {settings_instance.app_name}") # Will be "CachedKernelApp"

    # Call again to show it's cached (would be same object ID if we could print it easily)
    settings_instance_2 = get_settings()
    print(f"App Name from get_settings() again: {settings_instance_2.app_name}") # Still "CachedKernelApp"

    del os.environ["CK_APP_NAME"] # Cleanup demo env var

# End of config.py
# Ensure this replaces the old "config.py loaded" print and comments.
