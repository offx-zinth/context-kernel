import logging
import uvicorn # For running the FastAPI app
import sys

# ContextKernel imports
from contextkernel.utils.config import get_settings, AppSettings
from contextkernel.interfaces.api import app as fastapi_app

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def setup_logging(log_level_str: str, debug_mode: bool):
    """
    Configures basic logging for the application.
    """
    # Convert string log level from config to logging module's level constants
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    if debug_mode and log_level > logging.DEBUG:
        # If debug mode is on, ensure log level is at least DEBUG
        log_level = logging.DEBUG

    # Basic configuration for the root logger
    # This will apply to all loggers unless they have specific handlers
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to console
            # You could add logging.FileHandler("app.log") here for file logging
        ]
    )

    # Example of setting a more specific log level for a library (e.g., uvicorn)
    # uvicorn_logger = logging.getLogger("uvicorn.access")
    # uvicorn_logger.setLevel(logging.WARNING) # Quieten uvicorn access logs if needed

    logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")


def main():
    """
    Main entry point for the ContextKernel application.
    Initializes configurations, sets up logging, and starts the API server if enabled.
    """
    # 1. Load Application Configuration
    try:
        config: AppSettings = get_settings()
    except Exception as e:
        # Use a basic logger if config loading fails before full setup
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Fatal error: Could not load application settings. {e}", exc_info=True)
        sys.exit(1) # Exit if configuration is essential and fails to load

    # 2. Setup Logging
    # Uses log_level from server settings if available, otherwise from app_settings or a default.
    # Server log_level is often the one you want for the running Uvicorn instance.
    setup_logging(log_level_str=config.server.log_level, debug_mode=config.debug_mode)

    logger.info(f"Starting {config.app_name} v{config.version}")
    logger.info(f"Debug Mode: {'Enabled' if config.debug_mode else 'Disabled'}")

    # 3. Initialize Core Components (Conceptual)
    # Here you would initialize and wire up the main parts of your application.
    # For example:
    #   - Memory Systems (STM, LTM, GraphDB, RawCache)
    #     logger.info("Initializing memory systems...")
    #     stm = ShortTermMemory(...)
    #     ltm = LongTermMemory(db_url=config.database.url, ...)
    #     graph_db = GraphDatabase(db_url=config.database.graph_url, ...) # Assuming graph_url in db settings
    #
    #   - LLM Clients
    #     logger.info("Initializing LLM clients...")
    #     if config.llm.api_key:
    #         llm_client = OpenAIChat(api_key=config.llm.api_key.get_secret_value(), model=config.llm.model)
    #     else:
    #         logger.warning("LLM API Key not configured. LLM functionalities may be limited.")
    #         llm_client = None # Or a dummy client
    #
    #   - ContextAgent (The core orchestrator)
    #     logger.info("Initializing ContextAgent...")
    #     context_agent = ContextAgent(
    #         stm=stm,
    #         ltm=ltm,
    #         graph_db=graph_db,
    #         llm_client=llm_client,
    #         config=config # Pass relevant parts of config
    #     )
    #
    #   - Make components available to the FastAPI app
    #     This can be done via FastAPI app state, dependency injection, or by other means.
    #     Example using app.state:
    #     fastapi_app.state.context_agent = context_agent
    #     fastapi_app.state.config = config
    #     logger.info("Core components initialized and wired.")
    #
    # For now, these are placeholders.

    # 4. Start the API Server (if enabled)
    if config.server.enabled:
        logger.info(
            f"API Server enabled. Starting Uvicorn on "
            f"{config.server.host}:{config.server.port} "
            f"with log level: {config.server.log_level.lower()}"
        )
        try:
            uvicorn.run(
                app=fastapi_app,  # The FastAPI application instance
                host=config.server.host,
                port=config.server.port,
                log_level=config.server.log_level.lower(),
                # Consider adding other uvicorn settings like workers, reload (for dev), etc.
                # workers=config.server.workers if hasattr(config.server, 'workers') else None,
                # reload=config.debug_mode # Reload only in debug mode
            )
        except Exception as e:
            logger.critical(f"Failed to start Uvicorn server: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("API Server is disabled in the configuration. Application will not start a web server.")
        logger.info("Consider enabling the server or implementing a CLI mode if desired.")
        # If you had a CLI mode, you might invoke it here:
        # run_cli_mode(config, context_agent)

    logger.info(f"{config.app_name} has finished or server has stopped.")


if __name__ == "__main__":
    main()

# End of main.py
# Ensure this replaces the old "main.py loaded" print and comments.
