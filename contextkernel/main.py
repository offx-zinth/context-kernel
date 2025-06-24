import logging
import uvicorn # For running the FastAPI app
import sys
from typing import Optional, Any, Dict # For type hinting

# ContextKernel imports
from contextkernel.utils.config import get_settings, AppSettings, ConfigurationError
from contextkernel.interfaces.api import app as fastapi_app
from contextkernel.utils.state_manager import (
    AbstractStateManager,
    RedisStateManager,
    InMemoryStateManager,
)
# Assuming actual ContextAgent and other core logic components might not be fully implemented/available
# For now, we'll use placeholders or import if they exist.
# from contextkernel.core_logic.context_agent import ContextAgent # Actual import
# from contextkernel.core_logic.llm_listener import LLMListener     # Actual import
# from contextkernel.core_logic.llm_retriever import LLMRetriever # Actual import
# from contextkernel.core_logic.summarizer import Summarizer       # Actual import

# Import module configurations (already in config.py but good for explicitness here if needed)
from contextkernel.core_logic import NLPConfig # Updated import
from contextkernel.core_logic.llm_listener import LLMListenerConfig
from contextkernel.core_logic.llm_retriever import LLMRetrieverConfig
from contextkernel.core_logic.summarizer import SummarizerConfig


# Mock LLM client for now
from contextkernel.tests.mocks.mock_llm import MockLLM


# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Placeholder Core Logic Classes ---
# These would be replaced by actual imports from core_logic module
# For now, they are defined here to make main.py runnable and demonstrate wiring.

class Summarizer:
    def __init__(self, config: SummarizerConfig, llm_client: Any):
        self.config = config
        self.llm_client = llm_client
        logger.info(f"Placeholder Summarizer initialized with model: {config.hf_abstractive_model_name} and LLM: {llm_client.__class__.__name__}")

class LLMRetriever:
    def __init__(self, config: LLMRetrieverConfig, llm_client: Any):
        self.config = config
        self.llm_client = llm_client
        logger.info(f"Placeholder LLMRetriever initialized with embedding model: {config.embedding_model_name} and LLM: {llm_client.__class__.__name__}")

class LLMListener:
    def __init__(self, config: LLMListenerConfig, llm_client: Any):
        self.config = config
        self.llm_client = llm_client
        logger.info(f"Placeholder LLMListener initialized with entity model: {config.entity_extraction_model_name} and LLM: {llm_client.__class__.__name__}")

class ContextAgent:
    def __init__(self,
                 nlp_config: NLPConfig, # Updated from config: ContextAgentConfig
                 stm: AbstractStateManager,
                 ltm: AbstractStateManager, # Placeholder for dedicated LTM
                 graph_db: Any, # Placeholder for Graph DB interface
                 llm_client: Any,
                 summarizer_service: Summarizer,
                 retriever_service: LLMRetriever,
                 listener_service: LLMListener):
        self.config = config
        self.stm = stm
        self.ltm = ltm
        self.graph_db = graph_db
        self.llm_client = llm_client
        self.summarizer = summarizer_service
        self.retriever = retriever_service
        self.listener = listener_service
        logger.info(f"Placeholder ContextAgent initialized with LLM: {llm_client.__class__.__name__}, STM: {stm.__class__.__name__}, LTM: {ltm.__class__.__name__}")

    # Mock methods that api.py expects (copied from api.py mock for consistency for now)
    async def handle_chat(self, chat_message: Any, session_id: Optional[str], state_manager: AbstractStateManager) -> Any:
        from contextkernel.interfaces.api import ContextResponse # Avoid circular import at top level
        logger.info(f"ContextAgent: Handling chat for user {chat_message.user_id} in session {session_id}")
        new_context_id = session_id or f"session_{chat_message.user_id}_{abs(hash(chat_message.message))}"
        response_data = {"reply": f"Agent echo: {chat_message.message}", "history": []}

        current_state = await state_manager.get_state(new_context_id)
        if current_state:
            response_data["history"] = current_state.get("history", [])
        response_data["history"].append(chat_message.message)
        await state_manager.save_state(new_context_id, {"history": response_data["history"]})
        return ContextResponse(context_id=new_context_id, data=response_data)

    async def ingest_data(self, data: Any, settings: AppSettings) -> Any:
        from contextkernel.interfaces.api import IngestResponse # Avoid circular import
        logger.info(f"ContextAgent: Ingesting data from {data.source_uri or 'direct content'}")
        doc_id = data.document_id or f"doc_{abs(hash(data.source_uri or data.content))}"
        return IngestResponse(document_id=doc_id, status="success", message="Agent: Data ingested.")

    async def get_context_details(self, context_id: str, state_manager: AbstractStateManager) -> Optional[Any]:
        from contextkernel.interfaces.api import ContextResponse # Avoid circular import
        logger.info(f"ContextAgent: Retrieving context for {context_id}")
        state = await state_manager.get_state(context_id)
        if state:
            return ContextResponse(context_id=context_id, data=state)
        return None


# --- Global State Manager ---
# This will be initialized in main() and attached to app.state
# It needs to be accessible to the shutdown handler.
state_manager_instance: Optional[AbstractStateManager] = None

def setup_logging(log_level_str: str, debug_mode: bool):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    if debug_mode and log_level > logging.DEBUG:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")


@fastapi_app.on_event("shutdown")
async def shutdown_event():
    global state_manager_instance
    logger.info("Application shutdown event triggered.")
    if state_manager_instance and hasattr(state_manager_instance, 'close'):
        logger.info(f"Closing StateManager ({state_manager_instance.__class__.__name__})...")
        await state_manager_instance.close()
        logger.info("StateManager closed.")
    # Add closing for other resources like LLM clients if they have a close method
    if fastapi_app.state.llm_client and hasattr(fastapi_app.state.llm_client, 'close'):
        logger.info("Closing LLM client...")
        # Assuming close is synchronous for mock, make async if real client needs it
        if hasattr(fastapi_app.state.llm_client.close, "__call__"):
             fastapi_app.state.llm_client.close()
        logger.info("LLM client closed.")


def main():
    global state_manager_instance
    # 1. Load Application Configuration
    try:
        config: AppSettings = get_settings()
    except ConfigurationError as e:
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Fatal error: Could not load application settings. {e}", exc_info=True)
        sys.exit(1)

    # 2. Setup Logging
    setup_logging(log_level_str=config.server.log_level, debug_mode=config.debug_mode)

    logger.info(f"Starting {config.app_name} v{config.version}")
    logger.info(f"Debug Mode: {'Enabled' if config.debug_mode else 'Disabled'}")
    logger.info(f"State Manager Type: {config.state_manager_type}")

    # 3. Initialize StateManager
    try:
        if config.state_manager_type.lower() == "redis":
            logger.info(f"Initializing RedisStateManager with host: {config.redis.host}, port: {config.redis.port}, db: {config.redis.db}")
            password = config.redis.password.get_secret_value() if config.redis.password else None
            state_manager_instance = RedisStateManager(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=password
            )
        elif config.state_manager_type.lower() == "in_memory":
            logger.info("Initializing InMemoryStateManager.")
            state_manager_instance = InMemoryStateManager()
        else:
            logger.warning(f"Unknown state_manager_type: '{config.state_manager_type}'. Defaulting to InMemoryStateManager.")
            state_manager_instance = InMemoryStateManager()
        logger.info(f"{state_manager_instance.__class__.__name__} initialized successfully.")
    except ConfigurationError as e: # Catch errors from RedisStateManager init specifically
        logger.error(f"Fatal error: Could not initialize StateManager. {e}", exc_info=True)
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Fatal error: Unexpected error initializing StateManager. {e}", exc_info=True)
        sys.exit(1)


    # 4. Initialize LLM Client(s)
    logger.info("Initializing LLM client...")
    if config.llm.api_key:
        logger.info(f"LLM API Key found for provider: {config.llm.provider}. MockLLM will use it (if configured to).")
        # In a real scenario, you'd initialize the actual LLM client here, e.g.:
        # from some_llm_library import ActualLLMClient
        # llm_client = ActualLLMClient(api_key=config.llm.api_key.get_secret_value(), model=config.llm.model)
        llm_client = MockLLM(api_key=config.llm.api_key.get_secret_value(), model=config.llm.model or "mock-model-pro")
    else:
        logger.warning("LLM API Key not configured. Using MockLLM without API key.")
        llm_client = MockLLM(model=config.llm.model or "mock-model-basic")
    logger.info(f"{llm_client.__class__.__name__} initialized for LLM operations.")


    # 5. Initialize Core Logic Components
    logger.info("Initializing core logic components (Summarizer, Retriever, Listener)...")
    summarizer = Summarizer(config=config.summarizer, llm_client=llm_client)
    retriever = LLMRetriever(config=config.retriever, llm_client=llm_client)
    listener = LLMListener(config=config.listener, llm_client=llm_client)
    logger.info("Core logic components initialized.")

    # 6. Initialize ContextAgent
    logger.info("Initializing ContextAgent...")
    # For LTM and GraphDB, using InMemoryStateManagers as placeholders for now
    # In a production system, these would be specific implementations (e.g., a vector DB for LTM).
    ltm_placeholder = InMemoryStateManager()
    graph_db_placeholder = InMemoryStateManager() # Or a more specific mock/client for a graph database

    # ContextAgent is initialized with its dependencies. This pattern is key for testability,
    # as each dependency (STM, LTM, LLM, services) can be replaced with a mock during testing.
    context_agent = ContextAgent(
        nlp_config=config.nlp, # Updated from config.agent
        stm=state_manager_instance, # Using the chosen state manager for STM
        ltm=ltm_placeholder,       # Placeholder LTM
        graph_db=graph_db_placeholder, # Placeholder GraphDB
        llm_client=llm_client,
        summarizer_service=summarizer,
        retriever_service=retriever,
        listener_service=listener
    )
    logger.info("ContextAgent initialized.")

    # 7. Attach instances to fastapi_app.state for Dependency Injection
    logger.info("Attaching core components to FastAPI app state...")
    fastapi_app.state.settings = config
    fastapi_app.state.state_manager = state_manager_instance
    fastapi_app.state.context_agent = context_agent
    fastapi_app.state.llm_client = llm_client # Make LLM client accessible if needed elsewhere
    logger.info("Core components attached to app state.")

    # 8. Start the API Server (if enabled)
    if config.server.enabled:
        logger.info(
            f"API Server enabled. Starting Uvicorn on "
            f"{config.server.host}:{config.server.port} "
            f"with log level: {config.server.log_level.lower()}"
        )
        try:
            uvicorn.run(
                app=fastapi_app,
                host=config.server.host,
                port=config.server.port,
                log_level=config.server.log_level.lower(),
                # reload=config.debug_mode # Useful for development
            )
        except Exception as e:
            logger.critical(f"Failed to start Uvicorn server: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("API Server is disabled in the configuration. Application will not start a web server.")

    logger.info(f"{config.app_name} has finished or server has stopped.")


if __name__ == "__main__":
    main()

[end of contextkernel/main.py]
