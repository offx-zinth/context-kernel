import logging
import uvicorn # For running the FastAPI app
import sys
import asyncio # Added for asyncio functionality
from typing import Optional, Any, Dict, List # Updated List import
from dataclasses import dataclass # For StreamMessage
import datetime # For PlaceholderStructuredInsight, and raw_data_id generation in loop

# ContextKernel imports
from contextkernel.utils.config import get_settings, AppSettings, ConfigurationError
from contextkernel.interfaces.api import app as fastapi_app
from contextkernel.utils.state_manager import (
    AbstractStateManager,
    # RedisStateManager, # Not directly used in main_async after refactor
    # InMemoryStateManager,
)
# Actual core logic components will be imported
from contextkernel.core_logic.llm_listener import ContextAgent, ContextAgentConfig, StructuredInsight # LLMListener renamed to ContextAgent
from contextkernel.core_logic.llm_retriever import LLMRetriever, LLMRetrieverConfig
from contextkernel.core_logic.hallucination_detector import HallucinationDetector
from contextkernel.memory_system.memory_manager import MemoryManager
from contextkernel.core_logic.chunker import SemanticChunker
from contextkernel.core_logic.nlp_utils import initialize_matcher, NLPConfig # For SemanticChunker setup
import spacy # For SemanticChunker setup

# Import interfaces for MemoryManager dependencies
from contextkernel.memory_system.graph_db import GraphDBInterface
from contextkernel.memory_system.ltm import LTMInterface
from contextkernel.memory_system.stm import STMInterface
from contextkernel.memory_system.raw_cache import RawCacheInterface


# Mock LLM client for now (or actual client if configured)
from contextkernel.tests.mocks.mock_llm import MockLLM # DEV only, replace with actual

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- StreamMessage Data Class ---
@dataclass
class StreamMessage:
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


# --- Placeholder Core Logic Classes ---
class PlaceholderGraphDB(GraphDBInterface):
    async def create_node(self, node_id: str, data: Dict, label: Optional[str] = None): logger.info(f"MockGraphDB: Create node {node_id}")
    async def get_node(self, node_id: str) -> Optional[Dict]: logger.info(f"MockGraphDB: Get node {node_id}"); return None
    async def update_node(self, node_id: str, data: Dict): logger.info(f"MockGraphDB: Update node {node_id}")
    async def delete_node(self, node_id: str): logger.info(f"MockGraphDB: Delete node {node_id}")
    async def create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Optional[Dict] = None): logger.info(f"MockGraphDB: Create relationship {source_id}-{rel_type}->{target_id}")
    async def query(self, cypher_query: str, params: Optional[Dict] = None) -> Any: logger.info(f"MockGraphDB: Query {cypher_query}"); return []
    async def ensure_source_document_node(self, document_id: str, properties: Dict): logger.info(f"MockGraphDB: Ensure SourceDocument node {document_id} with props {list(properties.keys())}")
    async def add_memory_fragment_link(self, document_id: str, fragment_id: str, fragment_main_label: str, relationship_type: str, fragment_properties: Dict): logger.info(f"MockGraphDB: Add fragment link {document_id} -> {fragment_id} ({fragment_main_label} via {relationship_type})")
    async def add_entities_to_document(self, document_id: str, entities: list): logger.info(f"MockGraphDB: Add {len(entities)} entities to {document_id}")
    async def add_relations_to_document(self, document_id: str, relations: list): logger.info(f"MockGraphDB: Add {len(relations)} relations to {document_id}")

class PlaceholderLTM(LTMInterface):
    async def store_embedding(self, doc_id: str, embedding: Any, metadata: Dict): logger.info(f"MockLTM: Store embedding {doc_id}")
    async def get_embedding(self, doc_id: str) -> Optional[Any]: logger.info(f"MockLTM: Get embedding {doc_id}"); return None
    async def search_similar(self, query_embedding: Any, top_k: int = 5) -> list: logger.info(f"MockLTM: Search similar for query (embedding preview: {str(query_embedding)[:30]}...) top_k={top_k}"); return []
    async def delete_document(self, memory_id: str): logger.info(f"MockLTM: Delete document (embedding and raw) for {memory_id}")
    async def update_document(self, memory_id: str, updates: Dict): logger.info(f"MockLTM: Update document (embedding and metadata) for {memory_id} with updates {list(updates.keys())}")
    # LTMInterface might need store_document, get_document, etc. if MemoryManager calls those directly.
    # For now, assuming MemoryManager uses store_embedding, search_similar, delete_document, update_document.
    # Adding save_document as it's called by MemoryManager.store
    async def save_document(self, doc_id: str, text_content: str, embedding: List[float], metadata: Optional[Dict[str, Any]]=None) -> None: logger.info(f"MockLTM: Save document {doc_id}, text: '{text_content[:50]}...'")


class PlaceholderSTM(STMInterface):
    async def store_summary(self, summary_id: str, summary_text: str, metadata: Dict): logger.info(f"MockSTM: Store summary {summary_id}")
    async def get_summary(self, summary_id: str) -> Optional[str]: logger.info(f"MockSTM: Get summary {summary_id}"); return None
    async def delete_summary(self, summary_id: str): logger.info(f"MockSTM: Delete summary {summary_id}")
    async def update_summary(self, summary_id: str, updates: Dict): logger.info(f"MockSTM: Update summary {summary_id}")
    # Adding save_summary as it's called by MemoryManager.store
    async def save_summary(self, summary_id: str, summary_obj: Any, metadata: Optional[Dict[str, Any]]=None) -> None: logger.info(f"MockSTM: Save summary {summary_id}, obj_type: {type(summary_obj)}")


class PlaceholderRawCache(RawCacheInterface):
    async def store_raw_data(self, doc_id: str, data: Any, metadata: Optional[Dict] = None): logger.info(f"MockRawCache: Store raw data {doc_id}")
    async def get_raw_data(self, doc_id: str) -> Optional[Any]: logger.info(f"MockRawCache: Get raw data {doc_id}"); return None
    async def delete_raw_data(self, doc_id: str): logger.info(f"MockRawCache: Delete raw data {doc_id}")


# --- Cognitive Loop Functions ---
async def write_and_verify_loop(
    queue: asyncio.Queue,
    context_agent: ContextAgent, # Updated type hint
    detector: HallucinationDetector,
    mem_manager: MemoryManager
):
    logger.info("Write-and-Verify Loop started.")
    while True:
        message: StreamMessage = await queue.get()
        logger.info(f"[WriteLoop] Received message from source '{message.source}': {message.content[:50]}...")
        try:
            # 1. Process data with ContextAgent
            structured_insights: List[StructuredInsight] = await context_agent.process_data(
                raw_data_content=message.content,
                context_instructions=message.metadata.get("context_instructions") if message.metadata else None
            )
            logger.info(f"[WriteLoop] ContextAgent processed message, generated {len(structured_insights)} insight(s).")

            if not structured_insights:
                logger.info(f"[WriteLoop] No insights generated for message: {message.content[:50]}...")
                queue.task_done()
                continue

            for insight_idx, current_insight in enumerate(structured_insights):
                # Determine content for validation (e.g., summary or original preview)
                content_to_validate = current_insight.summary.text if current_insight.summary and current_insight.summary.text else current_insight.source_data_preview

                if not content_to_validate:
                    logger.warning(f"[WriteLoop] Insight {insight_idx+1} has no content (summary/preview) for validation. Skipping.")
                    continue

                logger.info(f"[WriteLoop] Validating insight {insight_idx+1}/{len(structured_insights)}: Preview '{content_to_validate[:50]}...'")

                # 2. Detect hallucinations
                validation_result = await detector.detect(content_to_validate)
                logger.info(f"[WriteLoop] Validation for insight {insight_idx+1}: {validation_result.is_valid}. Explanation: {validation_result.explanation}")

                if not validation_result.is_valid:
                    logger.warning(f"[WriteLoop] Hallucination detected for insight {insight_idx+1}. Validation: {validation_result.explanation}. Skipping storage.")
                    # Future: await mem_manager.store_hallucination_event(message, current_insight, validation_result)
                    continue

                # 3. Store the (validated) StructuredInsight
                # Construct a raw_data_id for insights from queued messages.
                msg_id_part = message.metadata.get("message_id", f"{message.source}_{message.metadata.get('received_at', str(datetime.datetime.utcnow().timestamp()))}") if message.metadata else f"{message.source}_{str(datetime.datetime.utcnow().timestamp())}"
                current_insight.raw_data_id = f"insight_msg_{msg_id_part}_chunk_{insight_idx}"

                logger.info(f"[WriteLoop] Storing insight {insight_idx+1} (ID: {current_insight.raw_data_id}) via MemoryManager.")
                await mem_manager.store(current_insight)
                logger.info(f"[WriteLoop] Insight {current_insight.raw_data_id} stored successfully.")

        except Exception as e:
            logger.error(f"[WriteLoop] Error processing message '{message.content[:50]}...': {e}", exc_info=True)
        finally:
            queue.task_done()

async def read_and_inject_loop(
    queue: asyncio.Queue, # This loop also listens to the same queue
    retriever: LLMRetriever
):
    logger.info("Read-and-Inject Loop started.")
    while True:
        message: StreamMessage = await queue.get()
        logger.info(f"[ReadLoop] Received message from source '{message.source}': {message.content[:50]}...")
        try:
            # 1. Retrieve relevant context using LLMRetriever
            task_description = message.metadata.get("task_description") if message.metadata else None
            filters = message.metadata.get("retrieval_filters") if message.metadata else None

            retrieval_response = await retriever.retrieve(
                query=message.content,
                task_description=task_description,
                filters=filters
                # top_k can be configured in LLMRetrieverConfig or passed here
            )
            logger.info(f"[ReadLoop] Retriever found {len(retrieval_response.items)} items. Message: {retrieval_response.message}")

            # 2. Inject context (for now, print/log)
            if retrieval_response.items:
                logger.info(f"[ReadLoop] Context for '{message.content[:50]}...':")
                for item_idx, item in enumerate(retrieval_response.items):
                    score_display = f"{item.score:.4f}" if item.score is not None else "N/A"
                    logger.info(f"  Item {item_idx+1}: Source='{item.source}', Score={score_display}, Content='{str(item.content)[:100]}...'")
                    # Example: await context_broadcaster.send_context(message.metadata.get("session_id"), item)
            else:
                logger.info(f"[ReadLoop] No relevant context found by retriever for '{message.content[:50]}...'")

        except Exception as e:
            logger.error(f"[ReadLoop] Error processing message for retrieval: {e}", exc_info=True)
        finally:
            queue.task_done()


# --- Global State Manager ---
state_manager_instance: Optional[AbstractStateManager] = None # Remains for potential other uses, but not central to loops

def setup_logging(log_level_str: str, debug_mode: bool):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    if debug_mode and log_level > logging.DEBUG:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s",
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

    if hasattr(fastapi_app.state, 'llm_client') and fastapi_app.state.llm_client and hasattr(fastapi_app.state.llm_client, 'close'):
        logger.info("Closing LLM client...")
        if hasattr(fastapi_app.state.llm_client.close, "__call__"):
            if asyncio.iscoroutinefunction(fastapi_app.state.llm_client.close):
                await fastapi_app.state.llm_client.close()
            else:
                fastapi_app.state.llm_client.close()
        logger.info("LLM client closed.")

    if hasattr(fastapi_app.state, 'cognitive_tasks'):
        logger.info("Cancelling cognitive loop tasks...")
        for task in fastapi_app.state.cognitive_tasks:
            if not task.done(): # Check if task is not already done
                 task.cancel()
        try:
            await asyncio.gather(*fastapi_app.state.cognitive_tasks, return_exceptions=True)
            logger.info("Cognitive loop tasks awaited after cancellation.")
        except asyncio.CancelledError:
            logger.info("Cognitive loop tasks were cancelled as expected.")
        except Exception as e:
            logger.error(f"Error during cognitive_tasks gather on shutdown: {e}", exc_info=True)

    # Close Spacy model if loaded and attached
    if hasattr(fastapi_app.state, 'nlp_spacy') and fastapi_app.state.nlp_spacy is not None:
        logger.info("Closing Spacy model (conceptual, as spacy.Language objects don't have explicit close).")
        fastapi_app.state.nlp_spacy = None # Allow garbage collection

    logger.info("FastAPI app shutdown sequence complete.")


async def main_async():
    global state_manager_instance
    try:
        config: AppSettings = get_settings()
    except ConfigurationError as e:
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Fatal error: Could not load application settings. {e}", exc_info=True)
        sys.exit(1)

    setup_logging(log_level_str=config.server.log_level, debug_mode=config.debug_mode)
    logger.info(f"Starting {config.app_name} v{config.version} (Async Mode)")

    if config.llm.api_key:
        llm_client = MockLLM(api_key=config.llm.api_key.get_secret_value(), model=config.llm.model or "mock-model-pro")
    else:
        llm_client = MockLLM(model=config.llm.model or "mock-model-basic")
    logger.info(f"{llm_client.__class__.__name__} initialized for LLM operations.")
    fastapi_app.state.llm_client = llm_client # Attach to app state for potential use/shutdown

    # Initialize DB Stubs
    graph_db_instance = PlaceholderGraphDB()
    ltm_instance = PlaceholderLTM()
    stm_instance = PlaceholderSTM()
    raw_cache_instance = PlaceholderRawCache()

    memory_manager = MemoryManager(
        graph_db=graph_db_instance,
        ltm=ltm_instance,
        stm=stm_instance,
        raw_cache=raw_cache_instance
    )
    logger.info("MemoryManager initialized with mock DBs.")

    # Initialize SemanticChunker dependencies
    if not hasattr(config, 'nlp_settings') or not isinstance(config.nlp_settings, NLPConfig):
        logger.warning("AppSettings 'nlp_settings' not found or not NLPConfig. Using default NLPConfig.")
        nlp_config = NLPConfig()
    else:
        nlp_config = config.nlp_settings

    nlp_spacy = None
    try:
        nlp_spacy = spacy.load(nlp_config.spacy_model_name)
        logger.info(f"Spacy model '{nlp_config.spacy_model_name}' loaded.")
        fastapi_app.state.nlp_spacy = nlp_spacy # For potential shutdown
    except OSError:
        logger.error(f"Spacy model '{nlp_config.spacy_model_name}' not found. Chunker functionality will be limited.")

    spacy_matcher = initialize_matcher(nlp_spacy) if nlp_spacy else None
    if spacy_matcher: logger.info("Spacy Matcher initialized.")

    semantic_chunker = SemanticChunker(
        nlp_model=nlp_spacy,
        matcher=spacy_matcher,
        intent_classifier=None, # Using None for intent_classifier for now
        use_spacy_matcher_first=nlp_config.use_spacy_matcher_first,
        intent_candidate_labels=nlp_config.intent_candidate_labels,
        default_intent_confidence=nlp_config.default_intent_confidence,
        high_confidence_threshold=nlp_config.high_confidence_threshold
    )
    logger.info("SemanticChunker initialized.")

    # Initialize LLMRetriever (used by HallucinationDetector and ReadLoop)
    # Assuming retriever_config is named 'retriever' in AppSettings
    if not hasattr(config, 'retriever') or not isinstance(config.retriever, LLMRetrieverConfig):
        logger.warning("AppSettings 'retriever' settings not found or not LLMRetrieverConfig. Using default LLMRetrieverConfig.")
        retriever_config = LLMRetrieverConfig()
    else:
        retriever_config = config.retriever

    llm_retriever = LLMRetriever(
        retriever_config=retriever_config,
        ltm_interface=ltm_instance,
        stm_interface=stm_instance,
        graphdb_interface=graph_db_instance,
        query_llm=llm_client
    )
    logger.info("LLMRetriever initialized.")

    hallucination_detector = HallucinationDetector(llm_client=llm_client, retriever=llm_retriever)
    logger.info("HallucinationDetector initialized.")

    # Initialize ContextAgent
    # Assuming context_agent config is named 'context_agent' in AppSettings
    if not hasattr(config, 'context_agent') or not isinstance(config.context_agent, ContextAgentConfig):
        logger.warning("AppSettings 'context_agent' settings not found or not ContextAgentConfig. Using default ContextAgentConfig.")
        context_agent_config_instance = ContextAgentConfig()
    else:
        context_agent_config_instance = config.context_agent

    context_agent_processor = ContextAgent(
        config=context_agent_config_instance,
        chunker=semantic_chunker,
        hallucination_detector=hallucination_detector,
        memory_manager=memory_manager,
        llm_client=llm_client
    )
    logger.info("ContextAgent initialized.")

    conversation_queue = asyncio.Queue()
    fastapi_app.state.conversation_queue = conversation_queue
    fastapi_app.state.settings = config # Make settings available to API endpoints
    # Attach other components if needed by API endpoints directly (besides loops)
    fastapi_app.state.context_agent_for_ingest = context_agent_processor # For /ingest

    logger.info("Creating cognitive loop tasks...")
    write_task = asyncio.create_task(
        write_and_verify_loop(
            queue=conversation_queue,
            context_agent=context_agent_processor,
            detector=hallucination_detector,
            mem_manager=memory_manager
        )
    )
    read_task = asyncio.create_task(
        read_and_inject_loop(
            queue=conversation_queue,
            retriever=llm_retriever
        )
    )
    fastapi_app.state.cognitive_tasks = [write_task, read_task]
    logger.info("Cognitive loop tasks created.")

    server_task = None
    if config.server.enabled:
        logger.info(f"API Server enabled. Starting Uvicorn on {config.server.host}:{config.server.port}")
        server_config = uvicorn.Config(
            app=fastapi_app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level.lower(),
            lifespan="on"
        )
        server = uvicorn.Server(server_config)
        server_task = asyncio.create_task(server.serve())
        logger.info("Uvicorn server task created.")
    else:
        logger.info("API Server is disabled. Only cognitive loops will run.")

    tasks_to_gather = [task for task in [write_task, read_task, server_task] if task is not None]
    logger.info(f"Running main event loop with {len(tasks_to_gather)} tasks.")
    try:
        await asyncio.gather(*tasks_to_gather)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.critical(f"Critical error in main asyncio.gather: {e}", exc_info=True)
    finally:
        logger.info(f"{config.app_name} main loop finished. Application shutting down.")
        # FastAPI shutdown event handles task cancellations and resource closing.

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Application terminated by user (KeyboardInterrupt in asyncio.run).")
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logger.critical(f"Unhandled exception in asyncio.run: {e}", exc_info=True)
        sys.exit(1)
