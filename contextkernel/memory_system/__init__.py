import asyncio
import logging
import uuid # For generating unique IDs if needed
from typing import Optional, Any, Dict, List # Added Dict, List

from contextkernel.utils.config import AppSettings
from .stm import STM
from .ltm import LTM
from .graph_db import GraphDB
from .raw_cache import RawCache
from .graph_indexer import GraphIndexer
from ....memory_manager import MemoryManager # Adjusted import path

# Import client libraries
from redis.asyncio import Redis as RedisClient
from neo4j import AsyncGraphDatabase, AsyncDriver

# Placeholder types for clients not having a specific library used directly here yet
# These would be more concrete if specific SDKs were used for instantiation here.
NLPClientType = Any
EmbeddingModelClientType = Any
VectorDBClientType = Any
RawContentStoreClientType = Any


logger = logging.getLogger(__name__)

class MemoryKernel:
    _instance: Optional['MemoryKernel'] = None
    _clients_initialized: bool = False
    _app_settings: Optional[AppSettings] = None

    # Declare client attributes for type hinting and clarity
    redis_client: RedisClient
    neo4j_driver: AsyncDriver
    memory_manager: MemoryManager # Added type hint
    # Generic clients for services; specific instances depend on AppSettings
    shared_nlp_client: NLPClientType
    shared_embedding_client: EmbeddingModelClientType
    vector_db_client: VectorDBClientType
    raw_content_store_client: RawContentStoreClientType


    @classmethod
    def get_instance(cls, app_settings: Optional[AppSettings] = None) -> 'MemoryKernel':
        if cls._instance is None:
            if app_settings is None:
                raise ValueError("AppSettings must be provided to create the first MemoryKernel instance.")
            logger.info("Creating new MemoryKernel instance.")
            cls._instance = cls(app_settings=app_settings)
        elif app_settings is not None and cls._app_settings is not app_settings:
            # This case handles if get_instance is called again with different settings.
            # Depending on desired behavior, could raise error, reinitialize, or ignore.
            # For now, let's log a warning and return the existing instance.
            logger.warning("MemoryKernel.get_instance() called with new AppSettings, but an instance already exists. "
                           "Returning the existing instance. Re-initialization is not supported via get_instance().")
        return cls._instance

    def __init__(self, app_settings: AppSettings):
        if MemoryKernel._instance is not None and MemoryKernel._instance is not self:
             # This check ensures that if __init__ is called directly after _instance is set,
             # it doesn't create a different object if not through get_instance logic.
             # However, standard singleton pattern usually makes __init__ raise error if _instance exists.
             # For this refactor, get_instance is the entry point.
            raise RuntimeError("MemoryKernel is a singleton, use MemoryKernel.get_instance(app_settings).")

        logger.info("Initializing MemoryKernel with provided AppSettings...")
        self._app_settings = app_settings # Store settings

        # Initialize clients first based on AppSettings
        # These clients will be passed to the components
        self._initialize_clients(app_settings)

        # Instantiate components in order of dependency, passing configs and clients
        logger.info("Instantiating MemoryKernel components...")

        self.raw_cache = RawCache(
            config=app_settings.redis_config,
            client=self.redis_client # Main Redis client
        )

        self.graph_db = GraphDB(
            config=app_settings.neo4j_config,
            driver=self.neo4j_driver
        )

        # LTM requires several clients and its embedding cache (can be self.raw_cache or a dedicated one)
        # For LTM's raw content store, decide based on a hypothetical config in AppSettings
        # or default to one (e.g., FileSystemConfig if S3Config is not fully set up)
        # For this example, let's assume AppSettings has a way to determine raw_content_store_type or provides both configs
        # and MemoryKernel picks one. Let's say it defaults to filesystem for simplicity here.
        # This part needs careful handling of which raw content store to use.
        # For now, let's assume a FileSystemConfig is primary for LTM's raw store.
        # A more robust way would be:
        # if app_settings.raw_content_store_type == 's3':
        #    active_raw_content_config = app_settings.s3_config
        #    active_raw_content_client = self.s3_client # (self.s3_client would need to be initialized)
        # else: # filesystem
        #    active_raw_content_config = app_settings.filesystem_config
        #    active_raw_content_client = self.raw_content_store_client # (initialized as FileSystem client placeholder)

        self.ltm = LTM(
            vector_db_config=app_settings.vector_db_config,
            vector_db_client=self.vector_db_client, # Placeholder, to be initialized in _initialize_clients
            raw_content_store_config=app_settings.filesystem_config, # Example: defaulting to filesystem
            raw_content_store_client=self.raw_content_store_client, # Placeholder
            embedding_config=app_settings.embedding_config,
            embedding_model_client=self.shared_embedding_client, # Placeholder
            embedding_cache=self.raw_cache # LTM uses the main RawCache for its embeddings
        )

        self.graph_indexer = GraphIndexer(
            graph_db=self.graph_db,
            nlp_config=app_settings.nlp_service_config, # Assuming one NLP config for now
            nlp_client=self.shared_nlp_client, # Placeholder
            embedding_config=app_settings.embedding_config,
            embedding_client=self.shared_embedding_client, # Placeholder, shared with LTM
            embedding_cache=self.raw_cache # GraphIndexer can also use RawCache
        )

        self.stm = STM(
            ltm=self.ltm,
            # Assuming summarizer and intent tagger use the same NLP service config and client for now
            summarizer_config=app_settings.nlp_service_config,
            summarizer_client=self.shared_nlp_client, # Placeholder
            intent_tagger_config=app_settings.nlp_service_config,
            intent_tagger_client=self.shared_nlp_client # Placeholder
        )

        # Instantiate MemoryManager
        self.memory_manager = MemoryManager(
            graph_db=self.graph_db,
            ltm=self.ltm,
            stm=self.stm,
            raw_cache=self.raw_cache
        )
        logger.info("MemoryManager instantiated.")

        logger.info("MemoryKernel components initialized with configurations from AppSettings.")
        MemoryKernel._instance = self # Set the instance after successful initialization

    def _initialize_clients(self, app_settings: AppSettings):
        """Helper method to initialize all necessary clients based on AppSettings."""
        if MemoryKernel._clients_initialized:
            logger.info("Clients already initialized. Skipping re-initialization.")
            return

        logger.info("Initializing shared clients based on AppSettings...")

        # Redis Client (for RawCache and potentially other uses)
        self.redis_client = RedisClient(
            host=app_settings.redis_config.host,
            port=app_settings.redis_config.port,
            password=app_settings.redis_config.password,
            db=app_settings.redis_config.db  # Default DB from config
        )
        logger.info(f"Redis client configured for host {app_settings.redis_config.host}:{app_settings.redis_config.port}, DB {app_settings.redis_config.db}")

        # Neo4j Driver
        self.neo4j_driver = AsyncGraphDatabase.driver(
            app_settings.neo4j_config.uri,
            auth=(app_settings.neo4j_config.user, app_settings.neo4j_config.password)
        )
        logger.info(f"Neo4j driver configured for URI: {app_settings.neo4j_config.uri}")

        # Placeholder/Mock Client Initializations
        # In a real app, these would instantiate actual SDK clients, e.g., OpenAI(), Pinecone(), etc.
        # For this refactoring, we'll assume these are set up as needed by their respective components' main() examples
        # or are simple mock objects if not using actual services.

        # Shared NLP Client (e.g., for GraphIndexer, STM summarizer/tagger)
        # This would depend on app_settings.nlp_service_config.provider
        # Example: if provider is "openai", client = OpenAI(api_key=...)
        self.shared_nlp_client = self._create_placeholder_client("NLP", app_settings.nlp_service_config.provider)
        logger.info(f"Shared NLP client (placeholder) created for provider: {app_settings.nlp_service_config.provider}")

        # Shared Embedding Client (e.g., for LTM, GraphIndexer)
        self.shared_embedding_client = self._create_placeholder_client("Embedding", app_settings.embedding_config.model_name)
        logger.info(f"Shared Embedding client (placeholder) created for model: {app_settings.embedding_config.model_name}")

        # VectorDB Client (for LTM)
        self.vector_db_client = self._create_placeholder_client("VectorDB", app_settings.vector_db_config.type)
        logger.info(f"VectorDB client (placeholder) created for type: {app_settings.vector_db_config.type}")

        # Raw Content Store Client (for LTM) - Example: FileSystem based
        # This logic could be more complex based on AppSettings (e.g. S3 vs. FileSystem)
        if hasattr(app_settings, 's3_config') and app_settings.s3_config.bucket_name: # Basic check for S3 preference
            self.raw_content_store_client = self._create_placeholder_client("S3RawContentStore", app_settings.s3_config.bucket_name)
            logger.info(f"RawContentStore client (placeholder for S3) created for bucket: {app_settings.s3_config.bucket_name}")
        else:
            self.raw_content_store_client = self._create_placeholder_client("FileSystemRawContentStore", app_settings.filesystem_config.base_path)
            logger.info(f"RawContentStore client (placeholder for FileSystem) created for path: {app_settings.filesystem_config.base_path}")

        MemoryKernel._clients_initialized = True

    def _create_placeholder_client(self, client_name: str, config_detail: str) -> Any:
        """Creates a generic placeholder client for demonstration."""
        class PlaceholderClient:
            def __init__(self, name, detail):
                self.name = name
                self.detail = detail
                logger.debug(f"PlaceholderClient: {self.name} ({self.detail}) instantiated.")
            async def boot(self): logger.debug(f"PlaceholderClient {self.name} booted."); return True # Mock boot
            async def shutdown(self): logger.debug(f"PlaceholderClient {self.name} shutdown.") # Mock shutdown
            def __repr__(self): return f"<PlaceholderClient for {self.name} ({self.detail})>"
        return PlaceholderClient(client_name, config_detail)

    async def boot(self):
        """
        Initializes all memory services and verifies client connections.
        """
        logger.info("MemoryKernel booting up...")
        # Verify primary client connections here
        try:
            await self.redis_client.ping()
            logger.info("Redis client connected successfully.")
        except Exception as e:
            logger.error(f"Redis client connection failed during boot: {e}")
            # Depending on policy, might raise error or try to continue
            # For now, just log and continue, components might fail individually

        try:
            await self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j driver connected successfully.")
        except Exception as e:
            logger.error(f"Neo4j driver connection failed during boot: {e}")
            # Log and continue

        # Boot all components
        # The components' boot methods should ideally use their injected clients
        # and configs to perform their specific initializations or health checks.
        component_boots = [
            self.raw_cache.boot(), # RawCache boot now pings Redis
            self.graph_db.boot(),  # GraphDB boot now verifies Neo4j connectivity
            self.ltm.boot(),       # LTM boot can check its clients (RawCache already pinged)
            self.graph_indexer.boot(),
            self.stm.boot()
        ]
        results = await asyncio.gather(*component_boots, return_exceptions=True)

        for result, name in zip(results, ["RawCache", "GraphDB", "LTM", "GraphIndexer", "STM"]):
            if isinstance(result, Exception):
                logger.error(f"Error booting component {name}: {result}")
            else:
                logger.info(f"Component {name} booted successfully.")

        logger.info("MemoryKernel boot process complete.")


    async def shutdown(self):
        """
        Shuts down all memory services and closes client connections created by MemoryKernel.
        """
        logger.info("MemoryKernel shutting down...")

        # Shutdown components first
        component_shutdowns = [
            self.stm.shutdown(),
            self.ltm.shutdown(),
            self.graph_indexer.shutdown(),
            self.graph_db.shutdown(),
            self.raw_cache.shutdown()
        ]
        results = await asyncio.gather(*component_shutdowns, return_exceptions=True)
        for result, name in zip(results, ["STM", "LTM", "GraphIndexer", "GraphDB", "RawCache"]):
            if isinstance(result, Exception):
                logger.error(f"Error shutting down component {name}: {result}")
            else:
                logger.info(f"Component {name} shutdown successfully.")

        # Close clients initialized by MemoryKernel
        logger.info("Closing shared clients...")
        if hasattr(self.redis_client, 'close'): # redis-py client uses close
            await self.redis_client.close()
            logger.info("Redis client closed.")

        if hasattr(self.neo4j_driver, 'close'):
            await self.neo4j_driver.close()
            logger.info("Neo4j driver closed.")

        # Placeholder clients might also have close/shutdown methods
        if hasattr(self.shared_nlp_client, 'shutdown'): await self.shared_nlp_client.shutdown()
        if hasattr(self.shared_embedding_client, 'shutdown'): await self.shared_embedding_client.shutdown()
        if hasattr(self.vector_db_client, 'shutdown'): await self.vector_db_client.shutdown()
        if hasattr(self.raw_content_store_client, 'shutdown'): await self.raw_content_store_client.shutdown()

        MemoryKernel._clients_initialized = False # Reset flag
        logger.info("MemoryKernel shutdown complete.")

async def main():
    # Example Usage for MemoryKernel
    # Note: The get_context and store_context methods have been removed from MemoryKernel.
    # Their logic will be handled by IngestionProcessor and RetrievalProcessor,
    # and MemoryManager will handle the storage details.
    # This main function needs to be updated to reflect those changes.
    logger.info("--- MemoryKernel Example Usage ---")

    # 1. Create AppSettings (loads from environment or defaults)
    # For a self-contained example, we might override some settings,
    # especially paths for FileSystemConfig, FAISS index, etc.
    # For real spaCy/SentenceTransformer models to be used by GraphIndexer/LTM,
    # ensure NLPServiceConfig.model and EmbeddingConfig.model_name are set appropriately.
    # (These are already defaulted in config.py or can be set via environment variables)
    settings = AppSettings()

    # Example: Override specific model names for GraphIndexer and LTM if needed for the test
    # settings.nlp_service_config.model = "en_core_web_sm"
    # settings.embedding_config.model_name = "all-MiniLM-L6-v2"
    # settings.ltm_vector_db_config.params["index_path"] = "temp_ltm_faiss.index" # Example temporary path
    # settings.ltm_raw_content_store_config.base_path = "temp_ltm_raw_content" # Example temporary path

    # For the MemoryKernel main example, we rely on default AppSettings.
    # Real model downloads (spaCy, SentenceTransformers, HuggingFace for STM) will occur
    # within GraphIndexer, LTM, and STM initializations if not cached.
    # Real Redis and Neo4j connections will be attempted.

    logger.info(f"Using AppSettings - Redis: {settings.redis_config.host}:{settings.redis_config.port}, DB {settings.redis_config.db}")
    logger.info(f"Neo4j: {settings.neo4j_config.uri}")
    logger.info(f"LTM FAISS path: {settings.vector_db_config.params.get('index_path', 'NotSet')}") # LTM uses vector_db_config
    logger.info(f"LTM Raw Content path: {settings.filesystem_config.base_path}") # LTM uses filesystem_config
    logger.info(f"GraphIndexer spaCy model: {settings.nlp_service_config.model}") # GraphIndexer uses nlp_service_config
    logger.info(f"Shared Embedding model: {settings.embedding_config.model_name}") # LTM & GraphIndexer use embedding_config
    logger.info(f"STM Summarizer model: {settings.nlp_service_config.model}") # STM uses nlp_service_config for its models by default path


    # 2. Get MemoryKernel instance
    kernel: Optional[MemoryKernel] = None
    try:
        kernel = MemoryKernel.get_instance(app_settings=settings)
        logger.info("MemoryKernel instance obtained.")

        # 3. Boot the kernel
        logger.info("Booting MemoryKernel...")
        await kernel.boot()
        logger.info("MemoryKernel boot complete.")

        # The following sections demonstrating store_context and get_context are now obsolete
        # as these methods have been removed from MemoryKernel.
        # New examples would involve IngestionProcessor and RetrievalProcessor.
        logger.info("--- MemoryKernel main example: store_context and get_context are removed ---")
        logger.info("--- Business logic now resides in IngestionProcessor and RetrievalProcessor ---")
        logger.info("--- MemoryManager handles direct storage operations ---")

        # Example: Accessing the memory_manager (if needed for direct operations, though typically not)
        if hasattr(kernel, 'memory_manager'):
            logger.info(f"MemoryManager is available: {kernel.memory_manager}")
        else:
            logger.warning("MemoryManager not found on kernel instance.")


        # Allow some time for async operations if any are truly backgrounded (not an issue with current structure)
        await asyncio.sleep(0.1)


    except Exception as e:
        logger.error(f"An error occurred during MemoryKernel main example: {e}", exc_info=True)
        logger.error("Please ensure Redis, Neo4j are running and accessible, and that models "
                     "(spaCy, SentenceTransformer, HuggingFace for STM) are correctly configured and downloadable.")
    finally:
        if kernel:
            # 6. Shutdown the kernel
            logger.info("Shutting down MemoryKernel...")
            await kernel.shutdown()
            logger.info("MemoryKernel shutdown complete.")

    # Singleton verification (already tested in previous main, can be kept or removed for brevity)
    # ...

if __name__ == "__main__":
    # Configure logging for the main test function
    # but for this example script, having it here ensures it's applied when run directly.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Ensure it logs to console

    # Note: For this main() to fully work with actual services,
    # Redis and Neo4j should be running and accessible with credentials from AppSettings.
    # The _initialize_clients method has placeholders for other service clients (NLP, VectorDB etc.)
    # which means LTM, GraphIndexer, STM will use these placeholders.
    # If Redis/Neo4j are not available, their respective clients might fail to connect in MemoryKernel.boot()
    # (or earlier in _initialize_clients if pings were there), and errors will be logged.
    # The example may proceed with components using stubbed/non-functional clients in such cases.
    asyncio.run(main())
