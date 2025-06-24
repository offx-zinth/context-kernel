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

    async def get_context(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves context relevant to the query from LTM, STM, and GraphDB.
        """
        logger.info(f"Received get_context request for query: '{query}', session_id: {session_id}")
        synthesis_log = []
        context_results: Dict[str, Any] = {
            "query": query,
            "retrieved_ltm_items": [],
            "recent_stm_turns": [],
            "related_graph_entities": [],
            "synthesis_log": synthesis_log,
        }

        # 1. Query Embedding
        query_embedding: Optional[List[float]] = None
        try:
            query_embedding = await self.ltm.generate_embedding(query)
            synthesis_log.append(f"Query embedding generated successfully (dim: {len(query_embedding)}).")
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}", exc_info=True)
            synthesis_log.append(f"Error generating query embedding: {e}")
            # Depending on policy, might return early or continue without LTM/some GraphDB features
            # For now, continue if possible.

        # 2. LTM Retrieval
        if query_embedding:
            try:
                ltm_items = await self.ltm.retrieve_relevant_memories(query_embedding=query_embedding, top_k=5)
                context_results["retrieved_ltm_items"] = ltm_items
                synthesis_log.append(f"Retrieved {len(ltm_items)} items from LTM.")
            except Exception as e:
                logger.error(f"Error retrieving from LTM: {e}", exc_info=True)
                synthesis_log.append(f"Error retrieving from LTM: {e}")
        else:
            synthesis_log.append("Skipping LTM retrieval due to missing query embedding.")

        # 3. STM Retrieval
        if session_id:
            try:
                stm_turns = await self.stm.get_recent_turns(session_id=session_id, num_turns=10)
                context_results["recent_stm_turns"] = stm_turns
                synthesis_log.append(f"Retrieved {len(stm_turns)} turns from STM for session '{session_id}'.")
            except Exception as e:
                logger.error(f"Error retrieving from STM for session '{session_id}': {e}", exc_info=True)
                synthesis_log.append(f"Error retrieving from STM: {e}")
        else:
            synthesis_log.append("No session_id provided, skipping STM retrieval.")

        # 4. GraphDB Retrieval (Simplified Entity Linking & Expansion)
        try:
            if self.graph_indexer.nlp_processor: # Check if spaCy model is loaded
                loop = asyncio.get_running_loop()
                doc = await loop.run_in_executor(None, self.graph_indexer.nlp_processor, query)
                extracted_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
                synthesis_log.append(f"Extracted entities from query: {extracted_entities}")

                graph_entities_data = []
                for entity in extracted_entities:
                    # Note: This simple query matches on node_id, but GraphIndexer creates entity nodes with `name`.
                    # Adjusting query to match on `name` property.
                    # Also, GraphDB stores entity type in `entity_type` property.
                    # A more robust approach would involve specific GraphDB methods for entity linking.
                    entity_query = (
                        "MATCH (e {name: $entity_name}) "
                        "OPTIONAL MATCH (e)-[r]-(related) "
                        "RETURN e, collect({relationship: type(r), target_node: properties(related)}) AS relations LIMIT 5"
                    )
                    # entity_query = "MATCH (e {name: $entity_name}) RETURN e LIMIT 1" # Simpler query
                    entity_graph_data = await self.graph_db.cypher_query(entity_query, {"entity_name": entity["text"]})
                    if entity_graph_data:
                        graph_entities_data.extend(entity_graph_data)
                context_results["related_graph_entities"] = graph_entities_data
                synthesis_log.append(f"Retrieved {len(graph_entities_data)} related items/entities from GraphDB based on query entities.")
            else:
                synthesis_log.append("GraphIndexer NLP processor not available, skipping GraphDB entity linking.")
        except Exception as e:
            logger.error(f"Error during GraphDB retrieval: {e}", exc_info=True)
            synthesis_log.append(f"Error during GraphDB retrieval: {e}")

        logger.info(f"Context retrieval complete for query '{query}'. Log: {synthesis_log}")
        return context_results

    async def store_context(self, data: Dict[str, Any], session_id: Optional[str] = None) -> bool:
        """
        Stores data into the appropriate memory components based on its structure.
        - "text_content" & "metadata": Processed by GraphIndexer, then stored in LTM.
        - "ephemeral_data": Stored in RawCache.
        - "turn_data": Added to STM if session_id is provided.
        """
        logger.info(f"Received store_context request. Session_id: {session_id}. Data keys: {list(data.keys())}")
        overall_success = True

        # 1. Ephemeral Data to RawCache
        ephemeral_content = data.get("ephemeral_data")
        if ephemeral_content:
            try:
                ephemeral_key = f"ephemeral_{data.get('chunk_id', str(uuid.uuid4()))}"
                await self.raw_cache.set(key=ephemeral_key, value=ephemeral_content, ttl_seconds=3600)
                logger.info(f"Stored ephemeral data to RawCache with key: {ephemeral_key}")
            except Exception as e:
                logger.error(f"Error storing ephemeral data to RawCache: {e}", exc_info=True)
                overall_success = False

        # 2. Conversational Turn to STM
        turn_content = data.get("turn_data")
        if session_id and turn_content:
            if isinstance(turn_content, dict):
                try:
                    await self.stm.add_turn(session_id=session_id, turn_data=turn_content)
                    logger.info(f"Added turn data to STM for session '{session_id}'.")
                except Exception as e:
                    logger.error(f"Error adding turn data to STM for session '{session_id}': {e}", exc_info=True)
                    overall_success = False
            else:
                logger.warning(f"Skipping STM storage: 'turn_data' is not a dictionary (type: {type(turn_content)}).")
                overall_success = False # Or handle as a non-critical warning

        # 3. Primary Content to GraphIndexer, GraphDB, and LTM
        text_content = data.get("text_content")
        if text_content: # Only proceed if there's primary text content
            chunk_id = data.get("chunk_id", str(uuid.uuid4()))
            metadata = data.get("metadata", {})

            # 3a. Process with GraphIndexer (stores in GraphDB)
            try:
                # GraphIndexer expects chunk_data to be a dict or str.
                # If it's just text_content and metadata, we can pass it as a dict.
                indexer_input_data = {"text_content": text_content, **metadata}
                processing_result = await self.graph_indexer.process_memory_chunk(
                    chunk_id=chunk_id,
                    chunk_data=indexer_input_data
                )
                logger.info(f"GraphIndexer processing result for chunk '{chunk_id}': {processing_result.get('status', 'unknown')}")
                if processing_result.get('status') != 'success':
                    overall_success = False
            except Exception as e:
                logger.error(f"Error processing chunk '{chunk_id}' with GraphIndexer: {e}", exc_info=True)
                overall_success = False

            # 3b. Store in LTM
            # Generate embedding (LTM's generate_embedding uses caching)
            # Then store the original text_content, this new embedding, and original metadata in LTM.
            try:
                ltm_embedding = await self.ltm.generate_embedding(text_content)
                if ltm_embedding: # Ensure embedding was generated
                    ltm_store_id = await self.ltm.store_memory_chunk(
                        chunk_id=chunk_id, # Use the same chunk_id for consistency if desired
                        text_content=text_content,
                        embedding=ltm_embedding,
                        metadata=metadata
                    )
                    logger.info(f"Stored text content for chunk '{chunk_id}' in LTM with ID '{ltm_store_id}'.")
                else:
                    logger.error(f"Failed to generate embedding for LTM storage of chunk '{chunk_id}'.")
                    overall_success = False
            except Exception as e:
                logger.error(f"Error storing chunk '{chunk_id}' content to LTM: {e}", exc_info=True)
                overall_success = False
        elif "ephemeral_data" not in data and "turn_data" not in data:
            # If no text_content, and no other specific data types handled, it's a bit of an empty call.
            logger.warning("store_context called with no 'text_content', 'ephemeral_data', or 'turn_data'. No primary storage action taken.")
            # overall_success could be set to False here if this is considered an invalid call.

        return overall_success

async def main():
    # Example Usage for MemoryKernel
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

        # 4. Example: Store some context items
        logger.info("--- Storing Context Examples ---")
        session_id_example = f"session_{str(uuid.uuid4())}"

        # Item 1: General text content
        await kernel.store_context({
            "chunk_id": "doc1_chunk1",
            "text_content": "The first document is about artificial intelligence and its impact on society. AI is rapidly evolving.",
            "metadata": {"source": "doc1", "type": "general_info", "tags": ["AI", "society"]}
        }, session_id=session_id_example)

        # Item 2: Conversational turn for STM and also as general text content
        turn_data_user = {"role": "user", "content": "What are the benefits of AI in healthcare?"}
        await kernel.store_context({
            "chunk_id": "chat1_turn1",
            "text_content": turn_data_user["content"], # Indexing the content part
            "metadata": {"source": "chat1", "user": "user_A", "tags": ["AI", "healthcare", "question"]},
            "turn_data": turn_data_user # For STM
        }, session_id=session_id_example)

        turn_data_assistant = {"role": "assistant", "content": "AI can improve diagnostics, personalize treatments, and accelerate research in healthcare."}
        await kernel.store_context({
            "chunk_id": "chat1_turn2",
            "text_content": turn_data_assistant["content"], # Indexing the content part
            "metadata": {"source": "chat1", "user": "assistant_B", "tags": ["AI", "healthcare", "answer"]},
            "turn_data": turn_data_assistant # For STM
        }, session_id=session_id_example)

        # Item 3: Ephemeral data
        await kernel.store_context({
            "chunk_id": "ephem1", # Optional, helps key generation
            "ephemeral_data": {"user_preference": "dark_mode", "timestamp": asyncio.get_running_loop().time()}
        })

        # Allow some time for async operations if any are truly backgrounded (not an issue with current structure)
        await asyncio.sleep(0.1)

        # 5. Example: Get context
        logger.info("--- Getting Context Example ---")
        query1 = "AI in society"
        retrieved_context1 = await kernel.get_context(query=query1, session_id=session_id_example)
        logger.info(f"Context for query '{query1}':\n{json.dumps(retrieved_context1, indent=2, default=str)}")
        assert len(retrieved_context1["retrieved_ltm_items"]) > 0, f"Expected LTM items for query: {query1}"

        query2 = "AI in healthcare"
        retrieved_context2 = await kernel.get_context(query=query2, session_id=session_id_example)
        logger.info(f"Context for query '{query2}' (with session_id '{session_id_example}'):\n{json.dumps(retrieved_context2, indent=2, default=str)}")
        assert len(retrieved_context2["recent_stm_turns"]) == 2, "Expected STM turns from current session"
        assert len(retrieved_context2["retrieved_ltm_items"]) > 0, f"Expected LTM items for query: {query2}"

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
