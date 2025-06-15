import asyncio
import logging
import uuid
import json
import hashlib # For embedding cache keys
from typing import List, Dict, Optional, Any, Union # For type hinting

from contextkernel.utils.config import (
    VectorDBConfig,
    EmbeddingConfig,
    S3Config,
    FileSystemConfig,
    RedisConfig # For RawCache's own config if needed by main example
)
from contextkernel.memory_system.raw_cache import RawCache
# Placeholder for actual client types - in a real scenario, these would be imported from SDKs
# e.g., from pinecone import PineconeClient
# from openai import OpenAI # or from sentence_transformers import SentenceTransformer
# from boto3 import client as S3Client

logger = logging.getLogger(__name__)

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Define placeholder types for clients for now if specific client SDKs are not yet integrated
# These would ideally be protocols or ABCs if we define an interface
# VectorDBClientType = Any # Will be replaced by FAISS index
# RawContentStoreClientType = Any # Will be replaced by FileSystem logic
# EmbeddingModelClientType = Any # Will be replaced by SentenceTransformer model


class LTM:
    """
    Long-Term Memory (LTM) system.
    Stores persistent knowledge using SentenceTransformer for embeddings, FAISS for vector search,
    and a FileSystem-based raw content store. Uses RawCache for embedding caching.
    """

    def __init__(self,
                 vector_db_config: VectorDBConfig, # Params for FAISS e.g. {"index_path": "faiss.index", "dimension": 384}
                 raw_content_store_config: Union[S3Config, FileSystemConfig], # Expecting FileSystemConfig
                 embedding_config: EmbeddingConfig,
                 embedding_cache: RawCache):

        logger.info("Initializing LongTermMemory (LTM) system with FAISS, SentenceTransformer, and FileSystem store.")

        self.vector_db_config = vector_db_config
        self.raw_content_store_config = raw_content_store_config
        self.embedding_config = embedding_config
        self.embedding_cache = embedding_cache

        self.embedding_model: Optional[SentenceTransformer] = None
        self.embedding_dimension: Optional[int] = None # Set in boot()
        self.faiss_index: Optional[faiss.Index] = None # Initialized in boot()

        # Mappings for FAISS index & metadata storage
        self.faiss_id_to_memory_id: Dict[int, str] = {}
        self.memory_id_to_faiss_id: Dict[str, int] = {}
        self.memory_id_metadata: Dict[str, Dict[str, Any]] = {}

        # Configuration shortcuts
        self._faiss_index_path: Optional[str] = self.vector_db_config.params.get("index_path") if self.vector_db_config.params else None
        self._raw_store_base_path: Optional[str] = None # Set in boot() if FileSystemConfig

        # Ensure vector_db_config.params exists if we expect "dimension" later, or handle its absence.
        if not self.vector_db_config.params:
            self.vector_db_config.params = {}


        if isinstance(self.raw_content_store_config, FileSystemConfig):
            # Path will be set and created in boot()
            pass # self._raw_store_base_path will be set in boot
        elif self.raw_content_store_config is not None: # If some other config type
             logger.warning(f"LTM Raw content store is type '{type(self.raw_content_store_config).__name__}', "
                            "but current implementation primarily supports FileSystemConfig. Operations might be limited.")
        else: # If None
            logger.error("LTM raw_content_store_config is not provided. Raw content storage will not function.")
            # raise ValueError("LTM requires a raw_content_store_config.")


        logger.info(f"LTM initialized. Embedding model to load: {self.embedding_config.model_name}, "
                    f"FAISS index target path: {self._faiss_index_path}, "
                    f"RawStore config type: {type(self.raw_content_store_config).__name__}, "
                    f"Cache: {type(self.embedding_cache.client).__name__ if hasattr(self.embedding_cache, 'client') else 'RawCache'}")

    async def boot(self):
        """
        Loads the SentenceTransformer model, initializes or loads the FAISS index,
        and ensures the raw content store directory exists if FileSystemConfig is used.
        """
        logger.info("LTM booting up...")
        try:
            # 1. Load SentenceTransformer model
            model_name = self.embedding_config.model_name or 'all-MiniLM-L6-v2' # Fallback model
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer model '{model_name}' loaded. Dimension: {self.embedding_dimension}.")

            # 2. Validate FAISS dimension from config if provided, otherwise use model's
            config_dimension = self.vector_db_config.params.get("dimension")
            if config_dimension and config_dimension != self.embedding_dimension:
                logger.error(f"Configured FAISS dimension ({config_dimension}) does not match model dimension ({self.embedding_dimension}).")
                raise ValueError("FAISS dimension mismatch.")
            elif not config_dimension:
                logger.info(f"FAISS dimension not in config, using model dimension: {self.embedding_dimension}")
                self.vector_db_config.params["dimension"] = self.embedding_dimension # Store for clarity or if new index is created

            # 3. Load or create FAISS index
            if self._faiss_index_path and os.path.exists(self._faiss_index_path):
                logger.info(f"Loading FAISS index from {self._faiss_index_path}...")
                self.faiss_index = faiss.read_index(self._faiss_index_path)
                logger.info(f"FAISS index loaded. Index has {self.faiss_index.ntotal} vectors of dimension {self.faiss_index.d}.")
                if self.faiss_index.d != self.embedding_dimension:
                    logger.error(f"Loaded FAISS index dimension ({self.faiss_index.d}) does not match model dimension ({self.embedding_dimension}).")
                    raise ValueError("Loaded FAISS index dimension mismatch.")
                # TODO: Load mappings and metadata persistence here.
            else:
                logger.info(f"FAISS index not found at {self._faiss_index_path} or path not configured. Creating new index.")
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
                logger.info(f"New FAISS index (IndexFlatL2) created with dimension {self.embedding_dimension}.")

            # 4. Setup Raw Content Store (FileSystem specific)
            if isinstance(self.raw_content_store_config, FileSystemConfig):
                self._raw_store_base_path = self.raw_content_store_config.base_path
                if not self._raw_store_base_path:
                    logger.error("FileSystemConfig selected but base_path is not set.")
                    raise ValueError("FileSystemConfig.base_path is required.")
                os.makedirs(self._raw_store_base_path, exist_ok=True)
                logger.info(f"Raw content store directory (FileSystem) '{self._raw_store_base_path}' ensured.")
            elif self.raw_content_store_config is None:
                 logger.warning("No raw_content_store_config provided. Raw content operations will fail.")
            else: # S3Config or other
                logger.info(f"Raw content store is type '{type(self.raw_content_store_config).__name__}'. "
                            "Ensure client for it is provided and handled by specific methods if not FileSystem.")


            # 5. Boot internal cache
            if not await self.embedding_cache.boot():
                 logger.warning("LTM: Embedding cache (RawCache) failed to boot or connect properly.")

            logger.info("LTM boot complete.")
            return True
        except Exception as e:
            logger.error(f"Error during LTM boot: {e}", exc_info=True)
            self.embedding_model = None # Ensure partial states are cleared
            self.faiss_index = None
            return False

    async def shutdown(self):
        """
        Saves the FAISS index if a path is configured and index exists.
        """
        logger.info("LTM shutting down...")
        if self.faiss_index is not None and self._faiss_index_path:
            try:
                if self.faiss_index.ntotal > 0: # Only save if there's something to save
                    logger.info(f"Saving FAISS index with {self.faiss_index.ntotal} vectors to {self._faiss_index_path}...")
                    faiss.write_index(self.faiss_index, self._faiss_index_path)
                    logger.info("FAISS index saved.")
                    # TODO: Persist self.faiss_id_to_memory_id and self.memory_id_metadata
                else:
                    logger.info("FAISS index is empty. Not saving to disk.")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self._faiss_index_path}: {e}", exc_info=True)

        self.embedding_model = None # Release model reference
        logger.info("LTM shutdown complete.")
        return True

    def _get_text_key(self, text_content: str) -> str:
        """Generates a cache key for text content."""
        return hashlib.md5(text_content.encode('utf-8')).hexdigest()

    async def _get_embedding_from_cache(self, text_key: str) -> Optional[List[float]]:
        logger.debug(f"Checking embedding cache for key: {text_key}")
        # Using the injected RawCache instance
        cached_value = await self.embedding_cache.get(key=text_key, namespace="ltm_embeddings")
        if cached_value:
            # Assuming cached_value is stored as JSON list or similar if RawCache serializes
            if isinstance(cached_value, list): # Or try json.loads if it's a string
                return cached_value
            try: # Try to load if it's a JSON string from cache
                loaded_list = json.loads(cached_value)
                if isinstance(loaded_list, list):
                    return loaded_list
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to decode cached embedding for key {text_key}. Value: {cached_value}")
        return None

    async def _set_embedding_to_cache(self, text_key: str, embedding: List[float]):
        logger.debug(f"Storing embedding in cache for key: {text_key}")
        # Using the injected RawCache instance. RawCache handles TTL via its default or set params.
        # RawCache now expects complex types to be serialized before set if necessary, or handles it.
        # For simplicity, assuming RawCache can handle list[float] or it's serialized to JSON string.
        await self.embedding_cache.set(key=text_key, value=embedding, namespace="ltm_embeddings")

    async def generate_embedding(self, text_content: str) -> List[float]:
        """
        Generates or retrieves a cached embedding for the given text content
        using the loaded SentenceTransformer model and RawCache.
        """
        if not self.embedding_model or not self.embedding_dimension:
            logger.error("Embedding model or dimension not initialized. Call boot() first.")
            # Fallback to a zero vector of a common dimension or raise error
            return [0.0] * (self.embedding_config.params.get("dimension",384) if self.embedding_config.params else 384)


        if not text_content:
            logger.warning("Empty text_content for embedding generation. Returning default zero vector.")
            return [0.0] * self.embedding_dimension

        text_key = f"embedding:{self._get_text_key(text_content)}" # Add prefix for clarity

        try:
            cached_embedding = await self._get_embedding_from_cache(text_key)
            if cached_embedding:
                logger.debug(f"Using cached embedding for text key: {text_key}")
                return cached_embedding
        except Exception as e:
            logger.warning(f"Error retrieving embedding from cache for key {text_key}: {e}. Will regenerate.")

        try:
            logger.debug(f"Generating new embedding for text: '{text_content[:50]}...'")
            # SentenceTransformer.encode can run in a thread pool if loop is None,
            # but for async, it's better to run it in an executor to not block the event loop.
            # For simplicity here, directly calling it. If performance becomes an issue, use asyncio.to_thread (Python 3.9+)
            # or loop.run_in_executor.
            loop = asyncio.get_running_loop()
            embedding_array = await loop.run_in_executor(
                None, # Uses default ThreadPoolExecutor
                self.embedding_model.encode,
                text_content
            )
            embedding = embedding_array.tolist()

            # Ensure embedding is a list of floats (it should be from tolist())
            if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                 #This case should ideally not happen with .tolist()
                 logger.warning(f"Generated embedding is not a list of floats. Type: {type(embedding)}. Attempting conversion.")
                 embedding = [float(x) for x in embedding]


        except Exception as e:
            logger.error(f"Error generating embedding for text '{text_content[:50]}...': {e}", exc_info=True)
            return [0.0] * self.embedding_dimension # Return zero vector on error

        try:
            await self._set_embedding_to_cache(text_key, embedding)
        except Exception as e:
            logger.warning(f"Error storing embedding to cache for key {text_key}: {e}.")

        return embedding

    async def store_memory_chunk(self, chunk_id: Optional[str], text_content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        if not self.faiss_index or not self._raw_store_base_path or not self.embedding_dimension:
            logger.error("LTM not properly booted (FAISS index, raw store path, or embedding dimension missing). Cannot store memory.")
            raise RuntimeError("LTM is not properly initialized. Call boot() first.")

        memory_id = f"ltm_mem_{chunk_id}" if chunk_id else str(uuid.uuid4())
        raw_content_id = f"raw_{memory_id}.json" # Store as JSON file

        logger.info(f"Storing memory chunk: ID='{memory_id}', RawID='{raw_content_id}'")

        # 1. Store Raw Content (FileSystem)
        raw_file_path = os.path.join(self._raw_store_base_path, raw_content_id)
        try:
            # Enrich metadata for raw store slightly
            raw_store_data = {
                "text_content": text_content,
                "metadata": metadata, # Original metadata
                "ltm_memory_id": memory_id, # Link back to the LTM memory ID
                "embedding_preview": embedding[:5] # Store a preview for inspection
            }
            with open(raw_file_path, 'w', encoding='utf-8') as f:
                json.dump(raw_store_data, f, indent=4)
            logger.debug(f"Stored raw content for {memory_id} at {raw_file_path}")
        except IOError as e:
            logger.error(f"Error storing raw content to {raw_file_path}: {e}", exc_info=True)
            # Decide if we should proceed without raw content or raise error
            raise # Re-raise for now, as vector DB entry without raw content might be problematic

        # 2. Vector DB Storage (FAISS)
        try:
            vector_db_metadata = metadata.copy()
            vector_db_metadata["raw_content_id"] = raw_content_id # Link to raw content file
            vector_db_metadata["original_memory_id"] = memory_id # For potential cross-referencing

            if memory_id in self.memory_id_to_faiss_id: # Update existing memory
                old_faiss_id = self.memory_id_to_faiss_id[memory_id]
                logger.info(f"Updating memory_id '{memory_id}' at FAISS index ID {old_faiss_id}.")
                # FAISS remove_ids expects an Int64Vector.
                ids_to_remove = np.array([old_faiss_id], dtype=np.int64)
                self.faiss_index.remove_ids(ids_to_remove)
                # Note: remove_ids compacts the index. New IDs might not be sequential in a simple way
                # after many removals. For IndexFlat, IDs are shifted.
                # This simple mapping will break if not careful.
                # A more robust approach uses IndexIDMap or rebuilds maps.
                # For this task's scope, we assume adding is more common or rebuild maps offline.
                # Let's re-add and update map; this is okay for IndexFlat if we don't rely on stable FAISS IDs after removal.
                # The new ID will be ntotal.

            embedding_np = np.array([embedding], dtype=np.float32)
            if embedding_np.shape[1] != self.embedding_dimension:
                logger.error(f"Embedding dimension mismatch for {memory_id}. Expected {self.embedding_dimension}, got {embedding_np.shape[1]}.")
                # Potentially try to recover raw content file deletion if critical, or mark as failed.
                os.remove(raw_file_path) # Clean up orphaned raw content file
                raise ValueError(f"Embedding dimension mismatch for {memory_id}.")

            self.faiss_index.add(embedding_np)
            new_faiss_id = self.faiss_index.ntotal - 1

            # Update mappings
            if memory_id in self.memory_id_to_faiss_id: # If it was an update
                old_faiss_id_for_map_cleanup = self.memory_id_to_faiss_id[memory_id]
                if old_faiss_id_for_map_cleanup in self.faiss_id_to_memory_id:
                    del self.faiss_id_to_memory_id[old_faiss_id_for_map_cleanup]

            self.faiss_id_to_memory_id[new_faiss_id] = memory_id
            self.memory_id_to_faiss_id[memory_id] = new_faiss_id
            self.memory_id_metadata[memory_id] = vector_db_metadata # Store metadata

            logger.debug(f"Stored embedding for {memory_id} at FAISS ID {new_faiss_id}. Metadata: {vector_db_metadata}")
        except Exception as e:
            logger.error(f"Error storing embedding to FAISS for {memory_id}: {e}", exc_info=True)
            # Clean up raw content file if vector storage failed
            if os.path.exists(raw_file_path):
                try:
                    os.remove(raw_file_path)
                    logger.info(f"Cleaned up raw content file {raw_file_path} due to FAISS error.")
                except OSError as ose:
                    logger.error(f"Error cleaning up raw content file {raw_file_path}: {ose}")
            raise # Re-raise

        logger.info(f"Memory chunk '{memory_id}' stored successfully.")
        return memory_id

    async def retrieve_relevant_memories(self, query_embedding: List[float], top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.faiss_index or not self._raw_store_base_path:
            logger.error("LTM not properly booted. Cannot retrieve memories.")
            return []
        if self.faiss_index.ntotal == 0:
            logger.info("FAISS index is empty. No memories to retrieve.")
            return []

        query_embedding_np = np.array([query_embedding], dtype=np.float32)

        # If filtering, fetch more results from FAISS to filter down.
        # This is a simple strategy; more advanced ones might involve pre-filtering or iterative fetching.
        k_to_fetch = top_k * 5 if metadata_filter else top_k
        k_to_fetch = min(k_to_fetch, self.faiss_index.ntotal) # Cannot fetch more than available

        logger.info(f"Retrieving up to {k_to_fetch} relevant memories from FAISS for query (embedding dim: {query_embedding_np.shape[1]}). Applying filter: {metadata_filter}")

        try:
            distances, faiss_ids = self.faiss_index.search(query_embedding_np, k_to_fetch)
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}", exc_info=True)
            return []

        results: List[Dict[str, Any]] = []
        if faiss_ids.size == 0 or faiss_ids[0][0] == -1: # faiss_ids[0][0] == -1 means no result for that query vector
            logger.info("No results from FAISS search for the given query embedding.")
            return results

        for i in range(faiss_ids.shape[1]): # Iterate through neighbors for the first (and only) query vector
            faiss_id = faiss_ids[0][i]
            if faiss_id == -1: continue # Should not happen if k_to_fetch <= ntotal and ntotal > 0

            memory_id = self.faiss_id_to_memory_id.get(int(faiss_id))
            if not memory_id:
                logger.warning(f"FAISS ID {faiss_id} not found in mapping. Skipping.")
                continue

            item_metadata_for_vector_db = self.memory_id_metadata.get(memory_id)
            if not item_metadata_for_vector_db:
                logger.warning(f"Metadata for memory_id '{memory_id}' (FAISS ID {faiss_id}) not found. Skipping.")
                continue

            # Apply metadata filter
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    meta_val = item_metadata_for_vector_db.get(key)
                    if isinstance(meta_val, list) and value not in meta_val:
                        match = False; break
                    elif not isinstance(meta_val, list) and meta_val != value:
                        match = False; break
                if not match:
                    logger.debug(f"Memory ID '{memory_id}' filtered out by metadata: {metadata_filter}")
                    continue

            # Retrieve raw content
            raw_content_id = item_metadata_for_vector_db.get("raw_content_id")
            if not raw_content_id:
                logger.warning(f"Raw content ID missing for memory_id '{memory_id}'. Skipping.")
                continue

            raw_file_path = os.path.join(self._raw_store_base_path, raw_content_id)
            try:
                with open(raw_file_path, 'r', encoding='utf-8') as f:
                    raw_content_data = json.load(f)

                results.append({
                    "memory_id": memory_id,
                    "text_content": raw_content_data["text_content"],
                    "metadata": raw_content_data["metadata"], # Return the original, richer metadata
                    "score": float(distances[0][i]), # FAISS L2 distance; smaller is better
                    "retrieval_source": "LTM_FAISS_FileSystem"
                })
                if len(results) >= top_k:
                    break # Reached desired number of filtered results
            except FileNotFoundError:
                logger.error(f"Raw content file not found: {raw_file_path} for memory_id '{memory_id}'. Data inconsistency.")
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Error reading/parsing raw content file {raw_file_path}: {e}")

        logger.info(f"Retrieved {len(results)} relevant memories after filtering.")
        return results

    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Attempting to retrieve memory by ID: {memory_id}")
        if not self._raw_store_base_path:
             logger.error("Raw store base path not configured. Cannot get memory by ID.")
             return None

        item_metadata_for_vector_db = self.memory_id_metadata.get(memory_id)
        if not item_metadata_for_vector_db:
            logger.warning(f"Memory ID '{memory_id}' not found in LTM's metadata store.")
            return None

        raw_content_id = item_metadata_for_vector_db.get("raw_content_id")
        if not raw_content_id:
            logger.error(f"CRITICAL: Raw content ID missing in metadata for memory_id '{memory_id}'.")
            return None

        raw_file_path = os.path.join(self._raw_store_base_path, raw_content_id)
        try:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                raw_content_data = json.load(f)

            # Optionally, retrieve FAISS ID and embedding if needed for the return value
            faiss_id = self.memory_id_to_faiss_id.get(memory_id)
            # Retrieving vector from FAISS by ID: self.faiss_index.reconstruct(faiss_id) if faiss_id is not None else None
            # For now, just return content and metadata.

            return {
                "memory_id": memory_id,
                "text_content": raw_content_data["text_content"],
                "metadata": raw_content_data["metadata"], # Original metadata
                "faiss_id_debug": faiss_id, # For debugging
                "vector_db_metadata_debug": item_metadata_for_vector_db # For debugging
            }
        except FileNotFoundError:
            logger.error(f"Raw content file not found: {raw_file_path} for memory_id '{memory_id}'. Data inconsistency.")
            return None
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading/parsing raw content file {raw_file_path}: {e}")
            return None

    async def schedule_archiving(self):
        # This would interact with the raw_content_store_client and vector_db_client
        # to move data to colder storage or apply archival policies.
        logger.info("LTM Archiving process scheduled (stubbed - no actual client interaction).")
        await asyncio.sleep(0.01)

    async def schedule_cleaning(self):
        # This would interact with vector_db_client to remove stale/irrelevant entries,
        # and potentially self.raw_content_store_client to delete orphaned raw data.
        logger.info("LTM Cleaning process scheduled (stubbed - no actual client interaction).")
        await asyncio.sleep(0.01)


async def main():
import tempfile # For creating temporary directories for the example
import shutil # For cleaning up temporary directories

async def main():
    if not logger.handlers:
        # Set logging level to DEBUG for more detailed output from LTM and other components
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- LTM Real Integration Example Usage ---")

    # Create a temporary directory for this example
    temp_dir = tempfile.mkdtemp(prefix="ltm_example_")
    faiss_index_file = os.path.join(temp_dir, "ltm_faiss.index")
    raw_content_path = os.path.join(temp_dir, "raw_content")

    logger.info(f"Temporary directory for example: {temp_dir}")
    logger.info(f"FAISS index will be at: {faiss_index_file}")
    logger.info(f"Raw content will be stored under: {raw_content_path}")

    # 1. Create Configurations
    # For FAISS, params should include 'index_path'. 'dimension' will be set by boot based on model if not present.
    vec_db_conf = VectorDBConfig(type="faiss", params={"index_path": faiss_index_file})
    raw_store_conf = FileSystemConfig(base_path=raw_content_path)
    embed_conf = EmbeddingConfig(model_name='all-MiniLM-L6-v2') # This model has 384 dimensions

    redis_conf_for_cache = RedisConfig(db=1)

    # Real RawCache for embedding caching
    redis_client_for_cache = None
    try:
        redis_client_for_cache = RedisClient(
            host=redis_conf_for_cache.host,
            port=redis_conf_for_cache.port,
            db=redis_conf_for_cache.db,
            password=redis_conf_for_cache.password,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        await redis_client_for_cache.ping()
        logger.info("Successfully connected to Redis for LTM's embedding cache.")
    except Exception as e:
        logger.error(f"Could not connect to Redis for LTM's cache: {e}. This example needs Redis to run.", exc_info=True)
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
        return

    embedding_cache_instance = RawCache(config=redis_conf_for_cache, client=redis_client_for_cache)

    # Instantiate LTM with real integrations
    ltm_system = LTM(
        vector_db_config=vec_db_conf,
        raw_content_store_config=raw_store_conf,
        embedding_config=embed_conf,
        embedding_cache=embedding_cache_instance
    )

    if not await ltm_system.boot(): # Loads models, FAISS index, creates dirs
        logger.error("LTM boot failed. Exiting example.")
        await redis_client_for_cache.close()
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
        return

    try:
        # --- Test LTM operations ---
        text1 = "The quick brown fox jumps over the lazy dog."
        meta1 = {"source": "test_doc_1", "chapter": 1, "tags": ["animal", "classic"]}

        logger.info(f"Generating embedding for: '{text1}'")
        emb1 = await ltm_system.generate_embedding(text1)
        assert len(emb1) == ltm_system.embedding_dimension, f"Embedding dimension mismatch. Expected {ltm_system.embedding_dimension}, got {len(emb1)}"
        logger.info(f"Generated embedding preview: {emb1[:5]}...")

        # Check if embedding was cached
        text1_cache_key = f"embedding:{ltm_system._get_text_key(text1)}"
        cached_emb1_val = await embedding_cache_instance.get(text1_cache_key, namespace="ltm_embeddings")
        assert cached_emb1_val is not None and isinstance(cached_emb1_val, list), "Embedding for text1 not found in cache or wrong type"
        logger.info("Embedding for text1 found in RawCache after generation.")

        text2 = "Artificial intelligence is transforming various global sectors."
        meta2 = {"source": "research_paper_001", "year": 2023, "tags": ["AI", "technology", "global"]}
        emb2 = await ltm_system.generate_embedding(text2)

        text3 = "Context Kernels provide a novel framework for AI memory management."
        meta3 = {"source": "blog_post_abc", "author": "Dr. AI", "tags": ["AI", "memory", "ContextKernel"]}
        emb3 = await ltm_system.generate_embedding(text3)

        logger.info("--- Storing memory chunks ---")
        mem_id1 = await ltm_system.store_memory_chunk(chunk_id="fox_example", text_content=text1, embedding=emb1, metadata=meta1)
        mem_id2 = await ltm_system.store_memory_chunk(chunk_id="ai_transform_example", text_content=text2, embedding=emb2, metadata=meta2)
        mem_id3 = await ltm_system.store_memory_chunk(chunk_id="ck_def_example", text_content=text3, embedding=emb3, metadata=meta3)
        logger.info(f"Stored memory IDs: {mem_id1}, {mem_id2}, {mem_id3}")
        assert ltm_system.faiss_index.ntotal == 3, f"FAISS index should have 3 vectors, has {ltm_system.faiss_index.ntotal}"

        logger.info(f"--- Retrieving memory by ID: {mem_id1} ---")
        retrieved_mem1 = await ltm_system.get_memory_by_id(mem_id1)
        logger.info(f"Retrieved memory {mem_id1}: {json.dumps(retrieved_mem1, indent=2)}")
        assert retrieved_mem1 is not None and retrieved_mem1["text_content"] == text1, "Failed to retrieve or content mismatch for mem_id1"

        logger.info(f"--- Retrieving relevant memories for query: 'AI memory framework' ---")
        query_text = "AI memory framework"
        query_emb = await ltm_system.generate_embedding(query_text)

        relevant_memories = await ltm_system.retrieve_relevant_memories(query_embedding=query_emb, top_k=2)
        logger.info(f"Relevant memories for '{query_text}':")
        for mem in relevant_memories:
            logger.info(f"  ID: {mem['memory_id']}, Score: {mem['score']:.4f}, Text='{mem['text_content'][:60]}...'")
        assert len(relevant_memories) > 0, "Expected some relevant memories for 'AI memory framework'"
        if relevant_memories:
             assert relevant_memories[0]['memory_id'] == mem_id3, "Context Kernel definition not found as most relevant for 'AI memory framework'"

        logger.info(f"--- Retrieving relevant memories with metadata filter (tags='AI') ---")
        relevant_memories_filtered = await ltm_system.retrieve_relevant_memories(
            query_embedding=query_emb, top_k=2, metadata_filter={"tags": "AI"} # Filter for "AI" tag
        )
        logger.info(f"Filtered relevant memories (tags='AI') for '{query_text}':")
        found_mem_id2 = False
        found_mem_id3 = False
        for mem in relevant_memories_filtered:
            logger.info(f"  ID: {mem['memory_id']}, Score: {mem['score']:.4f}, Text='{mem['text_content'][:60]}...', Metadata: {mem['metadata']}")
            assert "AI" in mem['metadata']['tags'], "Filtered memory does not contain 'AI' tag"
            if mem['memory_id'] == mem_id2: found_mem_id2 = True
            if mem['memory_id'] == mem_id3: found_mem_id3 = True
        assert found_mem_id2 and found_mem_id3, "Expected to find mem_id2 and mem_id3 with 'AI' tag"
        assert len(relevant_memories_filtered) == 2, "Expected 2 memories with 'AI' tag"

        # Test update: Store same memory_id again, should update
        logger.info(f"--- Testing update for memory_id: {mem_id1} ---")
        updated_text1 = "The very quick brown fox jumps swiftly over the dog."
        updated_emb1 = await ltm_system.generate_embedding(updated_text1) # New embedding for new text
        updated_meta1 = {**meta1, "version": 2, "status": "updated"}
        await ltm_system.store_memory_chunk(chunk_id="fox_example", text_content=updated_text1, embedding=updated_emb1, metadata=updated_meta1)

        assert ltm_system.faiss_index.ntotal == 3, "FAISS index should still have 3 vectors after update."
        retrieved_updated_mem1 = await ltm_system.get_memory_by_id(mem_id1)
        assert retrieved_updated_mem1["text_content"] == updated_text1, "Text content not updated."
        assert retrieved_updated_mem1["metadata"]["version"] == 2, "Metadata not updated."
        logger.info(f"Memory {mem_id1} updated and verified.")

        # Test FAISS persistence by shutting down and booting again
        logger.info("--- Testing FAISS persistence ---")
        await ltm_system.shutdown() # Saves index

        # New LTM instance, should load from the saved index
        ltm_system_rebooted = LTM(
            vector_db_config=vec_db_conf,
            raw_content_store_config=raw_store_conf,
            embedding_config=embed_conf,
            embedding_cache=embedding_cache_instance
        )
        assert await ltm_system_rebooted.boot(), "LTM failed to reboot"
        assert ltm_system_rebooted.faiss_index is not None, "FAISS index not loaded on reboot"
        assert ltm_system_rebooted.faiss_index.ntotal == 3, f"Rebooted FAISS index should have 3 vectors, found {ltm_system_rebooted.faiss_index.ntotal}"
        logger.info("FAISS index loaded after reboot with correct number of vectors.")

        # Perform a search with the rebooted LTM to ensure it's functional
        relevant_rebooted = await ltm_system_rebooted.retrieve_relevant_memories(query_embedding=query_emb, top_k=1)
        assert len(relevant_rebooted) == 1, "Search after reboot failed to return results."
        assert relevant_rebooted[0]['memory_id'] == mem_id3, "Search after reboot returned incorrect top result."
        logger.info("Search after reboot successful.")
        # Actual shutdown for the rebooted instance
        await ltm_system_rebooted.shutdown()


    except Exception as e:
        logger.error(f"An error occurred during LTM example operations: {e}", exc_info=True)
    finally:
        # Shutdown LTM (saves FAISS index) and other components
        logger.info("--- Shutting down LTM and other components (final) ---")
        # Ensure original ltm_system is shutdown if it wasn't the one rebooted and potentially failed
        if 'ltm_system_rebooted' not in locals() or ltm_system_rebooted is not ltm_system :
             if ltm_system and ltm_system.faiss_index is not None : #Check if it was booted
                await ltm_system.shutdown()

        if embedding_cache_instance: # Shutdown cache if it was initialized
            await embedding_cache_instance.shutdown()
        if redis_client_for_cache: # Close redis client if it was initialized
            await redis_client_for_cache.close()

        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")

    logger.info("--- LTM Real Integration Example Usage Complete ---")

if __name__ == "__main__":
    # This example needs `sentence-transformers`, `faiss-cpu`, and `redis` to be installed.
    # It also requires a running Redis instance for the RawCache.
    # `pip install sentence-transformers faiss-cpu redis`
    asyncio.run(main())
