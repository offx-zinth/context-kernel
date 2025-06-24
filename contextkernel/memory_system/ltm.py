"""
Module for the Long-Term Memory (LTM) system of the Context Kernel.

The LTM system orchestrates the storage and retrieval of memories over extended periods.
It leverages a pluggable VectorDB interface for managing embeddings and similarity search,
and a RawContentStore interface for handling the original raw content associated with memories.

Key functionalities include:
- Generating embeddings for text content using SentenceTransformer models.
- Caching generated embeddings via a RawCache instance.
- Storing raw content and its metadata through a RawContentStore.
- Storing embeddings and their metadata (linking to raw content) via a VectorDB.
- Retrieving relevant memories based on semantic similarity to query embeddings.
- Fetching specific memories by their unique identifiers.
"""
import asyncio
import logging
import uuid
import json
import hashlib # For embedding cache keys
import abc # For Abstract Base Classes
from typing import List, Dict, Optional, Any, Union # For type hinting

from contextkernel.utils.config import (
    VectorDBConfig,
    EmbeddingConfig,
    # S3Config, # Not directly used by LTM or its current components.
    FileSystemConfig,
    RedisConfig # Kept for RawCache's own config if needed by main example, not LTM directly.
)
from contextkernel.memory_system.raw_cache import RawCache

# Placeholder for actual client types - these would be imported from SDKs in a full implementation
# e.g., from pinecone import PineconeClient
# from openai import OpenAI
# from boto3 import client as S3Client

logger = logging.getLogger(__name__)
# Specific imports for default implementations are within those class blocks if possible,
# but FAISS and SentenceTransformer are core to the default LTM setup shown in main.
import os
import numpy as np
# Conditional import for FAISS and SentenceTransformer for FAISSVectorDB
try:
    import faiss
except ImportError:
    faiss = None # Handled in FAISSVectorDB if not available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None # Handled in FAISSVectorDB if not available

# EmbeddingModelClientType = Any # Placeholder for a generic embedding model client type

# Abstract Base Classes (ABCs) for Storage Backends

class VectorDB(abc.ABC):
    """
    Abstract Base Class defining the interface for a Vector Database.
    Implementations of this class are responsible for storing, managing,
    and searching vector embeddings.
    """

    @abc.abstractmethod
    def __init__(self, config: VectorDBConfig, embedding_config: EmbeddingConfig):
        """
        Initializes the vector database with necessary configurations.

        Args:
            config: Configuration specific to the vector database implementation
                    (e.g., paths, connection details).
            embedding_config: Configuration related to embeddings, primarily for
                              determining embedding dimensions if not explicitly provided by LTM.
        """
        pass

    @abc.abstractmethod
    async def boot(self) -> bool:
        """
        Boots up the vector database. This can include loading an index from disk,
        connecting to a remote service, or any other initialization steps.

        Returns:
            True if boot-up was successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def shutdown(self) -> bool:
        """
        Shuts down the vector database. This can include saving an index to disk,
        closing connections, or releasing resources.

        Returns:
            True if shutdown was successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def store_embedding(self, memory_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """
        Stores a single vector embedding and its associated metadata.

        Args:
            memory_id: A unique identifier for the memory chunk this embedding represents.
            embedding: The vector embedding as a list of floats.
            metadata: A dictionary of metadata associated with the embedding.
                      Should typically include a link to the raw content (e.g., raw_content_id).
        """
        pass

    @abc.abstractmethod
    async def store_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> None:
        """
        Stores a batch of embeddings efficiently.

        Args:
            embeddings_data: A list of dictionaries, where each dictionary contains:
                             - "memory_id": str (unique identifier)
                             - "embedding": List[float] (the vector embedding)
                             - "metadata": Dict (associated metadata)
        """
        pass

    @abc.abstractmethod
    async def retrieve_similar_embeddings(self, query_embedding: List[float], top_k: int,
                                          metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves embeddings that are semantically similar to a given query embedding.

        Args:
            query_embedding: The vector embedding to search against.
            top_k: The number of similar embeddings to retrieve.
            metadata_filter: (Optional) A dictionary to filter results based on metadata fields.
                             Filtering capabilities depend on the specific VectorDB implementation.

        Returns:
            A list of dictionaries, each representing a similar item, containing:
            - "memory_id": str (identifier of the retrieved memory)
            - "score": float (similarity score, interpretation depends on implementation, e.g., distance)
            - "metadata": Dict (metadata associated with the retrieved embedding)
        """
        pass

    @abc.abstractmethod
    async def get_embedding_by_id(self, memory_id: str) -> Optional[List[float]]:
        """
        Retrieves a specific embedding by its memory_id.
        Note: Some vector databases might not support direct reconstruction of vectors
        from their ID efficiently or at all.

        Args:
            memory_id: The unique identifier of the memory whose embedding is to be retrieved.

        Returns:
            The embedding as a list of floats if found, otherwise None.
        """
        pass

    @abc.abstractmethod
    async def get_metadata_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata associated with a specific memory_id.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The metadata dictionary if found, otherwise None.
        """
        pass

    @abc.abstractmethod
    async def delete_embedding(self, memory_id: str) -> None:
        """
        Deletes an embedding (and its associated metadata) by its memory_id.

        Args:
            memory_id: The unique identifier of the memory to delete.
        """
        pass

    @abc.abstractmethod
    async def delete_embeddings(self, memory_ids: List[str]) -> None:
        """
        Deletes multiple embeddings by their memory_ids.

        Args:
            memory_ids: A list of unique identifiers for the memories to delete.
        """
        pass

    @abc.abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """
        Gets the current status and statistics of the vector database.

        Returns:
            A dictionary containing status information, e.g.,
            {"total_vectors": int, "dimension": int, "index_type": str}.
            Specific content may vary by implementation.
        """
        pass


class RawContentStore(abc.ABC):
    """
    Abstract Base Class defining the interface for a Raw Content Store.
    Implementations of this class are responsible for storing and retrieving
    the original, unprocessed content (e.g., text, images, structured data)
    that LTM memories are based on.
    """

    @abc.abstractmethod
    def __init__(self, config: Union[FileSystemConfig, Any]): # Using Any for now as S3Config/RedisConfig are not concrete for this store yet
        """
        Initializes the raw content store with necessary configurations.

        Args:
            config: Configuration specific to the raw content store implementation
                    (e.g., base path for FileSystem, connection details for S3/Redis).
        """
        pass

    @abc.abstractmethod
    async def boot(self) -> bool:
        """
        Boots up the raw content store. This can include checking for path existence,
        connecting to a remote service, or other initialization.

        Returns:
            True if boot-up was successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def shutdown(self) -> bool:
        """
        Shuts down the raw content store, releasing any resources.

        Returns:
            True if shutdown was successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def store_content(self, content_id: str, content: Union[str, Dict, bytes],
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Stores raw content. The content_id is typically derived from the memory_id
        it's associated with in the LTM.

        Args:
            content_id: A unique identifier for this piece of content.
            content: The raw content, which can be a string, dictionary (e.g., for JSON), or bytes.
            metadata: (Optional) A dictionary of metadata associated with the content.
                      This metadata is specific to the raw content itself, not the vector embedding.
        """
        pass

    @abc.abstractmethod
    async def retrieve_content(self, content_id: str) -> Optional[Union[str, Dict, bytes]]:
        """
        Retrieves raw content by its content_id.

        Args:
            content_id: The unique identifier of the content to retrieve.

        Returns:
            The raw content (str, Dict, or bytes) if found, otherwise None.
            Implementations should handle deserialization if content was stored in a specific format (e.g., JSON).
        """
        pass

    @abc.abstractmethod
    async def get_content_metadata(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves any metadata stored alongside the raw content.
        Note: Not all stores might keep separate metadata if it's embedded within the content file (e.g., in a JSON structure).

        Args:
            content_id: The unique identifier of the content.

        Returns:
            The metadata dictionary if found and applicable, otherwise None.
        """
        pass

    @abc.abstractmethod
    async def delete_content(self, content_id: str) -> None:
        """
        Deletes content by its content_id.

        Args:
            content_id: The unique identifier of the content to delete.
        """
        pass

    @abc.abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """
        Gets the current status and statistics of the raw content store.

        Returns:
            A dictionary containing status information, e.g.,
            {"total_items": int, "storage_location_info": str}.
            Specific content may vary by implementation.
        """
        pass


# --- Concrete Implementations ---

class FAISSVectorDB(VectorDB):
    """
    A FAISS-based implementation of the VectorDB interface.
    Manages embeddings in a FAISS index and stores associated metadata.
    This implementation handles disk persistence for the FAISS index and metadata.
    """

    def __init__(self, config: VectorDBConfig, embedding_config: EmbeddingConfig):
        """
        Initializes the FAISSVectorDB.

        Args:
            config: VectorDBConfig, where `params` should contain "index_path" for FAISS file.
            embedding_config: EmbeddingConfig, used to determine embedding dimensions and model info.
        """
        logger.info(f"Initializing FAISSVectorDB with params: {config.params}")
        self.config = config
        self.embedding_config = embedding_config
        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_id_to_memory_id: Dict[int, str] = {} # Maps internal FAISS ID to LTM memory_id
        self.memory_id_to_faiss_id: Dict[str, int] = {} # Maps LTM memory_id to internal FAISS ID
        self.memory_id_metadata: Dict[str, Dict[str, Any]] = {} # Stores metadata, keyed by LTM memory_id

        self._index_path: Optional[str] = self.config.params.get("index_path") if self.config.params else None
        self._metadata_path: Optional[str] = None
        if self._index_path:
            self._metadata_path = self._index_path + ".metadata.json"

        self.embedding_dimension: Optional[int] = None # Will be set in boot

        if faiss is None:
            logger.error("FAISS library is not installed. FAISSVectorDB cannot operate. Please install 'faiss-cpu' or 'faiss-gpu'.")
            # This state should ideally prevent successful boot or operation.

    async def boot(self) -> bool:
        """
        Boots the FAISSVectorDB.
        - Determines embedding dimension (from EmbeddingConfig.dimension, EmbeddingConfig.params,
          model in EmbeddingConfig, or VectorDBConfig.params as fallback).
        - Loads FAISS index from `config.params["index_path"]` if it exists.
        - Loads associated metadata (mappings and per-memory metadata) from a JSON file.
        - If index or metadata don't exist, initializes a new FAISS index and empty mappings.
        - Validates consistency between loaded index, metadata, and expected dimensions.
        """
        if faiss is None: # Check again in case instance was created despite earlier log
            logger.error("FAISS library not installed. Cannot boot FAISSVectorDB.")
            return False

        logger.info(f"FAISSVectorDB booting up. Index path: {self._index_path}")
        try:
            # Determine embedding dimension. This is crucial for initializing or validating the FAISS index.
            # The LTM.boot() method should ensure EmbeddingConfig.dimension is authoritative after model load.
            if self.embedding_config and self.embedding_config.dimension:
                self.embedding_dimension = self.embedding_config.dimension
                logger.info(f"Using embedding dimension {self.embedding_dimension} from EmbeddingConfig.dimension (set by LTM).")
            elif self.embedding_config and self.embedding_config.params and "dimension" in self.embedding_config.params:
                self.embedding_dimension = self.embedding_config.params["dimension"]
                logger.info(f"Using embedding dimension {self.embedding_dimension} from EmbeddingConfig.params.")
            elif self.embedding_config and self.embedding_config.model_name and SentenceTransformer is not None:
                # This inference step can be slow and might be better handled by LTM explicitly setting EmbeddingConfig.dimension.
                try:
                    logger.info(f"Attempting to infer dimension from model: {self.embedding_config.model_name} (this might take a moment).")
                    temp_model = SentenceTransformer(self.embedding_config.model_name)
                    self.embedding_dimension = temp_model.get_sentence_embedding_dimension()
                    del temp_model # Release model
                    logger.info(f"Inferred embedding dimension {self.embedding_dimension} from model {self.embedding_config.model_name}.")
                except Exception as e:
                    logger.warning(f"Could not load model {self.embedding_config.model_name} to infer dimension: {e}. Will check VectorDBConfig as fallback.")
                    if self.config.params and "dimension" in self.config.params:
                         self.embedding_dimension = self.config.params.get("dimension")
                         logger.info(f"Using embedding dimension {self.embedding_dimension} from VectorDBConfig.params as fallback after model load failure.")
            elif self.config.params and "dimension" in self.config.params: # Fallback to VectorDB config params
                self.embedding_dimension = self.config.params.get("dimension")
                logger.info(f"Using embedding dimension {self.embedding_dimension} from VectorDBConfig.params.")

            if not self.embedding_dimension:
                logger.error("FAISSVectorDB: Embedding dimension could not be determined. "
                             "Ensure it's set in EmbeddingConfig (dimension or params.dimension), "
                             "inferable from EmbeddingConfig.model_name, or provided in VectorDBConfig.params.dimension.")
                return False

            logger.info(f"FAISSVectorDB will use embedding dimension: {self.embedding_dimension}")

            # Load FAISS index and metadata from disk if paths are configured and files exist
            if self._index_path and await asyncio.to_thread(os.path.exists, self._index_path):
                logger.info(f"Loading FAISS index from {self._index_path}...")
                self.faiss_index = await asyncio.to_thread(faiss.read_index, self._index_path)
                logger.info(f"FAISS index loaded. Index has {self.faiss_index.ntotal} vectors of dimension {self.faiss_index.d}.")
                if self.faiss_index.d != self.embedding_dimension:
                    logger.error(f"Loaded FAISS index dimension ({self.faiss_index.d}) does not match model dimension ({self.embedding_dimension}).")
                    return False # Critical mismatch

                # Load metadata associated with the FAISS index
                if self._metadata_path and await asyncio.to_thread(os.path.exists, self._metadata_path):
                    try:
                        with open(self._metadata_path, 'r', encoding='utf-8') as f:
                            content = await asyncio.to_thread(f.read)
                        if not content.strip(): # Check if metadata file is empty
                            logger.warning(f"FAISS metadata file {self._metadata_path} is empty.")
                            if self.faiss_index.ntotal > 0: # Index has data but metadata is empty
                                logger.error("FAISS index has data but metadata file is empty. This is a critical error.")
                                return False
                        else: # Metadata file has content, try to load
                            loaded_metadata = json.loads(content)
                            self.faiss_id_to_memory_id = {int(k): v for k, v in loaded_metadata.get("faiss_id_to_memory_id", {}).items()}
                            self.memory_id_to_faiss_id = loaded_metadata.get("memory_id_to_faiss_id", {})
                            self.memory_id_metadata = loaded_metadata.get("memory_id_metadata", {})
                            logger.info(f"Loaded FAISS metadata from {self._metadata_path}. Mappings: {len(self.faiss_id_to_memory_id)} IDs.")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from FAISS metadata file {self._metadata_path}: {e}")
                        if self.faiss_index.ntotal > 0: return False # Critical if index has data but metadata is corrupt
                    except Exception as e: # Catch other potential errors during metadata loading
                        logger.error(f"Error loading FAISS metadata from {self._metadata_path}: {e}", exc_info=True)
                        if self.faiss_index.ntotal > 0: return False # Critical if index has data
                else: # No metadata file found
                    logger.info(f"No FAISS metadata file found at {self._metadata_path}. Starting with empty mappings.")
                    if self.faiss_index.ntotal > 0: # Index has data but no metadata file
                        logger.error("FAISS index has data but no metadata file found. This implies data loss or corruption.")
                        return False # Forcing consistency
                    # If index is also empty, it's fine to start with fresh empty mappings
                    self.faiss_id_to_memory_id = {}
                    self.memory_id_to_faiss_id = {}
                    self.memory_id_metadata = {}
            else: # No index file found, or path not configured; create a new index
                logger.info(f"FAISS index not found at {self._index_path} (or path not configured). Creating new FAISS index.")
                if not self.embedding_dimension: # Should have been determined by now
                     logger.error("Cannot create new FAISS index without determined embedding_dimension.")
                     return False
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension) # Using L2 distance
                self.faiss_id_to_memory_id = {}
                self.memory_id_to_faiss_id = {}
                self.memory_id_metadata = {}
                logger.info(f"New FAISS index (IndexFlatL2) created with dimension {self.embedding_dimension}.")

            return True
        except Exception as e: # Catch-all for any other unexpected error during boot
            logger.error(f"Error during FAISSVectorDB boot: {e}", exc_info=True)
            self.faiss_index = None # Ensure index is None if boot fails
            return False

    async def shutdown(self) -> bool:
        """
        Saves the FAISS index and its associated metadata to disk if configured.
        """
        logger.info("FAISSVectorDB shutting down...")
        if self.faiss_index is not None and self._index_path:
            try:
                if self.faiss_index.ntotal > 0: # Only save if there's something to save
                    logger.info(f"Saving FAISS index with {self.faiss_index.ntotal} vectors to {self._index_path}...")
                    await asyncio.to_thread(faiss.write_index, self.faiss_index, self._index_path)
                    logger.info("FAISS index saved.")

                    # Save metadata atomically (write to temp file then replace)
                    if self._metadata_path:
                        metadata_to_save = {
                            "faiss_id_to_memory_id": self.faiss_id_to_memory_id,
                            "memory_id_to_faiss_id": self.memory_id_to_faiss_id,
                            "memory_id_metadata": self.memory_id_metadata
                        }
                        temp_metadata_path = self._metadata_path + ".tmp"
                        try:
                            with open(temp_metadata_path, 'w', encoding='utf-8') as f:
                                await asyncio.to_thread(json.dump, metadata_to_save, f, indent=4)
                            await asyncio.to_thread(os.replace, temp_metadata_path, self._metadata_path)
                            logger.info(f"FAISS metadata saved to {self._metadata_path}")
                        except IOError as e:
                            logger.error(f"IOError saving FAISS metadata to {self._metadata_path}: {e}", exc_info=True)
                            if await asyncio.to_thread(os.path.exists, temp_metadata_path): # Clean up temp file if error
                                await asyncio.to_thread(os.remove, temp_metadata_path)
                        except Exception as e:
                             logger.error(f"Unexpected error saving FAISS metadata: {e}", exc_info=True)
                             if await asyncio.to_thread(os.path.exists, temp_metadata_path):
                                await asyncio.to_thread(os.remove, temp_metadata_path)
                    else:
                        logger.warning("FAISS metadata path not configured. Metadata not saved.")
                else: # Index is empty
                    logger.info("FAISS index is empty. Not saving index to disk.")
                    # If index is empty, also remove existing metadata file for consistency if it exists
                    if self._metadata_path and await asyncio.to_thread(os.path.exists, self._metadata_path):
                        try:
                            await asyncio.to_thread(os.remove, self._metadata_path)
                            logger.info(f"Deleted existing FAISS metadata file {self._metadata_path} as index is empty.")
                        except OSError as e:
                            logger.error(f"Error deleting FAISS metadata file {self._metadata_path}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self._index_path}: {e}", exc_info=True)
        return True

    async def store_embedding(self, memory_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Stores a single embedding. Handles updates by removing and re-adding the vector."""
        if not self.faiss_index or self.embedding_dimension is None:
            logger.error("FAISSVectorDB not properly booted or dimension not set. Cannot store embedding.")
            raise RuntimeError("FAISSVectorDB not properly booted or dimension not set.")

        embedding_np = np.array([embedding], dtype=np.float32)
        if embedding_np.shape[1] != self.embedding_dimension:
            logger.error(f"Embedding dimension mismatch for {memory_id}. Expected {self.embedding_dimension}, got {embedding_np.shape[1]}.")
            raise ValueError(f"Embedding dimension mismatch for {memory_id}.")

        # Handle update if memory_id already exists
        if memory_id in self.memory_id_to_faiss_id:
            old_faiss_id = self.memory_id_to_faiss_id[memory_id]
            logger.info(f"Updating memory_id '{memory_id}' previously at FAISS ID {old_faiss_id}.")
            # FAISS's remove_ids can be complex with IndexFlatL2 as it reorders IDs.
            # For robust updates, IndexIDMap2 is better. Here, we remove then re-add,
            # accepting that the internal FAISS ID for this memory_id will change.
            ids_to_remove = np.array([old_faiss_id], dtype=np.int64)
            try:
                await asyncio.to_thread(self.faiss_index.remove_ids, ids_to_remove)
                # Clean up old mapping from faiss_id_to_memory_id.
                # Other FAISS IDs might have shifted, this is a known limitation for IndexFlatL2.
                if old_faiss_id in self.faiss_id_to_memory_id:
                    del self.faiss_id_to_memory_id[old_faiss_id]
                logger.info(f"Removed old FAISS entry for {memory_id} at FAISS ID {old_faiss_id}. Will re-add.")
            except Exception as e:
                logger.error(f"Error removing FAISS ID {old_faiss_id} for update of {memory_id}: {e}. Proceeding to add may cause duplicates or errors.")
                # Depending on desired strictness, could raise an error here.

        # Add the new (or updated) embedding
        await asyncio.to_thread(self.faiss_index.add, embedding_np)
        new_faiss_id = self.faiss_index.ntotal - 1 # FAISS IDs are 0-indexed

        # Update mappings
        self.faiss_id_to_memory_id[new_faiss_id] = memory_id
        self.memory_id_to_faiss_id[memory_id] = new_faiss_id
        self.memory_id_metadata[memory_id] = metadata
        logger.debug(f"Stored embedding for {memory_id} at new FAISS ID {new_faiss_id}. Metadata: {metadata}")

    async def store_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> None:
        """Stores a batch of embeddings. Updates are handled by falling back to single store_embedding."""
        if not self.faiss_index or self.embedding_dimension is None:
            logger.error("FAISSVectorDB not properly booted or dimension not set. Cannot store embeddings.")
            raise RuntimeError("FAISSVectorDB not properly booted or dimension not set.")

        if not embeddings_data: return

        new_embeddings_memory_ids = []
        new_embeddings_vectors = []
        new_embeddings_metadata = []

        for data in embeddings_data:
            memory_id = data["memory_id"]
            embedding = data["embedding"]
            metadata = data["metadata"]

            if memory_id in self.memory_id_to_faiss_id:
                # If an ID already exists, handle it as an update using the single store method.
                logger.warning(f"Memory ID {memory_id} in batch already exists. Updating via single store_embedding.")
                await self.store_embedding(memory_id, embedding, metadata)
            else:
                # Validate embedding dimension before batching
                embedding_np_check = np.array(embedding, dtype=np.float32)
                if embedding_np_check.shape[0] != self.embedding_dimension: # Assuming embedding is List[float]
                    logger.error(f"Embedding for {memory_id} has incorrect dimension {len(embedding)}, expected {self.embedding_dimension}. Skipping.")
                    continue
                new_embeddings_memory_ids.append(memory_id)
                new_embeddings_vectors.append(embedding) # Store as list, convert to np.array later
                new_embeddings_metadata.append(metadata)

        if not new_embeddings_vectors: # All items might have been updates or skips
            logger.info("No new embeddings to add in batch after filtering updates/errors.")
            return

        embeddings_np = np.array(new_embeddings_vectors, dtype=np.float32)

        start_faiss_id = self.faiss_index.ntotal
        await asyncio.to_thread(self.faiss_index.add, embeddings_np) # Add all new embeddings in one go

        # Update mappings for the newly added embeddings
        for i, memory_id in enumerate(new_embeddings_memory_ids):
            new_faiss_id = start_faiss_id + i
            self.faiss_id_to_memory_id[new_faiss_id] = memory_id
            self.memory_id_to_faiss_id[memory_id] = new_faiss_id
            self.memory_id_metadata[memory_id] = new_embeddings_metadata[i]
        logger.info(f"Stored batch of {len(new_embeddings_memory_ids)} new embeddings. Handled {len(embeddings_data) - len(new_embeddings_memory_ids)} as updates/skips.")

    async def retrieve_similar_embeddings(self, query_embedding: List[float], top_k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieves similar embeddings, with optional metadata filtering (post-retrieval)."""
        if not self.faiss_index :
            logger.info("FAISS index not booted. No memories to retrieve.")
            return []
        if self.faiss_index.ntotal == 0:
            logger.info("FAISS index is empty. No memories to retrieve.")
            return []

        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        if query_embedding_np.shape[1] != self.embedding_dimension:
            logger.error(f"Query embedding dimension mismatch. Expected {self.embedding_dimension}, got {query_embedding_np.shape[1]}.")
            return []

        # Simple filter strategy: fetch more candidates if filtering, then filter in Python.
        # This is not optimal for large-scale filtering but works for basic cases.
        # A more advanced implementation might use FAISS's filtering capabilities if available with the index type.
        k_to_fetch = top_k * 5 if metadata_filter and top_k > 0 else top_k
        k_to_fetch = min(k_to_fetch, self.faiss_index.ntotal) # Don't fetch more than available
        if k_to_fetch == 0 and top_k > 0 :
             k_to_fetch = top_k
        if k_to_fetch == 0: # Avoids FAISS error for k=0 search
            return []

        # Perform search in FAISS index
        distances, faiss_ids_arr = await asyncio.to_thread(
            self.faiss_index.search, query_embedding_np, k_to_fetch
        )

        results: List[Dict[str, Any]] = []
        if faiss_ids_arr.size == 0 or (faiss_ids_arr.ndim > 1 and faiss_ids_arr[0][0] == -1) : # Check for no results
            return results

        # Process results and apply metadata filter if any
        for i in range(faiss_ids_arr.shape[1]): # Iterate through neighbors for the query
            faiss_id = int(faiss_ids_arr[0][i])
            if faiss_id == -1: continue # Should not happen if k_to_fetch <= ntotal and ntotal > 0

            memory_id = self.faiss_id_to_memory_id.get(faiss_id)
            if not memory_id: # Should not happen if mappings are consistent
                logger.warning(f"FAISS ID {faiss_id} found in search but not in faiss_id_to_memory_id map. Possible inconsistency.")
                continue

            item_metadata = self.memory_id_metadata.get(memory_id)
            if not item_metadata: # Should not happen
                logger.warning(f"Metadata for memory_id '{memory_id}' (FAISS ID {faiss_id}) not found. Skipping.")
                continue

            # Apply metadata filter (post-retrieval)
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    meta_val = item_metadata.get(key)
                    if isinstance(value, list): # Filter value is a list, check for any common item
                        if not isinstance(meta_val, list) or not any(v_item in meta_val for v_item in value):
                            match = False; break
                    elif isinstance(meta_val, list): # Metadata value is list, filter value must be in it
                        if value not in meta_val:
                            match = False; break
                    elif meta_val != value: # Direct comparison
                        match = False; break
                if not match:
                    logger.debug(f"Memory ID '{memory_id}' filtered out by metadata: {metadata_filter}")
                    continue

            results.append({
                "memory_id": memory_id,
                "score": float(distances[0][i]), # L2 distance from FAISS, smaller is better
                "metadata": item_metadata
            })
            if len(results) >= top_k: break # Stop if we have enough results after filtering
        return results

    async def get_embedding_by_id(self, memory_id: str) -> Optional[List[float]]:
        """Retrieves a specific embedding vector by its memory_id, if supported by the FAISS index."""
        if not self.faiss_index:
            logger.warning("FAISS index not available for get_embedding_by_id.")
            return None
        faiss_id = self.memory_id_to_faiss_id.get(memory_id)
        if faiss_id is None:
            logger.debug(f"Memory ID {memory_id} not found in memory_id_to_faiss_id map.")
            return None
        if faiss_id < 0 or faiss_id >= self.faiss_index.ntotal: # Validate FAISS ID range
            logger.warning(f"FAISS ID {faiss_id} for {memory_id} is out of bounds for current index size {self.faiss_index.ntotal}.")
            return None

        try:
            # FAISS `reconstruct` method can retrieve the vector if the index supports it (e.g., IndexFlatL2 does).
            if not self.faiss_index.is_trained: # Should always be true for IndexFlatL2 after creation
                 logger.warning(f"FAISS index not trained, cannot reconstruct vector for {memory_id}")
                 return None
            vector = await asyncio.to_thread(self.faiss_index.reconstruct, faiss_id)
            return vector.tolist() if vector is not None else None
        except RuntimeError as e: # `reconstruct` can raise RuntimeError for some IDs or index types
            logger.warning(f"Could not reconstruct vector for FAISS ID {faiss_id} (memory_id: {memory_id}): {e}")
            return None
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error reconstructing FAISS ID {faiss_id} for {memory_id}: {e}", exc_info=True)
            return None


    async def get_metadata_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific memory_id."""
        return self.memory_id_metadata.get(memory_id)

    async def delete_embedding(self, memory_id: str) -> None:
        """Deletes a single embedding by its memory_id."""
        await self.delete_embeddings([memory_id]) # Delegate to batch delete

    async def delete_embeddings(self, memory_ids: List[str]) -> None:
        """Deletes multiple embeddings. Note: FAISS ID remapping with IndexFlatL2 can be complex."""
        if not self.faiss_index:
            logger.warning("FAISS index not available for delete_embeddings.")
            return

        faiss_ids_to_remove_list = []
        memory_ids_successfully_identified_for_removal = []

        for memory_id in memory_ids:
            faiss_id = self.memory_id_to_faiss_id.get(memory_id)
            if faiss_id is not None:
                # Ensure the FAISS ID is within the current valid range of the index
                if faiss_id >= 0 and faiss_id < self.faiss_index.ntotal:
                    faiss_ids_to_remove_list.append(faiss_id)
                    memory_ids_successfully_identified_for_removal.append(memory_id)
                else:
                    logger.warning(f"FAISS ID {faiss_id} for memory_id {memory_id} is out of bounds for current index size {self.faiss_index.ntotal}. Skipping deletion for this ID.")
                    # Clean up potentially stale mappings if ID is bad but present
                    self.memory_id_to_faiss_id.pop(memory_id, None)
                    # Corresponding faiss_id_to_memory_id entry is harder to clean without iterating or knowing the bad ID.
            else:
                logger.warning(f"Memory ID {memory_id} not found for deletion in mappings.")

        if not faiss_ids_to_remove_list:
            logger.info("No valid FAISS IDs found to remove.")
            return

        # Critical Note for IndexFlatL2 (and similar simple indexes like IndexFlatIP):
        # faiss.Index.remove_ids() re-compacts the index. This means that the FAISS IDs of vectors
        # *after* the removed ones will change. This invalidates existing `faiss_id_to_memory_id` and
        # `memory_id_to_faiss_id` mappings for those shifted vectors.
        # A robust solution for frequent deletions typically involves:
        # 1. Using `faiss.IndexIDMap2` on top of an index: This maps external IDs to internal FAISS IDs
        #    and handles removals correctly without broadly invalidating other IDs.
        # 2. Or, marking vectors as "deleted" in metadata and periodically rebuilding the index and maps from "active" vectors.
        # This current implementation proceeds with `remove_ids` for simplicity and then clears mappings for the
        # deleted items. It acknowledges that other FAISS IDs in existing maps might become stale if not carefully managed.
        logger.warning("FAISSVectorDB.delete_embeddings with IndexFlatL2: `remove_ids` operation will "
                       "invalidate FAISS ID mappings for vectors beyond those removed. This implementation "
                       "removes specified items but does not currently resynchronize all other potentially shifted IDs. "
                       "For production use with frequent deletes, consider IndexIDMap2 or a periodic index rebuild strategy.")

        # FAISS remove_ids expects a sorted array of unique IDs.
        ids_to_remove_np = np.array(sorted(list(set(faiss_ids_to_remove_list))), dtype=np.int64)

        try:
            num_removed = await asyncio.to_thread(self.faiss_index.remove_ids, ids_to_remove_np)
            # The return value of remove_ids (num_removed) might not be consistently useful across all FAISS versions/index types.
            logger.info(f"Attempted removal of {len(ids_to_remove_np)} unique FAISS IDs. FAISS remove_ids result (if any): {num_removed}.")

            # Update internal mappings for the explicitly deleted items
            for memory_id in memory_ids_successfully_identified_for_removal:
                old_faiss_id = self.memory_id_to_faiss_id.pop(memory_id, None)
                if old_faiss_id is not None: # If it was in memory_id_to_faiss_id map
                    self.faiss_id_to_memory_id.pop(old_faiss_id, None) # Remove from reverse map too
                self.memory_id_metadata.pop(memory_id, None) # Remove associated metadata

            # Acknowledge potential inconsistency of other mappings due to ID shifts post-removal.
            if num_removed is not None and num_removed > 0 :
                 logger.warning(f"FAISS IDs may have been remapped after deletion. Mappings for non-deleted items could be stale. Total items now: {self.faiss_index.ntotal}")
                 # A full remap or resync strategy would be needed here for perfect consistency with IndexFlatL2.

        except RuntimeError as e: # Catch errors from faiss.remove_ids (e.g., ID not found if index is strict)
            logger.error(f"RuntimeError during FAISS remove_ids: {e}. Mappings for intended deletions might be partially complete.", exc_info=True)
            # Attempt to clean up mappings for those successfully identified even if FAISS call failed,
            # as they might be inconsistent if some removals occurred before the error.
            for memory_id in memory_ids_successfully_identified_for_removal:
                if memory_id in self.memory_id_to_faiss_id: # If it wasn't popped before error
                    old_faiss_id = self.memory_id_to_faiss_id.pop(memory_id, None)
                    if old_faiss_id is not None:
                        self.faiss_id_to_memory_id.pop(old_faiss_id, None)
                    self.memory_id_metadata.pop(memory_id, None)
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during FAISS delete_embeddings: {e}", exc_info=True)


    async def get_status(self) -> Dict[str, Any]:
        """Returns status of the FAISS database, including total vectors and dimension."""
        if not self.faiss_index:
            return {"total_vectors": 0, "dimension": self.embedding_dimension or 0, "status": "Not Initialized", "error": "FAISS index is None."}

        status_dict = {
            "total_vectors": self.faiss_index.ntotal,
            "dimension": self.faiss_index.d if self.faiss_index else self.embedding_dimension, # faiss_index.d is authoritative if index exists
            "index_path": self._index_path or "Not configured",
            "metadata_path": self._metadata_path or "Not configured",
            "is_trained": self.faiss_index.is_trained if self.faiss_index else False, # For IndexFlatL2, always true after init
            "status": "OK"
        }
        # Example: Add specific FAISS index type if useful
        # status_dict["faiss_index_type"] = type(self.faiss_index).__name__ if self.faiss_index else "N/A"
        return status_dict


class FileSystemRawContentStore(RawContentStore):
    """
    A RawContentStore implementation that uses the local file system for storage.
    Content is typically stored as individual files, potentially using JSON for
    structured data (content + metadata).
    """

    def __init__(self, config: Union[FileSystemConfig, Any]): # S3Config, RedisConfig removed from Union for now
        """
        Initializes the FileSystemRawContentStore.

        Args:
            config: Must be a FileSystemConfig instance containing the `base_path` for storage.

        Raises:
            ValueError: If the provided config is not a FileSystemConfig.
        """
        if not isinstance(config, FileSystemConfig):
            raise ValueError("FileSystemRawContentStore requires a FileSystemConfig.")
        logger.info(f"Initializing FileSystemRawContentStore with base_path: {config.base_path}")
        self.config = config
        self._base_path: Optional[str] = None # Set during boot

    async def boot(self) -> bool:
        """
        Boots the FileSystemRawContentStore. Ensures the `base_path` directory exists.
        """
        logger.info("FileSystemRawContentStore booting up...")
        if not self.config.base_path:
            logger.error("FileSystemConfig base_path is not set.")
            return False
        self._base_path = self.config.base_path
        try:
            # Create the base directory if it doesn't exist.
            await asyncio.to_thread(os.makedirs, self._base_path, exist_ok=True)
            logger.info(f"Raw content store directory (FileSystem) '{self._base_path}' ensured.")
            return True
        except Exception as e:
            logger.error(f"Error creating directory for FileSystemRawContentStore at {self._base_path}: {e}", exc_info=True)
            return False

    async def shutdown(self) -> bool:
        """Shuts down the FileSystemRawContentStore. No specific actions needed for filesystem."""
        logger.info("FileSystemRawContentStore shutting down. (No specific actions to take for filesystem based store)")
        return True

    async def store_content(self, content_id: str, content: Union[str, Dict, bytes], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Stores content to a file named after `content_id` in the `base_path`.
        - If `content` is a dict, or if `content` is a string and `metadata` is provided,
          it's stored as a JSON object: `{"content": ..., "metadata": ...}`.
        - If `content` is a string and no `metadata`, it's stored as a raw text file.
        - If `content` is bytes, it's stored as a raw binary file. Metadata for bytes is currently logged as a warning and not stored with the bytes.
        """
        if not self._base_path:
            raise RuntimeError("FileSystemRawContentStore not properly booted or base_path not set.")

        file_path = os.path.join(self._base_path, content_id) # Assume content_id is a valid filename component

        data_to_store: Any
        mode: str
        encoding: Optional[str]

        if isinstance(content, dict):
            # Store dictionary content directly, with associated metadata in a structured way.
            data_to_store = {"content": content, "metadata": metadata or {}}
            mode = 'w'
            encoding = 'utf-8'
        elif isinstance(content, str):
            if metadata: # If string content has metadata, store both in a JSON structure
                 data_to_store = {"content": content, "metadata": metadata}
                 mode = 'w'
                 encoding = 'utf-8'
            else: # Store raw string directly
                data_to_store = content
                mode = 'w'
                encoding = 'utf-8'
        elif isinstance(content, bytes):
            if metadata:
                # Storing bytes with separate metadata is complex for a single file strategy.
                # This simple implementation logs a warning and stores only the bytes.
                # A more advanced store might use a sidecar file for metadata or a structured binary format.
                logger.warning(f"Storing bytes content for {content_id} with metadata. Metadata will NOT be stored directly with bytes in this simple implementation.")
            data_to_store = content
            mode = 'wb'
            encoding = None
        else:
            logger.error(f"Unsupported content type for store_content: {type(content)}")
            raise TypeError(f"Unsupported content type: {type(content)}")

        try:
            logger.debug(f"Storing content for ID '{content_id}' at {file_path}")
            # Define synchronous I/O operation to be run in a thread
            def _io_op():
                with open(file_path, mode, encoding=encoding) as f:
                    if isinstance(data_to_store, dict) or (isinstance(data_to_store, str) and metadata and data_to_store is not content): # JSON cases
                        json.dump(data_to_store, f, indent=4)
                    else: # Raw string (without metadata) or bytes
                        f.write(data_to_store)

            await asyncio.to_thread(_io_op) # Execute blocking I/O in a separate thread
            logger.info(f"Stored content for ID '{content_id}' at {file_path}")
        except IOError as e:
            logger.error(f"IOError storing content to {file_path}: {e}", exc_info=True)
            raise
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error storing content to {file_path}: {e}", exc_info=True)
            raise


    async def retrieve_content(self, content_id: str) -> Optional[Union[str, Dict, bytes]]:
        """
        Retrieves content from a file. Tries to load as JSON first, then plain text, then raw bytes.
        """
        if not self._base_path:
            logger.error("FileSystemRawContentStore not properly booted or base_path not set.")
            return None

        file_path = os.path.join(self._base_path, content_id)
        if not await asyncio.to_thread(os.path.exists, file_path):
            logger.debug(f"Content file not found: {file_path} for ID '{content_id}'")
            return None

        try:
            logger.debug(f"Retrieving content for ID '{content_id}' from {file_path}")
            # This heuristic relies on how store_content saves data.
            def _io_op():
                try: # Try JSON (covers dict and str+metadata cases)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # If it was stored as {"content": ..., "metadata": ...}, LTM expects this package.
                        # If it was stored as just a JSON string or list/dict directly, that's also fine.
                        return data
                except json.JSONDecodeError:
                    # Not JSON, try reading as plain text (covers raw string stored without metadata)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            return f.read()
                    except UnicodeDecodeError:
                        # Not plain text, try reading as bytes (covers raw bytes stored)
                        with open(file_path, 'rb') as f:
                            return f.read()
                except Exception as e_inner:
                    logger.error(f"Error during tiered read for {file_path}: {e_inner}")
                    return None

            content = await asyncio.to_thread(_io_op)
            if content is not None:
                 logger.info(f"Retrieved content for ID '{content_id}' from {file_path}")
            return content
        except IOError as e:
            logger.error(f"IOError retrieving content from {file_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving content from {file_path}: {e}", exc_info=True)
            return None


    async def get_content_metadata(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata if content was stored as a JSON object with a "metadata" key.
        Returns None if the content is not JSON or lacks this structure.
        """
        if not self._base_path:
            logger.error("FileSystemRawContentStore not properly booted or base_path not set.")
            return None

        file_path = os.path.join(self._base_path, content_id)
        if not await asyncio.to_thread(os.path.exists, file_path):
            logger.debug(f"Content file not found for metadata check: {file_path}")
            return None

        try:
            def _io_op_metadata():
                # Only try to read as JSON, as that's how structured content+metadata is stored.
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and "metadata" in data:
                    return data["metadata"]
                # If it's a JSON file but not in the {"content": ..., "metadata": ...} structure,
                # then it's considered to have no separate metadata according to this store's convention.
                return None

            metadata = await asyncio.to_thread(_io_op_metadata)
            if metadata:
                logger.info(f"Retrieved metadata for ID '{content_id}'")
            return metadata
        except json.JSONDecodeError:
            logger.debug(f"File {file_path} is not JSON, cannot extract structured metadata.")
            return None
        except IOError as e:
            logger.error(f"IOError reading metadata from {file_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading metadata from {file_path}: {e}", exc_info=True)
            return None


    async def delete_content(self, content_id: str) -> None:
        """Deletes the content file associated with content_id."""
        if not self._base_path:
            raise RuntimeError("FileSystemRawContentStore not properly booted or base_path not set.")

        file_path = os.path.join(self._base_path, content_id)
        try:
            if await asyncio.to_thread(os.path.exists, file_path):
                await asyncio.to_thread(os.remove, file_path)
                logger.info(f"Deleted content file: {file_path}")
            else:
                logger.info(f"Content file not found for deletion: {file_path}")
        except IOError as e:
            logger.error(f"IOError deleting content file {file_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting content file {file_path}: {e}", exc_info=True)
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Returns status of the FileSystem store, including base path and approximate file count."""
        if not self._base_path:
            return {"base_path": "Not configured", "total_files": 0, "status": "Not Initialized"}

        total_files = 0
        try:
            if await asyncio.to_thread(os.path.isdir, self._base_path):
                # Note: os.listdir can be slow for directories with a very large number of files.
                # For production systems, a more scalable way to track total_files might be needed if this becomes a bottleneck.
                total_files = await asyncio.to_thread(lambda: len(os.listdir(self._base_path)))
        except Exception as e:
            logger.warning(f"Could not count files in {self._base_path} for status: {e}")

        return {
            "base_path": self._base_path,
            "total_files": total_files,
            "status": "OK" if self._base_path else "Not Initialized" # Should always be OK if _base_path is set
        }


class LTM:
    """
    Long-Term Memory (LTM) system.
    Orchestrates embedding generation (with caching), vector storage via a VectorDB interface,
    and raw content storage via a RawContentStore interface.
    """

    def __init__(self,
                 vector_db: VectorDB,
                 raw_content_store: RawContentStore,
                 embedding_config: EmbeddingConfig,
                 embedding_cache: RawCache):
        """
        Initializes the LongTermMemory system.

        Args:
            vector_db: An instance of a class implementing the VectorDB interface.
            raw_content_store: An instance of a class implementing the RawContentStore interface.
            embedding_config: Configuration for embedding generation (model name, device, etc.).
            embedding_cache: A RawCache instance for caching generated embeddings.
        """
        logger.info("Initializing LongTermMemory (LTM) system with provided VectorDB and RawContentStore.")

        self.vector_db = vector_db
        self.raw_content_store = raw_content_store
        self.embedding_config = embedding_config
        self.embedding_cache = embedding_cache

        self.embedding_model: Optional[SentenceTransformer] = None # Loaded during boot()
        self.embedding_dimension: Optional[int] = None # Determined after model loading in boot()

        logger.info(f"LTM initialized. Embedding model target: {self.embedding_config.model_name}, "
                    f"VectorDB type: {type(self.vector_db).__name__}, "
                    f"RawContentStore type: {type(self.raw_content_store).__name__}, "
                    f"Embedding Cache type: {type(self.embedding_cache).__name__}")

    async def boot(self) -> bool:
        """
        Boots the LTM system:
        1. Loads the SentenceTransformer model for embedding generation.
        2. Sets/updates `EmbeddingConfig.dimension` with the actual model dimension.
        3. Boots the provided VectorDB instance (passing the updated `embedding_config`).
        4. Boots the provided RawContentStore instance.
        5. Boots the embedding cache.

        Returns:
            True if all components boot successfully, False otherwise.
        """
        logger.info("LTM booting up...")
        try:
            # 1. Load SentenceTransformer model for embedding generation
            model_name = self.embedding_config.model_name or 'all-MiniLM-L6-v2' # Use a fallback if not specified
            if SentenceTransformer is None:
                logger.error("SentenceTransformer library not installed. LTM cannot generate embeddings.")
                return False # Critical dependency for LTM's core function

            loop = asyncio.get_running_loop()
            logger.info(f"Loading SentenceTransformer model: {model_name}...")
            self.embedding_model = await loop.run_in_executor(None, SentenceTransformer, model_name)
            # Determine and store the actual embedding dimension from the loaded model
            self.embedding_dimension = await loop.run_in_executor(None, self.embedding_model.get_sentence_embedding_dimension)
            logger.info(f"SentenceTransformer model '{model_name}' loaded. Actual dimension: {self.embedding_dimension}.")

            # Update EmbeddingConfig with the true dimension from the model.
            # This ensures that components like FAISSVectorDB use the correct dimension.
            if self.embedding_config.params is None:
                self.embedding_config.params = {}
            if self.embedding_config.dimension != self.embedding_dimension: # If primary field differs
                 logger.info(f"Updating EmbeddingConfig.dimension from {self.embedding_config.dimension} to model's actual: {self.embedding_dimension}")
                 self.embedding_config.dimension = self.embedding_dimension
            # Also ensure params reflects this, as some components might read from there.
            self.embedding_config.params["dimension"] = self.embedding_dimension


            # 2. Boot VectorDB (it will use the embedding_config, now updated with correct dimension)
            if not await self.vector_db.boot():
                logger.error("LTM: VectorDB component failed to boot.")
                return False
            logger.info("LTM: VectorDB component booted successfully.")

            # 3. Boot RawContentStore
            if not await self.raw_content_store.boot():
                logger.error("LTM: RawContentStore component failed to boot.")
                # If RawContentStore fails, attempt to shutdown already booted VectorDB
                if hasattr(self.vector_db, 'shutdown') and callable(self.vector_db.shutdown):
                    await self.vector_db.shutdown()
                return False
            logger.info("LTM: RawContentStore component booted successfully.")

            # 4. Boot embedding cache
            if not await self.embedding_cache.boot():
                 logger.warning("LTM: Embedding cache (RawCache) failed to boot or connect properly. Performance may be affected.")
            else:
                logger.info("LTM: Embedding cache booted successfully.")

            logger.info("LTM boot sequence complete.")
            return True
        except Exception as e: # Catch any unexpected error during the boot sequence
            logger.error(f"Critical error during LTM boot: {e}", exc_info=True)
            self.embedding_model = None # Clear potentially partially initialized model
            # Attempt to shutdown components that might have booted before the failure
            if hasattr(self.vector_db, 'shutdown') and callable(self.vector_db.shutdown) and self.vector_db.get_status().get("status") != "Not Initialized":
                await self.vector_db.shutdown()
            if hasattr(self.raw_content_store, 'shutdown') and callable(self.raw_content_store.shutdown) and self.raw_content_store.get_status().get("status") != "Not Initialized":
                await self.raw_content_store.shutdown()
            if hasattr(self.embedding_cache, 'shutdown') and callable(self.embedding_cache.shutdown): # Assuming cache might have a status too
                 await self.embedding_cache.shutdown()
            return False

    async def shutdown(self):
        """
        Shuts down the LTM system and its components:
        - VectorDB.
        - RawContentStore.
        - Embedding cache.
        - Releases the embedding model reference.
        """
        logger.info("LTM shutting down...")
        # Shutdown components in reverse order of boot is often a good practice,
        # though not strictly necessary here if they are independent.
        if self.embedding_cache and hasattr(self.embedding_cache, 'shutdown') and callable(self.embedding_cache.shutdown):
            await self.embedding_cache.shutdown()
            logger.info("LTM: Embedding cache shutdown.")
        if self.raw_content_store and hasattr(self.raw_content_store, 'shutdown') and callable(self.raw_content_store.shutdown):
            await self.raw_content_store.shutdown()
            logger.info("LTM: RawContentStore component shutdown.")
        if self.vector_db and hasattr(self.vector_db, 'shutdown') and callable(self.vector_db.shutdown):
            await self.vector_db.shutdown()
            logger.info("LTM: VectorDB component shutdown.")

        self.embedding_model = None # Release model object
        self.embedding_dimension = None
        logger.info("LTM shutdown complete. Embedding model released.")
        return True

    def _get_text_key(self, text_content: str) -> str:
        """Generates a deterministic cache key for text content using MD5 hash."""
        return hashlib.md5(text_content.encode('utf-8')).hexdigest()

    async def _get_embedding_from_cache(self, text_key: str) -> Optional[List[float]]:
        """
        Retrieves an embedding from the cache.
        Assumes cached value is either a list of floats or a JSON string representing one.
        """
        logger.debug(f"Checking embedding cache for key: {text_key}")
        cached_value = await self.embedding_cache.get(key=text_key, namespace="ltm_embeddings")
        if cached_value:
            if isinstance(cached_value, list): # Already a list of floats
                return cached_value
            try: # Attempt to load if it's a JSON string representation of a list
                loaded_list = json.loads(cached_value)
                if isinstance(loaded_list, list):
                    return loaded_list
            except (json.JSONDecodeError, TypeError) as e: # Catch JSON errors or if cached_value is not string/bytes
                logger.warning(f"Failed to decode cached embedding for key {text_key}. Value type: {type(cached_value)}, Error: {e}")
        return None

    async def _set_embedding_to_cache(self, text_key: str, embedding: List[float]):
        """
        Stores an embedding in the cache.
        The RawCache implementation is expected to handle serialization if necessary (e.g., for Redis).
        """
        logger.debug(f"Storing embedding in cache for key: {text_key}")
        await self.embedding_cache.set(key=text_key, value=embedding, namespace="ltm_embeddings")

    async def generate_embedding(self, text_content: str) -> List[float]:
        """
        Generates or retrieves a cached embedding for the given text content
        using the loaded SentenceTransformer model and configured cache.

        Args:
            text_content: The text for which to generate an embedding.

        Returns:
            The generated or cached embedding as a list of floats.
            Returns a zero vector of appropriate dimension on error or if model not available.
        """
        if not self.embedding_model or not self.embedding_dimension:
            logger.error("LTM Embedding model or dimension not initialized. Call boot() first.")
            # Fallback to a zero vector, using dimension from config if model not loaded.
            dim = self.embedding_dimension or (self.embedding_config.params.get("dimension", 384)
                   if self.embedding_config and self.embedding_config.params
                   else 384) # Default to a common dimension if all else fails
            return [0.0] * dim

        if not text_content: # Handle empty input gracefully
            logger.warning("Empty text_content for embedding generation. Returning default zero vector.")
            return [0.0] * self.embedding_dimension

        text_key = f"embedding:{self._get_text_key(text_content)}" # Namespaced key for clarity

        # 1. Try to retrieve from cache
        try:
            cached_embedding = await self._get_embedding_from_cache(text_key)
            if cached_embedding:
                logger.debug(f"Using cached embedding for text key: {text_key}")
                return cached_embedding
        except Exception as e: # Catch errors during cache retrieval
            logger.warning(f"Error retrieving embedding from cache for key {text_key}: {e}. Will regenerate.")

        # 2. If not cached or cache retrieval failed, generate new embedding
        try:
            logger.debug(f"Generating new embedding for text: '{text_content[:50]}...'")
            loop = asyncio.get_running_loop()
            # SentenceTransformer.encode is CPU-bound; run in executor to avoid blocking event loop.
            embedding_array = await loop.run_in_executor(
                None, self.embedding_model.encode, text_content
            )
            embedding = embedding_array.tolist() # Convert numpy array to list

            # Basic validation of the generated embedding structure
            if not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding): # Allow ints too, convert later
                 logger.warning(f"Generated embedding is not a list of numbers. Type: {type(embedding)}. Attempting conversion.")
                 embedding = [float(x) for x in embedding]
            elif not all(isinstance(x, float) for x in embedding): # If list of numbers, ensure all are float
                 embedding = [float(x) for x in embedding]


        except Exception as e:
            logger.error(f"Error generating embedding for text '{text_content[:50]}...': {e}", exc_info=True)
            return [0.0] * self.embedding_dimension # Return zero vector on error during generation

        # 3. Store the newly generated embedding in cache
        try:
            await self._set_embedding_to_cache(text_key, embedding)
        except Exception as e: # Catch errors during cache storage
            logger.warning(f"Error storing embedding to cache for key {text_key}: {e}.")

        return embedding

    async def store_memory_chunk(self, chunk_id: Optional[str], text_content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Stores a memory chunk, including its raw content and vector embedding.
        It orchestrates storage with the configured VectorDB and RawContentStore.

        Args:
            chunk_id: An optional ID for the chunk. If None, a UUID is generated.
                      This ID is used as a base for the LTM `memory_id`.
            text_content: The raw text content of the memory.
            embedding: The pre-computed vector embedding of the text_content.
            metadata: A dictionary of metadata associated with this memory chunk.

        Returns:
            The unique `memory_id` assigned to this stored memory chunk.

        Raises:
            RuntimeError: If LTM components (VectorDB, RawContentStore) are not initialized.
            ValueError: If embedding dimension mismatch occurs (checked by VectorDB).
            Any exceptions raised by the underlying VectorDB or RawContentStore during storage.
        """
        if not self.vector_db or not self.raw_content_store or not self.embedding_dimension:
            logger.error("LTM components (VectorDB, RawContentStore, or embedding dimension) not initialized. Cannot store memory.")
            raise RuntimeError("LTM is not properly initialized. Call boot() first.")

        # Generate a unique memory_id for LTM's tracking.
        # If chunk_id is provided, use it to make the memory_id more predictable/relatable if desired.
        memory_id = f"ltm_mem_{chunk_id}" if chunk_id else str(uuid.uuid4())

        # Define a consistent naming convention for the raw content identifier, linking it to memory_id.
        raw_content_id = f"raw_content_{memory_id}.json" # Example: store as JSON

        logger.info(f"Storing memory chunk: ID='{memory_id}', RawContentID='{raw_content_id}'")

        # 1. Store Raw Content via RawContentStore
        # The RawContentStore implementation (e.g., FileSystemRawContentStore)
        # will handle how the content and its specific metadata are stored.
        try:
            raw_store_metadata = metadata.copy() # Pass original user-provided metadata
            # Add an internal link back to the LTM memory_id for potential auditing or direct access if needed.
            raw_store_metadata["_ltm_memory_id"] = memory_id

            await self.raw_content_store.store_content(
                content_id=raw_content_id,
                content=text_content, # The actual text content
                metadata=raw_store_metadata
            )
            logger.debug(f"Stored raw content for {memory_id} via RawContentStore.")
        except Exception as e:
            logger.error(f"Error storing raw content for {memory_id} via RawContentStore: {e}", exc_info=True)
            raise # Re-raise critical error, as vector DB entry without raw content is problematic

        # 2. Store Embedding via VectorDB
        try:
            # Prepare metadata for the VectorDB. This must include a link to the raw_content_id.
            vector_db_metadata = metadata.copy() # Start with original user-provided metadata
            vector_db_metadata["raw_content_id"] = raw_content_id
            # The memory_id itself is the primary key for the vector in VectorDB.

            await self.vector_db.store_embedding(
                memory_id=memory_id,
                embedding=embedding,
                metadata=vector_db_metadata
            )
            logger.debug(f"Stored embedding for {memory_id} via VectorDB.")
        except Exception as e:
            logger.error(f"Error storing embedding to VectorDB for {memory_id}: {e}", exc_info=True)
            # If vector storage fails, attempt to clean up the already stored raw content to avoid orphans.
            try:
                await self.raw_content_store.delete_content(raw_content_id)
                logger.info(f"Cleaned up raw content {raw_content_id} due to VectorDB storage error.")
            except Exception as clean_e:
                logger.error(f"Error cleaning up raw content {raw_content_id} after VectorDB error: {clean_e}", exc_info=True)
            raise # Re-raise the original VectorDB error

        logger.info(f"Memory chunk '{memory_id}' stored successfully.")
        return memory_id

    async def retrieve_relevant_memories(self, query_embedding: List[float], top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memories based on semantic similarity to a query embedding.
        It first gets similar embedding information from VectorDB, then fetches
        the corresponding raw content from RawContentStore.

        Args:
            query_embedding: The embedding of the query.
            top_k: The maximum number of relevant memories to retrieve.
            metadata_filter: (Optional) A filter to apply to metadata during vector search.

        Returns:
            A list of dictionaries, each representing a retrieved memory, including its
            `memory_id`, `text_content`, original `metadata`, and `score`.
        """
        if not self.vector_db or not self.raw_content_store:
            logger.error("LTM components (VectorDB or RawContentStore) not initialized. Cannot retrieve memories.")
            return []

        logger.info(f"Retrieving up to {top_k} relevant memories. Applying filter: {metadata_filter}")

        # 1. Retrieve similar embeddings from VectorDB
        # VectorDB returns a list of items, each with memory_id, score, and its stored metadata (which includes raw_content_id)
        try:
            similar_embeddings_info = await self.vector_db.retrieve_similar_embeddings(
                query_embedding=query_embedding,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
        except Exception as e:
            logger.error(f"Error retrieving similar embeddings from VectorDB: {e}", exc_info=True)
            return []

        if not similar_embeddings_info:
            logger.info("No similar embeddings found by VectorDB.")
            return []

        # 2. Retrieve raw content for each result using raw_content_id from VectorDB's metadata
        results: List[Dict[str, Any]] = []
        for item_info in similar_embeddings_info:
            memory_id = item_info.get("memory_id")
            score = item_info.get("score")
            vector_db_item_metadata = item_info.get("metadata", {})

            if not memory_id:
                logger.warning("Found item from VectorDB without memory_id. Skipping.")
                continue

            raw_content_id = vector_db_item_metadata.get("raw_content_id")
            if not raw_content_id:
                logger.warning(f"Raw content ID missing in VectorDB metadata for memory_id '{memory_id}'. Skipping.")
                continue

            try:
                # Retrieve the content package from RawContentStore. This might be a dict (e.g., {"content": ..., "metadata": ...})
                # or raw string/bytes depending on how FileSystemRawContentStore (or other implementation) stores it.
                retrieved_package = await self.raw_content_store.retrieve_content(raw_content_id)

                text_content: Optional[str] = None
                original_user_metadata: Optional[Dict[str, Any]] = None

                if isinstance(retrieved_package, dict):
                    # This assumes a convention where RawContentStore (like FileSystemRawContentStore)
                    # stores a dictionary containing the actual text content and the original user metadata.
                    text_content = retrieved_package.get("content")
                    original_user_metadata = retrieved_package.get("metadata")
                    if original_user_metadata and "_ltm_memory_id" in original_user_metadata: # Clean up internal link
                        del original_user_metadata["_ltm_memory_id"]
                elif isinstance(retrieved_package, str):
                    # If the store returns a raw string, it means metadata was not stored with it,
                    # or it was a simple string storage. The metadata from VectorDB is the best source here for user metadata.
                    text_content = retrieved_package
                    original_user_metadata = {k: v for k, v in vector_db_item_metadata.items() if k != "raw_content_id"}
                elif isinstance(retrieved_package, bytes):
                    # If bytes, try to decode as UTF-8 for text_content. If this is not desired,
                    # the handling of bytes content needs to be more specific for the application.
                    try:
                        text_content = retrieved_package.decode('utf-8')
                        logger.debug(f"Retrieved bytes content for {raw_content_id}, decoded as UTF-8.")
                        original_user_metadata = {k: v for k, v in vector_db_item_metadata.items() if k != "raw_content_id"}
                    except UnicodeDecodeError:
                        logger.warning(f"Retrieved bytes content for {raw_content_id} could not be decoded as UTF-8. Skipping.")
                        continue # Cannot provide text_content for this memory
                else:
                    logger.warning(f"Retrieved content for {raw_content_id} is of unhandled type: {type(retrieved_package)}. Skipping.")
                    continue

                if text_content is None: # Should be caught by earlier checks, but as a safeguard
                    logger.warning(f"Text content missing for memory_id '{memory_id}' from raw content {raw_content_id}. Skipping.")
                    continue

                results.append({
                    "memory_id": memory_id,
                    "text_content": text_content,
                    "metadata": original_user_metadata or {}, # Prefer metadata from raw store if structured, else from VDB
                    "score": score,
                    "retrieval_source": f"LTM_{type(self.vector_db).__name__}_{type(self.raw_content_store).__name__}"
                })
            except Exception as e:
                logger.error(f"Error retrieving or processing raw content for {raw_content_id} (memory_id '{memory_id}'): {e}", exc_info=True)

        logger.info(f"Retrieved {len(results)} relevant memories after processing.")
        return results


    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific memory (text content and metadata) by its LTM memory_id.
        This involves fetching metadata from the VectorDB (which includes the raw_content_id)
        and then retrieving the actual content from the RawContentStore.

        Args:
            memory_id: The unique identifier of the LTM memory.

        Returns:
            A dictionary containing the memory's `memory_id`, `text_content`, and `metadata`,
            or None if not found or an error occurs.
        """
        logger.info(f"Attempting to retrieve memory by ID: {memory_id}")
        if not self.vector_db or not self.raw_content_store:
            logger.error("LTM components not initialized. Cannot get memory by ID.")
            return None

        # 1. Get metadata from VectorDB; this metadata should include the raw_content_id.
        vector_db_item_metadata = await self.vector_db.get_metadata_by_id(memory_id)
        if not vector_db_item_metadata:
            logger.warning(f"Metadata for memory ID '{memory_id}' not found in VectorDB.")
            return None

        raw_content_id = vector_db_item_metadata.get("raw_content_id")
        if not raw_content_id:
            # This indicates an inconsistency if a vectorDB entry exists without a link to its raw content.
            logger.error(f"CRITICAL: Raw content ID missing in VectorDB metadata for memory_id '{memory_id}'.")
            return None

        # 2. Retrieve raw content using raw_content_id from RawContentStore.
        try:
            retrieved_package = await self.raw_content_store.retrieve_content(raw_content_id)

            text_content: Optional[str] = None
            original_user_metadata: Optional[Dict[str, Any]] = None

            # Similar unpacking logic as in retrieve_relevant_memories
            if isinstance(retrieved_package, dict):
                text_content = retrieved_package.get("content")
                original_user_metadata = retrieved_package.get("metadata")
                if original_user_metadata and "_ltm_memory_id" in original_user_metadata: # Clean up internal link
                    del original_user_metadata["_ltm_memory_id"]
            elif isinstance(retrieved_package, str):
                text_content = retrieved_package
                original_user_metadata = {k: v for k, v in vector_db_item_metadata.items() if k != "raw_content_id"}
            elif isinstance(retrieved_package, bytes):
                try:
                    text_content = retrieved_package.decode('utf-8')
                    original_user_metadata = {k: v for k, v in vector_db_item_metadata.items() if k != "raw_content_id"}
                except UnicodeDecodeError:
                     logger.warning(f"Raw content {raw_content_id} (bytes) for {memory_id} could not be UTF-8 decoded for text_content.")
                     return None # Or handle as non-textual memory if system supports it
            else: # Should not happen if RawContentStore returns Union[str, Dict, bytes, None]
                logger.warning(f"Retrieved content for {raw_content_id} (memory {memory_id}) is of unhandled type: {type(retrieved_package)}.")
                return None

            if text_content is None: # If content part was None in dict or other issue
                logger.error(f"Text content could not be extracted for memory_id '{memory_id}' from raw content {raw_content_id}.")
                return None

            return {
                "memory_id": memory_id,
                "text_content": text_content,
                "metadata": original_user_metadata or {}, # Ensure metadata is at least an empty dict
            }
        except Exception as e:
            logger.error(f"Error retrieving raw content {raw_content_id} for memory_id '{memory_id}': {e}", exc_info=True)
            return None

    async def schedule_archiving(self):
        """Placeholder for LTM archiving logic."""
        # This would interact with the raw_content_store and vector_db
        # to move data to colder storage or apply archival policies.
        logger.info("LTM Archiving process scheduled (stubbed - no actual client interaction).")
        await asyncio.sleep(0.01) # Simulate async operation

    async def schedule_cleaning(self):
        """Placeholder for LTM cleaning logic."""
        # This would interact with vector_db to remove stale/irrelevant entries,
        # and potentially self.raw_content_store to delete orphaned raw data.
        logger.info("LTM Cleaning process scheduled (stubbed - no actual client interaction).")
        await asyncio.sleep(0.01) # Simulate async operation


async def main():
    """Example usage of the LTM system with FAISSVectorDB and FileSystemRawContentStore."""
    import tempfile
    import shutil

    # Setup basic logging for the example
    if not logger.handlers:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- LTM Modular Integration Example Usage ---")

    # Create temporary directories for FAISS index and raw content
    temp_dir = tempfile.mkdtemp(prefix="ltm_example_")
    faiss_index_file = os.path.join(temp_dir, "ltm_faiss.index")
    raw_content_path = os.path.join(temp_dir, "raw_content")
    logger.info(f"Temporary directory for example: {temp_dir}")

    # 1. Create Configurations
    # EmbeddingConfig: specifies the model and target dimension (can be inferred by LTM if model is known)
    embed_conf = EmbeddingConfig(model_name='all-MiniLM-L6-v2', params={"dimension": 384})
    # VectorDBConfig: FAISS specific, pointing to the index file path
    vec_db_conf = VectorDBConfig(type="faiss", params={"index_path": faiss_index_file})
    # RawContentStoreConfig: FileSystem specific, pointing to a base storage path
    raw_store_conf = FileSystemConfig(base_path=raw_content_path)

    # Simplified RawCache (in-memory dict) for embedding caching
    class MockEmbeddingCache(RawCache): # Basic RawCache stub for example
        def __init__(self): self._cache = {}
        async def boot(self): logger.debug("MockEmbeddingCache booted."); return True
        async def shutdown(self): self._cache.clear(); logger.debug("MockEmbeddingCache shutdown."); return True
        async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
            return self._cache.get(f"{namespace}:{key}" if namespace else key)
        async def set(self, key: str, value: Any, namespace: Optional[str] = None, ttl_seconds: Optional[int] = None):
            self._cache[f"{namespace}:{key}" if namespace else key] = value
        async def delete(self, key: str, namespace: Optional[str] = None):
            self._cache.pop(f"{namespace}:{key}" if namespace else key, None)
        async def get_status(self): return {"type": "MockEmbeddingCache", "item_count": len(self._cache)}

    embedding_cache_instance = MockEmbeddingCache()

    # --- First Run: Instantiate components, boot LTM, and store data ---
    logger.info("--- LTM First Run: Setup and Storing data ---")

    # Instantiate concrete VectorDB and RawContentStore
    faiss_db_run1 = FAISSVectorDB(config=vec_db_conf, embedding_config=embed_conf)
    fs_raw_store_run1 = FileSystemRawContentStore(config=raw_store_conf)

    # Instantiate LTM with these components
    ltm_system_run1 = LTM(
        vector_db=faiss_db_run1,
        raw_content_store=fs_raw_store_run1,
        embedding_config=embed_conf,
        embedding_cache=embedding_cache_instance
    )

    if not await ltm_system_run1.boot(): # Boots LTM, which in turn boots its components
        logger.error("LTM Run 1 boot failed. Exiting example.")
        shutil.rmtree(temp_dir) # Clean up temp directory
        return

    # Variables to hold IDs and query embedding for cross-run verification
    mem_id1_run1, mem_id2_run1, mem_id3_run1, mem_id4_anon = None, None, None, None
    text4_anon_content = "An anonymous contribution to knowledge." # Store for assertion later
    query_embedding_run1 = None

    try:
        # Example data
        text1 = "The quick brown fox jumps over the lazy dog."
        meta1 = {"source": "test_doc_1", "chapter": 1, "tags": ["animal", "classic"]}
        emb1 = await ltm_system_run1.generate_embedding(text1) # Generate embedding via LTM

        text2 = "Artificial intelligence is transforming various global sectors."
        meta2 = {"source": "research_paper_001", "year": 2023, "tags": ["AI", "technology", "global"]}
        emb2 = await ltm_system_run1.generate_embedding(text2)

        text3 = "Context Kernels provide a novel framework for AI memory management."
        meta3 = {"source": "blog_post_abc", "author": "Dr. AI", "tags": ["AI", "memory", "ContextKernel"]}
        emb3 = await ltm_system_run1.generate_embedding(text3)

        # Store memories using LTM
        mem_id1_run1 = await ltm_system_run1.store_memory_chunk(chunk_id="fox_example", text_content=text1, embedding=emb1, metadata=meta1)
        mem_id2_run1 = await ltm_system_run1.store_memory_chunk(chunk_id="ai_transform_example", text_content=text2, embedding=emb2, metadata=meta2)
        mem_id3_run1 = await ltm_system_run1.store_memory_chunk(chunk_id="ck_def_example", text_content=text3, embedding=emb3, metadata=meta3)

        # Test with chunk_id = None (LTM should generate a UUID)
        meta4_anon = {"source": "anonymous", "tags": ["uuid_test"]}
        emb4_anon = await ltm_system_run1.generate_embedding(text4_anon_content)
        mem_id4_anon = await ltm_system_run1.store_memory_chunk(chunk_id=None, text_content=text4_anon_content, embedding=emb4_anon, metadata=meta4_anon)

        logger.info(f"Run 1: Stored memory IDs: {mem_id1_run1}, {mem_id2_run1}, {mem_id3_run1}, {mem_id4_anon}")
        assert "ltm_mem_" not in mem_id4_anon, f"Expected '{mem_id4_anon}' to be a raw UUID if chunk_id is None, not prefixed with 'ltm_mem_' by LTM's internal logic."

        # Check status of underlying FAISS DB via LTM's vector_db
        vdb_status_run1 = await ltm_system_run1.vector_db.get_status()
        assert vdb_status_run1["total_vectors"] == 4, f"Run 1: FAISS DB should have 4 vectors, got {vdb_status_run1['total_vectors']}"
        logger.info(f"Run 1: VectorDB status via LTM: {vdb_status_run1}")

        # Prepare a query embedding for retrieval tests
        query_text_run1 = "AI memory framework"
        query_embedding_run1 = await ltm_system_run1.generate_embedding(query_text_run1)

        await ltm_system_run1.shutdown() # This will save FAISS index and metadata via FAISSVectorDB.shutdown()
        logger.info("--- LTM First Run: Shutdown complete. Data should be persisted. ---")

    except Exception as e:
        logger.error(f"An error occurred during LTM Run 1: {e}", exc_info=True)
    finally:
        # Ensure shutdown is called even if errors occurred mid-way, if system was booted.
        if ltm_system_run1 and ltm_system_run1.embedding_model:
            await ltm_system_run1.shutdown()


    # --- Second Run: Load persisted data and verify ---
    logger.info("--- LTM Second Run: Loading persisted data ---")

    # Instantiate new set of components for Run 2, pointing to the same persisted paths
    faiss_db_run2 = FAISSVectorDB(config=vec_db_conf, embedding_config=embed_conf)
    fs_raw_store_run2 = FileSystemRawContentStore(config=raw_store_conf)

    ltm_system_run2 = LTM(
        vector_db=faiss_db_run2,
        raw_content_store=fs_raw_store_run2,
        embedding_config=embed_conf, # Same embedding config
        embedding_cache=embedding_cache_instance # Can reuse cache, or use a new one
    )

    if not await ltm_system_run2.boot(): # Boot again, this should load persisted data
        logger.error("LTM Run 2 boot failed. Exiting example.")
        shutil.rmtree(temp_dir)
        return

    try:
        # Verify VectorDB status after loading
        vdb_status_run2 = await ltm_system_run2.vector_db.get_status()
        assert vdb_status_run2["total_vectors"] == 4, f"Run 2: FAISS DB should have 4 vectors after loading, got {vdb_status_run2['total_vectors']}"
        logger.info(f"Run 2: FAISS DB loaded with {vdb_status_run2['total_vectors']} vectors. Status: {vdb_status_run2}")

        # Verify RawContentStore status (e.g., file count)
        rcs_status_run2 = await ltm_system_run2.raw_content_store.get_status()
        # We expect 4 raw content files corresponding to the 4 stored memories.
        assert rcs_status_run2["total_files"] == 4, f"Run 2: Raw Content Store should have 4 files, got {rcs_status_run2['total_files']}"
        logger.info(f"Run 2: Raw Content Store status: {rcs_status_run2}")

        # Verify retrieval of the anonymous chunk by its generated ID
        logger.info(f"Run 2: Retrieving memory by ID: {mem_id4_anon}")
        retrieved_mem4_anon_run2 = await ltm_system_run2.get_memory_by_id(mem_id4_anon)
        assert retrieved_mem4_anon_run2 is not None, f"Run 2: Failed to retrieve memory with ID {mem_id4_anon}"
        assert retrieved_mem4_anon_run2["text_content"] == text4_anon_content, "Run 2: Content mismatch for retrieved anonymous chunk"
        assert retrieved_mem4_anon_run2["metadata"]["source"] == "anonymous", "Run 2: Metadata mismatch for retrieved anonymous chunk"
        logger.info(f"Run 2: Successfully retrieved and verified memory {mem_id4_anon} by ID.")

        # Verify retrieval by ID for one of the named memories stored in Run 1
        logger.info(f"Run 2: Retrieving memory by ID: {mem_id1_run1}")
        retrieved_mem1_run2 = await ltm_system_run2.get_memory_by_id(mem_id1_run1)
        assert retrieved_mem1_run2 is not None, f"Run 2: Failed to retrieve memory {mem_id1_run1}"
        assert retrieved_mem1_run2["text_content"] == text1, "Run 2: Content mismatch for retrieved mem_id1" # text1 was defined in Run 1
        assert retrieved_mem1_run2["metadata"]["source"] == "test_doc_1", "Run 2: Metadata mismatch for retrieved mem_id1"
        logger.info(f"Run 2: Successfully retrieved and verified memory {mem_id1_run1} by ID.")

        # Verify search functionality still works and returns expected results
        logger.info(f"Run 2: Retrieving relevant memories for query: '{query_text_run1}'")
        # Use query_embedding_run1 generated in the first run as it's for the same text and model
        relevant_memories_run2 = await ltm_system_run2.retrieve_relevant_memories(query_embedding=query_embedding_run1, top_k=1)
        assert len(relevant_memories_run2) >= 1, "Run 2: Search failed to return results."
        # The top result should be mem_id3_run1 ("Context Kernels provide a novel framework...")
        retrieved_top_memory = relevant_memories_run2[0]
        assert retrieved_top_memory['memory_id'] == mem_id3_run1, \
            f"Run 2: Search returned incorrect top result. Expected {mem_id3_run1}, got {retrieved_top_memory['memory_id']}"
        logger.info(f"Run 2: Search successful. Top result: {retrieved_top_memory['memory_id']} with score {retrieved_top_memory['score']}")

        # Verify metadata for a specific memory ID from VectorDB perspective (via LTM's vector_db attribute)
        specific_vdb_meta = await ltm_system_run2.vector_db.get_metadata_by_id(mem_id2_run1)
        assert specific_vdb_meta is not None, f"Run 2: VectorDB metadata for {mem_id2_run1} not found."
        assert specific_vdb_meta["raw_content_id"] == f"raw_content_{mem_id2_run1}.json", \
            f"Run 2: VectorDB raw_content_id mismatch for {mem_id2_run1}"
        logger.info(f"Run 2: Verified specific VectorDB metadata for {mem_id2_run1}.")

        logger.info("--- LTM Second Run: All persistence tests passed ---")

    except Exception as e:
        logger.error(f"An error occurred during LTM Run 2 operations: {e}", exc_info=True)
    finally:
        if ltm_system_run2 and ltm_system_run2.embedding_model:
            await ltm_system_run2.shutdown()

        # Clean up the temporary directory created for the example
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")

    logger.info("--- LTM Modular Integration Example Usage Complete ---")


if __name__ == "__main__":
    # This example demonstrates the LTM system with FAISSVectorDB and FileSystemRawContentStore.
    # It requires `sentence-transformers` and `faiss-cpu` (or `faiss-gpu`) to be installed.
    # Example: pip install sentence-transformers faiss-cpu
    asyncio.run(main())
