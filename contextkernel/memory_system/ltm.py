import asyncio
import logging
import uuid
import json
import hashlib # For embedding cache keys
from typing import List, Dict, Optional, Any # For type hinting

# Configure basic logging
# Ensure logger is available if this module is used standalone for testing
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Commenting out basicConfig to avoid conflict if MemoryKernel also calls it.
# Module-level logger should be fine.
logger = logging.getLogger(__name__)

class LTM:
    """
    Long-Term Memory (LTM) system.
    Stores persistent knowledge and enables recall via vector search and metadata filtering.
    Currently stubbed, does not connect to real databases or services.
    """

    def __init__(self, vector_db_config: Optional[Dict]=None, raw_store_config: Optional[Dict]=None, cache_config: Optional[Dict]=None):
        logger.info("Initializing LongTermMemory (LTM) system (stubbed).")

        self._vector_db_stub: Dict[str, Dict[str, Any]] = {} # memory_id -> {"embedding": [], "metadata": {}, "raw_id": str}
        self._raw_store_stub: Dict[str, Dict[str, Any]] = {} # raw_id (e.g., s3_key) -> {"text_content": str, "metadata": {}}
        self._embedding_cache_stub: Dict[str, List[float]] = {} # text_hash_key -> embedding

        self.vector_db_config = vector_db_config or {"type": "stubbed_vector_db", "params": {}}
        self.raw_store_config = raw_store_config or {"type": "stubbed_raw_store", "params": {}}
        self.cache_config = cache_config or {"type": "stubbed_redis_cache", "params": {}}

        logger.info(f"LTM initialized with VectorDB: {self.vector_db_config}, RawStore: {self.raw_store_config}, Cache: {self.cache_config}")

    async def boot(self):
        """
        Simulates connecting to external services (Vector DB, Raw Store, Cache).
        """
        logger.info("LTM booting up... (simulating connections)")
        await asyncio.sleep(0.01)
        logger.info("LTM boot complete. Stubbed services are 'online'.")
        return True

    async def shutdown(self):
        """
        Simulates disconnecting from external services.
        """
        logger.info("LTM shutting down... (simulating disconnections)")
        await asyncio.sleep(0.01)
        logger.info("LTM shutdown complete.")
        return True

    def _get_text_key(self, text_content: str) -> str:
        return hashlib.md5(text_content.encode('utf-8')).hexdigest()

    async def _get_embedding_from_cache(self, text_key: str) -> Optional[List[float]]:
        logger.debug(f"Checking embedding cache for key: {text_key}")
        return self._embedding_cache_stub.get(text_key)

    async def _set_embedding_to_cache(self, text_key: str, embedding: List[float]):
        logger.debug(f"Storing embedding in cache for key: {text_key}")
        if len(self._embedding_cache_stub) > 1000:
            self._embedding_cache_stub.pop(next(iter(self._embedding_cache_stub)))
        self._embedding_cache_stub[text_key] = embedding

    async def generate_embedding(self, text_content: str) -> List[float]:
        """
        Generates or retrieves a cached embedding for the given text content.
        (Stubbed: Does not call a real embedding model).
        """
        if not text_content: # Handle empty string case
            logger.warning("Empty text_content for embedding generation. Returning default zero vector.")
            return [0.0] * 10 # Fixed size dummy embedding

        text_key = self._get_text_key(text_content)
        cached_embedding = await self._get_embedding_from_cache(text_key)
        if cached_embedding:
            logger.debug(f"Using cached embedding for text key: {text_key}")
            return cached_embedding

        logger.debug(f"Generating new embedding for text (stubbed): '{text_content[:50]}...'")
        await asyncio.sleep(0.02)
        embedding = [float(ord(c)) / 256.0 for c in text_content[:10]] # Simple char-based embedding
        embedding.extend([0.0] * (10 - len(embedding))) # Pad to fixed length 10

        await self._set_embedding_to_cache(text_key, embedding)
        return embedding

    async def store_memory_chunk(self, chunk_id: Optional[str], text_content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        if chunk_id is None:
            memory_id = str(uuid.uuid4())
            logger.info(f"No chunk_id provided, generated new memory_id: {memory_id}")
        else:
            # Use a prefix to avoid potential collision if chunk_id is also used elsewhere
            memory_id = f"ltm_mem_{chunk_id}"

        raw_content_id = f"raw_{memory_id}"

        logger.info(f"Storing memory chunk: ID='{memory_id}', RawID='{raw_content_id}'")

        self._raw_store_stub[raw_content_id] = {
            "text_content": text_content,
            "metadata": metadata,
            "ltm_memory_id": memory_id
        }
        logger.debug(f"Stored in raw_store_stub: {raw_content_id} -> '{text_content[:50]}...', Metadata: {metadata}")

        vector_db_metadata = metadata.copy()
        vector_db_metadata["raw_content_id"] = raw_content_id

        self._vector_db_stub[memory_id] = {
            "embedding": embedding,
            "metadata": vector_db_metadata,
            "raw_id": raw_content_id
        }
        logger.debug(f"Stored in vector_db_stub: {memory_id} -> Embedding (len {len(embedding)}), Metadata: {vector_db_metadata}")

        logger.info(f"Memory chunk '{memory_id}' stored successfully.")
        return memory_id

    async def retrieve_relevant_memories(self, query_embedding: List[float], top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving relevant memories: top_k={top_k}, filter={metadata_filter} (stubbed search)")

        candidate_ids = list(self._vector_db_stub.keys())
        results = []

        if metadata_filter:
            logger.debug(f"Applying metadata filter: {metadata_filter}")
            filtered_ids = []
            for mem_id in candidate_ids:
                item_metadata = self._vector_db_stub[mem_id]["metadata"]
                match = True
                for key, value in metadata_filter.items():
                    # Basic check: allows filtering for lists if value is in list metadata
                    meta_val = item_metadata.get(key)
                    if isinstance(meta_val, list) and value not in meta_val:
                        match = False
                        break
                    elif not isinstance(meta_val, list) and meta_val != value:
                        match = False
                        break
                if match:
                    filtered_ids.append(mem_id)
            candidate_ids = filtered_ids
            logger.debug(f"After metadata filtering, {len(candidate_ids)} candidates: {candidate_ids}")

        scored_candidates = []
        for mem_id in candidate_ids:
            item_embedding = self._vector_db_stub[mem_id]["embedding"]
            score = 0.0
            if len(item_embedding) == len(query_embedding) and len(item_embedding) > 0 :
                score = sum(x * y for x, y in zip(item_embedding, query_embedding)) / len(item_embedding) # Normalized dot product
            else:
                logger.warning(f"Dimension mismatch or zero length embedding for {mem_id}. Assigning zero score.")
            scored_candidates.append((mem_id, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        for i in range(min(top_k, len(scored_candidates))):
            mem_id, score = scored_candidates[i]
            vector_db_entry = self._vector_db_stub[mem_id]
            raw_content_id = vector_db_entry["raw_id"]

            raw_store_entry = self._raw_store_stub.get(raw_content_id)
            if raw_store_entry:
                results.append({
                    "memory_id": mem_id,
                    "text_content": raw_store_entry["text_content"],
                    "metadata": raw_store_entry["metadata"],
                    "score": score,
                    "retrieval_source": "LTM"
                })
            else:
                logger.warning(f"Could not find raw content for ID '{raw_content_id}' linked from memory '{mem_id}'.")

        logger.info(f"Retrieved {len(results)} relevant memories.")
        return results

    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Attempting to retrieve memory by ID: {memory_id}")
        vector_db_entry = self._vector_db_stub.get(memory_id)
        if not vector_db_entry:
            logger.warning(f"Memory ID '{memory_id}' not found in vector DB stub.")
            return None

        raw_content_id = vector_db_entry["raw_id"]
        raw_store_entry = self._raw_store_stub.get(raw_content_id)
        if not raw_store_entry:
            logger.error(f"CRITICAL: Raw content for ID '{raw_content_id}' (linked from memory '{memory_id}') not found! Data inconsistency in stub.")
            return None

        return {
            "memory_id": memory_id,
            "text_content": raw_store_entry["text_content"],
            "metadata": raw_store_entry["metadata"],
            "embedding_preview": vector_db_entry["embedding"][:5]
        }

    async def schedule_archiving(self):
        logger.info("LTM Archiving process scheduled (stubbed - no action taken).")
        await asyncio.sleep(0.01)

    async def schedule_cleaning(self):
        logger.info("LTM Cleaning process scheduled (stubbed - no action taken).")
        await asyncio.sleep(0.01)


async def main():
    # Setup basic logging for the example, if not already configured by an importer
    if not logger.handlers: # Check if logger already has handlers
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- LTM Example Usage ---")

    ltm_system = LTM()
    await ltm_system.boot()

    text1 = "The quick brown fox jumps over the lazy dog."
    meta1 = {"source": "test_doc_1", "chapter": 1, "type": "sentence", "tags": ["animal", "classic"]}
    emb1 = await ltm_system.generate_embedding(text1)

    text2 = "AIs are transforming the world across various domains."
    meta2 = {"source": "research_paper_001", "year": 2023, "type": "statement", "tags": ["AI", "technology"]}
    emb2 = await ltm_system.generate_embedding(text2)

    text3 = "Context Kernels are a new approach to AI memory."
    meta3 = {"source": "blog_post_abc", "author": "Dr. AI", "type": "definition", "tags": ["AI", "memory", "ContextKernel"]}
    emb3 = await ltm_system.generate_embedding(text3)

    mem_id1 = await ltm_system.store_memory_chunk(chunk_id="fox_dog", text_content=text1, embedding=emb1, metadata=meta1)
    mem_id2 = await ltm_system.store_memory_chunk(chunk_id="ai_transform", text_content=text2, embedding=emb2, metadata=meta2)
    mem_id3 = await ltm_system.store_memory_chunk(chunk_id="context_kernel_def", text_content=text3, embedding=emb3, metadata=meta3)

    logger.info(f"Stored memory IDs: {mem_id1}, {mem_id2}, {mem_id3}")

    retrieved_mem1 = await ltm_system.get_memory_by_id(mem_id1)
    logger.info(f"Retrieved memory {mem_id1}: {json.dumps(retrieved_mem1, indent=2)}")

    query_text = "Information about AI memory systems."
    query_emb = await ltm_system.generate_embedding(query_text)

    logger.info(f"\n--- Retrieving relevant memories for query: '{query_text}' ---")
    relevant_memories = await ltm_system.retrieve_relevant_memories(query_embedding=query_emb, top_k=2)
    for mem in relevant_memories:
        logger.info(f"Relevant memory: ID={mem['memory_id']}, Score={mem['score']:.4f}, Text='{mem['text_content'][:50]}...'")

    logger.info(f"\n--- Retrieving relevant memories with metadata filter (type='definition') ---")
    relevant_memories_filtered = await ltm_system.retrieve_relevant_memories(
        query_embedding=query_emb, top_k=2, metadata_filter={"type": "definition"}
    )
    for mem in relevant_memories_filtered:
        logger.info(f"Filtered relevant memory (type=definition): ID={mem['memory_id']}, Score={mem['score']:.4f}, Text='{mem['text_content'][:50]}...'")
        assert mem['metadata']['type'] == 'definition'

    logger.info(f"\n--- Retrieving relevant memories with metadata filter (tags='AI') ---")
    relevant_memories_tags_filtered = await ltm_system.retrieve_relevant_memories(
        query_embedding=query_emb, top_k=3, metadata_filter={"tags": "AI"} # Assumes 'AI' is in the list of tags
    )
    for mem in relevant_memories_tags_filtered:
        logger.info(f"Filtered relevant memory (tags=AI): ID={mem['memory_id']}, Score={mem['score']:.4f}, Text='{mem['text_content'][:50]}...'")
        assert "AI" in mem['metadata']['tags']

    mem_id_auto = await ltm_system.store_memory_chunk(chunk_id=None, text_content="Auto ID test for LTM", embedding=await ltm_system.generate_embedding("Auto ID test for LTM"), metadata={"source":"auto_id_test_ltm"})
    retrieved_auto_mem = await ltm_system.get_memory_by_id(mem_id_auto)
    logger.info(f"Retrieved auto-ID memory {mem_id_auto}: {json.dumps(retrieved_auto_mem, indent=2)}")

    await ltm_system.schedule_archiving()
    await ltm_system.schedule_cleaning()

    await ltm_system.shutdown()
    logger.info("--- LTM Example Usage Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
