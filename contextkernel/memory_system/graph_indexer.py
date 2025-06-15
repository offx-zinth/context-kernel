import asyncio
import logging
import json
import hashlib # For creating deterministic cache keys

import asyncio
import logging
import json
import hashlib # For creating deterministic cache keys
from typing import Any, Callable, List, Dict, Optional # Added Callable for embedding_generator

from .graph_db import GraphDB
from contextkernel.utils.config import NLPServiceConfig, EmbeddingConfig, Neo4jConfig, RedisConfig
from contextkernel.memory_system.raw_cache import RawCache
import spacy
from sentence_transformers import SentenceTransformer

# Ensure logger is available; basicConfig might be called by importing module or main.
# If run standalone, this basicConfig will apply.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphIndexer:
    """
    Ingests structured memory chunks, creates graph node+edge representations from them,
    and pushes them to GraphDB. Uses spaCy for NLP and SentenceTransformer for embeddings.
    """

    def __init__(self,
                 graph_db: GraphDB,
                 nlp_config: NLPServiceConfig, # Expects nlp_config.model to be a spaCy model name
                 embedding_config: EmbeddingConfig, # Expects embedding_config.model_name for SentenceTransformer
                 embedding_cache: RawCache):

        if not isinstance(graph_db, GraphDB):
            logger.error("GraphIndexer initialized with invalid GraphDB instance.")
            raise ValueError("GraphIndexer requires a valid GraphDB instance.")

        self.graph_db = graph_db
        self.nlp_config = nlp_config
        self.embedding_config = embedding_config
        self.embedding_cache = embedding_cache

        # Load NLP (spaCy) model
        # Ensure you have the spaCy model: python -m spacy download en_core_web_sm (or other model)
        try:
            self.nlp_processor = spacy.load(self.nlp_config.model or "en_core_web_sm") # Fallback to en_core_web_sm
            logger.info(f"spaCy model '{self.nlp_config.model or "en_core_web_sm"}' loaded successfully.")
        except OSError as e:
            logger.error(f"Could not load spaCy model '{self.nlp_config.model}'. "
                         f"Please download it (e.g., python -m spacy download {self.nlp_config.model}). Error: {e}")
            # Depending on policy, either raise error or allow continuation with NLP features disabled.
            # For now, let's raise to make it explicit.
            raise RuntimeError(f"Failed to load spaCy model: {self.nlp_config.model}") from e
            # self.nlp_processor = None # Or set to None and handle gracefully in methods

        # Load Embedding (SentenceTransformer) model
        try:
            model_name = self.embedding_config.model_name or 'all-MiniLM-L6-v2' # Fallback model
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer model '{model_name}' loaded. Dimension: {self.embedding_dimension}.")
        except Exception as e:
            logger.error(f"Could not load SentenceTransformer model '{self.embedding_config.model_name}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load SentenceTransformer model: {self.embedding_config.model_name}") from e
            # self.embedding_model = None
            # self.embedding_dimension = None


        logger.info(f"GraphIndexer initialized with GraphDB, spaCy model: '{self.nlp_processor.meta['name'] if self.nlp_processor else 'None'}', "
                    f"SentenceTransformer model: '{self.embedding_model.tokenizer.name_or_path if self.embedding_model else 'None'}', and Embedding Cache.")

    async def boot(self):
        """
        Ensures that the embedding cache is booted. Models are loaded in __init__.
        """
        logger.info("GraphIndexer booting...")
        if self.embedding_cache: # embedding_cache can be None if not provided or failed.
            if not await self.embedding_cache.boot():
                logger.warning("GraphIndexer: Embedding cache (RawCache) failed to boot or connect properly.")
        else:
            logger.warning("GraphIndexer: No embedding cache provided.")
        logger.info("GraphIndexer boot complete.")
        return True

    async def shutdown(self):
        """
        Shuts down the embedding cache. Models do not require explicit shutdown typically.
        """
        logger.info("GraphIndexer shutting down...")
        if self.embedding_cache:
            await self.embedding_cache.shutdown()
        # spaCy and SentenceTransformer models usually don't need explicit async shutdown.
        self.nlp_processor = None # Release models
        self.embedding_model = None
        return True

    async def _parse_chunk(self, chunk_id: str, chunk_data: str | dict) -> dict:
        """
        Simulates parsing the raw memory chunk.
        Input can be raw text or already somewhat structured data.
        """
        logger.debug(f"Parsing chunk ID: {chunk_id} (stubbed)")
        await asyncio.sleep(0.01) # Simulate parsing time
        if isinstance(chunk_data, str):
            return {"text_content": chunk_data, "source_format": "text", "original_chunk_id": chunk_id}
        elif isinstance(chunk_data, dict):
            # Ensure text_content is present for downstream tasks, even if it's a string representation
            if "text_content" not in chunk_data:
                chunk_data["text_content"] = json.dumps(chunk_data)
            return {**chunk_data, "source_format": "dict", "original_chunk_id": chunk_id}
        else:
            logger.warning(f"Unknown chunk data type for ID {chunk_id}: {type(chunk_data)}")
            return {"error": "Unknown data type", "original_data": str(chunk_data), "original_chunk_id": chunk_id}

    async def _extract_entities_topics_intents(self, chunk_id: str, parsed_data: dict) -> dict:
        """
        Extracts entities, topics, and intents using the loaded spaCy model.
        Topics are simplified to noun chunks. Intents are placeholder.
        """
        text_content = parsed_data.get("text_content", "")
        if not text_content or not self.nlp_processor:
            logger.warning(f"No text content or NLP processor not available for chunk ID: {chunk_id}. Skipping NLP extraction.")
            return {"entities": [], "topics": [], "intents": ["unknown"]}

        logger.debug(f"Extracting entities, topics, intents for chunk ID: {chunk_id} using spaCy model '{self.nlp_processor.meta['name']}'")

        loop = asyncio.get_running_loop()
        try:
            # Process text with spaCy model in a thread pool executor
            doc = await loop.run_in_executor(None, self.nlp_processor, text_content)
        except Exception as e:
            logger.error(f"Error processing text with spaCy for chunk {chunk_id}: {e}", exc_info=True)
            return {"entities": [], "topics": [], "intents": ["nlp_error"]}

        entities = [{"text": ent.text, "type": ent.label_, "span": (ent.start_char, ent.end_char)} for ent in doc.ents]

        # Simplified topics: using noun chunks
        topics = list(set(chunk.text for chunk in doc.noun_chunks))
        if not topics and len(doc) > 0: # Fallback for very short texts or no noun chunks
            topics = [token.lemma_ for token in doc if token.pos_ == "NOUN"][:5] # Max 5 nouns as topics

        # Simplified intents: placeholder logic
        intents = ["information_extraction"] # Default or very basic keyword matching
        if "question" in text_content.lower() or "?" in text_content:
            intents.append("question_answering_intent")

        logger.debug(f"spaCy extracted {len(entities)} entities, {len(topics)} topics for chunk {chunk_id}.")
        return {
            "entities": entities,
            "topics": topics,
            "intents": intents,
        }

    async def _resolve_coreferences(self, chunk_id: str, text_data: str, entities: list) -> list:
        """
        Placeholder for co-reference resolution.
        A dedicated coreference model/component (e.g., from spacy-experimental or a Hugging Face model)
        would be needed for a full implementation.
        """
        logger.debug(f"Co-reference resolution for chunk ID: {chunk_id} is currently a stub.")
        # In a real implementation, this would involve processing text_data and entities
        # with a coreference resolution model and updating entity mentions.
        return entities

    def _get_text_key(self, text_data: str) -> str:
        # Using model name in key to ensure different models don't have key collisions
        return hashlib.md5(f"{self.embedding_config.model_name}:{text_data}".encode('utf-8')).hexdigest()

    async def _get_cached_embedding(self, text_key: str) -> Optional[List[float]]:
        logger.debug(f"Attempting to get cached embedding from RawCache for key: {text_key}")
        if not self.embedding_cache: return None
        cached_value = await self.embedding_cache.get(key=text_key, namespace="graph_indexer_embeddings")
        if cached_value:
            if isinstance(cached_value, list):
                return cached_value
            try:
                loaded_list = json.loads(cached_value) # Assuming embeddings are stored as JSON strings
                if isinstance(loaded_list, list):
                    return loaded_list
                logger.warning(f"Decoded cached value for {text_key} is not a list: {type(loaded_list)}")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to decode cached embedding for key {text_key}. Value: '{str(cached_value)[:100]}...', Error: {e}")
        return None

    async def _set_cached_embedding(self, text_key: str, embedding: List[float]):
        logger.debug(f"Caching embedding to RawCache for key: {text_key}")
        if not self.embedding_cache: return
        await self.embedding_cache.set(key=text_key, value=embedding, namespace="graph_indexer_embeddings")

    async def _generate_embeddings(self, chunk_id: str, text_data: str) -> Optional[List[float]]:
        """
        Generates embeddings for text data using the loaded SentenceTransformer model. Uses caching.
        """
        if not self.embedding_model or not self.embedding_dimension:
            logger.error("SentenceTransformer model or dimension not available for embedding generation.")
            return None

        if not text_data:
            logger.warning(f"Empty text_data for embedding generation for chunk ID: {chunk_id}")
            return [0.0] * self.embedding_dimension

        text_key = self._get_text_key(text_data) # Includes model name now

        try:
            cached_embedding = await self._get_cached_embedding(text_key)
            if cached_embedding:
                logger.debug(f"Found cached embedding for chunk ID: {chunk_id} (key: {text_key})")
                return cached_embedding
        except Exception as e:
            logger.warning(f"Error retrieving embedding from cache for key {text_key}: {e}. Will regenerate.")

        logger.debug(f"Generating new embedding for chunk ID: {chunk_id} using '{self.embedding_model._first_module().tokenizer.name_or_path}' for text: '{text_data[:50]}...'")

        loop = asyncio.get_running_loop()
        try:
            # SentenceTransformer's encode is CPU-bound; run in executor.
            embedding_array = await loop.run_in_executor(
                None, # Uses default ThreadPoolExecutor
                self.embedding_model.encode,
                text_data,
                # convert_to_tensor=False # Already defaults to numpy array
            )
            embedding = embedding_array.tolist() # Convert numpy array to list
            if not isinstance(embedding, list) or (embedding and not isinstance(embedding[0], float)):
                 logger.warning(f"Generated embedding for {chunk_id} is not a list of floats. Type: {type(embedding)}. Attempting conversion.")
                 embedding = [float(x) for x in embedding]

        except Exception as e:
            logger.error(f"Error generating embedding for chunk {chunk_id}: {e}", exc_info=True)
            return [0.0] * self.embedding_dimension # Return zero vector on error

        try:
            await self._set_cached_embedding(text_key, embedding)
        except Exception as e:
            logger.warning(f"Error storing embedding to cache for key {text_key}: {e}.")

        return embedding

    async def _construct_graph_elements(
        self,
        chunk_id: str,
        parsed_data: dict,
        nlp_extractions: dict,
        embedding: list[float] | None
    ) -> tuple[list[dict], list[dict]]:
        logger.debug(f"Constructing graph elements for chunk ID: {chunk_id}")
        nodes_to_create = []
        edges_to_create = []

        chunk_node_id = f"chunk_{chunk_id}"
        chunk_properties = {
            "type": "MemoryChunk",
            "source_id": parsed_data.get("original_chunk_id", chunk_id),
            "text_content_snippet": parsed_data.get("text_content", "")[:250], # Store a snippet
        }
        if embedding:
            chunk_properties["embedding_vector_stub"] = embedding
        if parsed_data.get("source_format") == "dict":
            chunk_properties["original_structure"] = {k:v for k,v in parsed_data.items() if k not in ["text_content", "embedding_vector_stub"]}


        nodes_to_create.append({"id": chunk_node_id, "properties": chunk_properties, "labels": ["MemoryChunk"]})

        for entity in nlp_extractions.get("entities", []):
            entity_text_clean = entity['text'].replace(' ', '_').replace('.', '').lower()
            entity_node_id = f"entity_{entity.get('type', 'Generic').lower()}_{entity_text_clean}"
            nodes_to_create.append({
                "id": entity_node_id,
                "properties": {"name": entity["text"], "entity_type": entity["type"]},
                "labels": ["Entity", entity["type"].capitalize()]
            })
            edges_to_create.append({
                "source": chunk_node_id,
                "target": entity_node_id,
                "type": "MENTIONS_ENTITY",
                "properties": {"span": entity.get("span", (0,0))}
            })

        for topic in nlp_extractions.get("topics", []):
            topic_node_id = f"topic_{topic.replace(' ', '_').lower()}"
            nodes_to_create.append({
                "id": topic_node_id,
                "properties": {"name": topic},
                "labels": ["Topic"]
            })
            edges_to_create.append({"source": chunk_node_id, "target": topic_node_id, "type": "HAS_TOPIC"})

        for intent_text in nlp_extractions.get("intents", []):
            intent_node_id = f"intent_{intent_text.replace(' ', '_').lower()}"
            nodes_to_create.append({
                "id": intent_node_id,
                "properties": {"name": intent_text},
                "labels": ["Intent"]
            })
            edges_to_create.append({"source": chunk_node_id, "target": intent_node_id, "type": "HAS_INTENT"})

        logger.debug(f"Constructed {len(nodes_to_create)} nodes and {len(edges_to_create)} edges for chunk ID: {chunk_id}")
        return nodes_to_create, edges_to_create

    async def process_memory_chunk(self, chunk_id: str, chunk_data: str | dict) -> dict:
        logger.info(f"--- Starting processing for memory chunk ID: {chunk_id} ---")

        parsed_data = await self._parse_chunk(chunk_id, chunk_data)
        if "error" in parsed_data:
            logger.error(f"Failed to parse chunk {chunk_id}: {parsed_data['error']}")
            return {"status": "failed", "reason": "parsing_error", "details": parsed_data['error'], "chunk_id": chunk_id}

        text_content_for_nlp = parsed_data.get("text_content", "")
        if not text_content_for_nlp and isinstance(parsed_data, dict): # Fallback for dicts without text_content
             text_content_for_nlp = json.dumps(parsed_data)


        nlp_extractions = await self._extract_entities_topics_intents(chunk_id, parsed_data)

        entities_after_coref = await self._resolve_coreferences(chunk_id, text_content_for_nlp, nlp_extractions["entities"])
        nlp_extractions["entities"] = entities_after_coref

        embedding = await self._generate_embeddings(chunk_id, text_content_for_nlp)

        nodes_to_create, edges_to_create = await self._construct_graph_elements(
            chunk_id, parsed_data, nlp_extractions, embedding
        )

        nodes_created_count = 0
        edges_created_count = 0
        nodes_existed_count = 0
        edges_existed_count = 0

        unique_nodes_to_create = {node['id']: node for node in nodes_to_create}.values()


        for node_data in unique_nodes_to_create:
            existing_node = await self.graph_db.get_node(node_data["id"])
            if not existing_node:
                success = await self.graph_db.create_node(
                    node_id=node_data["id"],
                    properties=node_data["properties"],
                    labels=node_data["labels"]
                )
                if success:
                    nodes_created_count += 1
            else:
                nodes_existed_count +=1
                # Optionally update if needed:
                # await self.graph_db.update_node(node_data["id"], node_data["properties"])


        unique_edges_to_create = {}
        for edge in edges_to_create:
            key = (edge["source"], edge["target"], edge["type"])
            if key not in unique_edges_to_create:
                 unique_edges_to_create[key] = edge

        for edge_data in unique_edges_to_create.values():
            existing_edge = await self.graph_db.get_edge(
                edge_data["source"], edge_data["target"], edge_data["type"]
            )
            if not existing_edge:
                success = await self.graph_db.create_edge(
                    source_node_id=edge_data["source"],
                    target_node_id=edge_data["target"],
                    relationship_type=edge_data["type"],
                    properties=edge_data.get("properties")
                )
                if success:
                    edges_created_count += 1
            else:
                edges_existed_count +=1
                # Optionally update if needed:
                # await self.graph_db.update_edge_properties(edge_data["source"], edge_data["target"], edge_data["type"], edge_data.get("properties", {}))


        summary = {
            "status": "success",
            "chunk_id": chunk_id,
            "nodes_processed": len(unique_nodes_to_create),
            "edges_processed": len(list(unique_edges_to_create.values())),
            "nodes_created_in_db": nodes_created_count,
            "nodes_already_existed_in_db": nodes_existed_count,
            "edges_created_in_db": edges_created_count,
            "edges_already_existed_in_db": edges_existed_count,
            "nlp_summary": {
                "entities_found": len(nlp_extractions["entities"]),
                "topics_identified": nlp_extractions["topics"],
                "intents_derived": nlp_extractions["intents"]
            }
        }
        logger.info(f"--- Finished processing memory chunk ID: {chunk_id}. Summary: {json.dumps(summary, indent=2)} ---")
        return summary

async def main():
    # Default basicConfig is already at the top of the file for module-level logging
    # If running this main, ensure it's set.
    # Ensure basicConfig is called for the logger to output messages.
    # This is especially important if running the file directly.
    # Set logging level to DEBUG for more detailed output if needed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- GraphIndexer Example Usage (with Real Local Models) ---")
    logger.warning("This example uses real spaCy and SentenceTransformer models.")
    logger.warning("Ensure you have downloaded the necessary models:")
    logger.warning("  `python -m spacy download en_core_web_sm` (or other specified spaCy model)")
    logger.warning("  SentenceTransformer model will be downloaded automatically on first use if not cached.")
    logger.warning("A Redis instance is recommended for RawCache, otherwise it will use a mock.")


    # 1. Configurations
    # Using a mock GraphDB as it's not the focus of this specific component test and requires a live Neo4j.
    neo4j_conf = Neo4jConfig(uri="bolt://mockhost:7687", user="mockuser", password="mockpassword")

    # NLPServiceConfig: provider can be 'spacy', model is the spaCy model name.
    nlp_conf = NLPServiceConfig(provider="spacy", model="en_core_web_sm")

    # EmbeddingConfig: model_name is for SentenceTransformer.
    embed_conf = EmbeddingConfig(model_name='all-MiniLM-L6-v2') # A common, small SentenceTransformer model (384 dims)

    redis_conf_for_cache = RedisConfig(db=2) # Use a different DB for GraphIndexer's cache

    # 2. Mock GraphDB (as it's an external dependency not being tested here)
    from neo4j import AsyncGraphDatabase, AsyncDriver # Required for GraphDB type hint
    class MockNeo4jDriver:
        async def verify_connectivity(self): logger.info("MockNeo4jDriver: verify_connectivity called"); return True
        async def close(self): logger.info("MockNeo4jDriver: close called")
        # Add other methods that GraphDB might call if its internal logic changes.
        # For now, GraphDB's productionized methods will form queries but execute against this mock.
        # The `_execute_query_neo4j` in GraphDB would need to be adapted if we want it to return mock data.
        # For GraphIndexer test, we mostly care that calls to graph_db.create_node etc. don't fail.

    class MockGraphDB(GraphDB):
        def __init__(self, config, driver):
            super().__init__(config, driver)
            self._mock_nodes = {}
            self._mock_edges = {}
            logger.info("MockGraphDB initialized for GraphIndexer test.")

        async def _execute_query_neo4j(self, query: str, parameters: Optional[Dict] = None, write: bool = False) -> List[Dict[str, Any]]:
            logger.info(f"MockGraphDB received query: {query} with params: {parameters} (write={write})")
            # Simulate behavior for specific queries if needed for the test, e.g., node existence checks.
            if "MERGE (n {" in query and "node_id:" in query and write: # Node creation
                node_id = parameters.get("node_id")
                if node_id:
                    self._mock_nodes[node_id] = parameters.get("props", {})
                    return [{"created_node_id": node_id}] # Simulate return value
            elif "MATCH (n {" in query and "node_id:" in query and not write: # Node get
                 node_id = parameters.get("node_id")
                 if node_id in self._mock_nodes:
                     return [{"n": self._mock_nodes[node_id]}] # Simulate return value
            elif "MATCH (a {" in query and "MERGE (a)-[r:" in query and write: # Edge creation
                # Minimal simulation for edge creation
                return [{"relationship_type": parameters.get("relationship_type", "MOCKED_REL")}]

            return [] # Default empty response for other queries

        async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
            logger.debug(f"MockGraphDB.get_node called for {node_id}")
            return self._mock_nodes.get(node_id)

        async def create_node(self, node_id: str, properties: dict, labels: Optional[List[str]] = None) -> bool:
            logger.debug(f"MockGraphDB.create_node called for {node_id}")
            self._mock_nodes[node_id] = {"node_id":node_id, **properties, "_labels": labels}
            return True

        async def get_edge(self, source_node_id: str, target_node_id: str, relationship_type: str) -> Optional[Dict[str, Any]]:
            logger.debug(f"MockGraphDB.get_edge called for {source_node_id}-{relationship_type}->{target_node_id}")
            return self._mock_edges.get((source_node_id, target_node_id, relationship_type))

        async def create_edge(self, source_node_id: str, target_node_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
            logger.debug(f"MockGraphDB.create_edge called for {source_node_id}-{relationship_type}->{target_node_id}")
            self._mock_edges[(source_node_id, target_node_id, relationship_type)] = properties or {}
            return True


    mock_driver = MockNeo4jDriver()
    # We pass the real Neo4jConfig, but the MockGraphDB uses a MockNeo4jDriver which bypasses actual connection.
    graph_db_instance = MockGraphDB(config=neo4j_conf, driver=mock_driver)
    await graph_db_instance.boot()


    # RawCache for Embeddings (using real RawCache, potentially with a mock Redis client if Redis is down)
    from redis.asyncio import Redis as RedisClient
    redis_client_for_indexer_cache = None
    try:
        redis_client_for_indexer_cache = RedisClient(
            host=redis_conf_for_cache.host, port=redis_conf_for_cache.port,
            db=redis_conf_for_cache.db, password=redis_conf_for_cache.password,
            socket_timeout=1, socket_connect_timeout=1 # Short timeouts for example
        )
        await redis_client_for_indexer_cache.ping()
        logger.info("Successfully connected to Redis for GraphIndexer's embedding cache.")
    except Exception as e:
        logger.warning(f"Could not connect to Redis for GraphIndexer's cache: {e}. Embedding caching will use RawCache's in-memory stub if its client is a mock.")
        class MockRedisClient: # Minimal mock for RawCache
             async def ping(self): return True
             async def set(self, name, value, ex=None, px=None, nx=False, xx=False): pass
             async def get(self, name): return None
             async def delete(self, *names): pass; return 0
             async def exists(self, *names): return 0
             async def expire(self, name, time): return False
             async def pttl(self, name): return -2
             async def close(self): pass
        redis_client_for_indexer_cache = MockRedisClient() # type: ignore

    embedding_cache_instance = RawCache(config=redis_conf_for_cache, client=redis_client_for_indexer_cache)
    await embedding_cache_instance.boot()

    # 3. Instantiate GraphIndexer
    # It will load the actual spaCy and SentenceTransformer models specified in nlp_conf and embed_conf.
    try:
        indexer = GraphIndexer(
            graph_db=graph_db_instance,
            nlp_config=nlp_conf,
            embedding_config=embed_conf,
            embedding_cache=embedding_cache_instance
        )
        await indexer.boot()
    except RuntimeError as e: # Catch model loading errors specifically if they are raised from __init__
        logger.error(f"Failed to initialize GraphIndexer due to model loading error: {e}", exc_info=True)
        logger.error("Ensure you have downloaded the spaCy model (e.g., python -m spacy download en_core_web_sm) "
                     "and that SentenceTransformer can access its model.")
        if redis_client_for_indexer_cache and hasattr(redis_client_for_indexer_cache, 'close'):
            await redis_client_for_indexer_cache.close()
        if hasattr(mock_driver, 'close'):
            await mock_driver.close()
        return # Exit if models can't be loaded

    # Test with some data
    chunk1_id = "mem_chunk_real_nlp_001"
    chunk1_data = "Alice works at Wonderland Inc. Bob is her manager. They are working on the Context Kernel project which involves AI."

    # Check embedding cache after processing
    # Note: _generate_embeddings is called inside process_memory_chunk
    # We can check the cache for an artifact of this call.
    text_key_chunk1 = indexer._get_text_key(chunk1_data) # Get the expected key

    # Process chunk1
    result1 = await indexer.process_memory_chunk(chunk1_id, chunk1_data)
    # logger.info(f"Result for {chunk1_id}: {json.dumps(result1, indent=2)}") # Logged by method

    cached_emb_chunk1 = await embedding_cache_instance.get(text_key_chunk1, namespace="graph_indexer_embeddings")
    logger.info(f"Embedding for chunk1_data {'FOUND' if cached_emb_chunk1 else 'NOT FOUND'} in RawCache after processing.")
    assert cached_emb_chunk1 is not None # Verify it was cached by _generate_embeddings

    chunk2_id = "mem_chunk_002"
    chunk2_data_dict = { # Renamed to avoid conflict with internal var name
        "document_id": "doc_xyz",
        "page_number": 5,
        "text_content": "The Context Kernel aims to improve memory systems in AI. Alice is a key contributor to the AI field.",
        "author": "SystemReport"
    }

    chunk3_id = "mem_chunk_003"
    chunk3_data = "" # Test with empty string

    result2 = await indexer.process_memory_chunk(chunk2_id, chunk2_data_dict)
    result3 = await indexer.process_memory_chunk(chunk3_id, chunk3_data)

    logger.info(f"--- Reprocessing {chunk1_id} to test idempotency and caching ---")
    # This call to _generate_embeddings should hit the cache for chunk1_data
    result1_reprocessed = await indexer.process_memory_chunk(chunk1_id, chunk1_data)


    logger.info("--- Inspecting GraphDB (sample, via graph_db_instance) ---")
    # GraphDB calls are stubbed, so these will reflect the in-memory stub state of GraphDB
    if not isinstance(neo4j_driver, MockNeo4jDriver): # Only if we have a real connection attempt
        alice_node = await graph_db_instance.get_node("entity_person_alice")
        logger.info(f"Alice node from GraphDB: {json.dumps(alice_node, indent=2)}")

        ck_node = await graph_db_instance.get_node("entity_project_context_kernel")
        logger.info(f"Context Kernel node from GraphDB: {json.dumps(ck_node, indent=2)}")

        chunk1_node_graph = await graph_db_instance.get_node("chunk_mem_chunk_001") # Changed var name
        logger.info(f"Chunk1 node properties from GraphDB: {json.dumps(chunk1_node_graph.get('properties') if chunk1_node_graph else None, indent=2)}")

        edge_alice_mentioned = await graph_db_instance.get_edge("chunk_mem_chunk_001", "entity_person_alice", "MENTIONS_ENTITY")
        logger.info(f"Chunk1 MENTIONS_ENTITY Alice from GraphDB: {json.dumps(edge_alice_mentioned, indent=2)}")

    # Shutdown components
    await indexer.shutdown()
    await embedding_cache_instance.shutdown()
    await graph_db_instance.shutdown() # GraphDB shutdown

    # Close client connections
    if hasattr(redis_client_for_indexer_cache, 'close'):
         await redis_client_for_indexer_cache.close()
    if hasattr(neo4j_driver, 'close'):
        await neo4j_driver.close()

    logger.info("--- GraphIndexer Example Usage (with Injected Dependencies) Complete ---")

if __name__ == "__main__":
    # Logging is configured at the top of the file.
    # This example requires Neo4j and Redis servers running for full functionality,
    # otherwise, it falls back to mocks for Neo4j driver and RawCache's internal stub for Redis.
    asyncio.run(main())
