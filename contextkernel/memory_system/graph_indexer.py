import asyncio
import logging
import json
import hashlib # For creating deterministic cache keys

# Assuming GraphDB is in the same package directory
from .graph_db import GraphDB

# Configure basic logging
# logger = logging.getLogger(__name__) # Already configured at higher level or in main
# Ensure logger is available if this module is used standalone for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphIndexer:
    """
    Ingests structured memory chunks, creates graph node+edge representations from them,
    and pushes them to GraphDB.
    """

    def __init__(self, graph_db_instance: GraphDB):
        if not isinstance(graph_db_instance, GraphDB):
            logger.error("GraphIndexer initialized with invalid GraphDB instance.")
            raise ValueError("GraphIndexer requires a valid GraphDB instance.")
        self.graph_db = graph_db_instance
        self._embedding_cache = {} # Simple dict for embedding caching stub
        logger.info("GraphIndexer initialized with GraphDB instance.")

    async def boot(self):
        """
        Async method to perform any setup for the GraphIndexer.
        """
        logger.info("GraphIndexer booted.")
        # In a real scenario, this might load models or connect to NLP services.
        await asyncio.sleep(0.01) # Simulate setup time
        return True

    async def shutdown(self):
        """
        Async method to perform any cleanup for the GraphIndexer.
        """
        logger.info("GraphIndexer shutting down.")
        # In a real scenario, this might release resources.
        await asyncio.sleep(0.01) # Simulate cleanup time
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
        Simulates NLP/LLM calls to extract entities, topics, and intents.
        """
        logger.debug(f"Extracting entities, topics, intents for chunk ID: {chunk_id} (stubbed NLP/LLM call)")
        await asyncio.sleep(0.02) # Simulate API call or model inference time

        text_content = parsed_data.get("text_content", "")

        entities = []
        if "Alice" in text_content:
            entities.append({"text": "Alice", "type": "PERSON", "span": (text_content.find("Alice"), text_content.find("Alice")+len("Alice"))})
        if "Bob" in text_content:
            entities.append({"text": "Bob", "type": "PERSON", "span": (text_content.find("Bob"),text_content.find("Bob")+len("Bob"))})
        if "Context Kernel" in text_content:
            entities.append({"text": "Context Kernel", "type": "PROJECT", "span": (text_content.find("Context Kernel"), text_content.find("Context Kernel")+len("Context Kernel"))})
        if "Wonderland Inc." in text_content:
             entities.append({"text": "Wonderland Inc.", "type": "ORGANIZATION", "span": (text_content.find("Wonderland Inc."), text_content.find("Wonderland Inc.")+len("Wonderland Inc."))})


        topics = ["AI", "Memory Systems"] if "AI" in text_content or "Kernel" in text_content else ["General"]
        intents = ["Information Sharing"]

        return {
            "entities": entities,
            "topics": topics,
            "intents": intents,
        }

    async def _resolve_coreferences(self, chunk_id: str, text_data: str, entities: list) -> list:
        """
        Simulates co-reference resolution. (Stubbed: no actual change)
        """
        logger.debug(f"Resolving coreferences for chunk ID: {chunk_id} (stubbed)")
        await asyncio.sleep(0.01)
        return entities

    def _get_text_key(self, text_data: str) -> str:
        return hashlib.md5(text_data.encode('utf-8')).hexdigest()

    async def _get_cached_embedding(self, text_key: str) -> list[float] | None:
        logger.debug(f"Attempting to get cached embedding for key: {text_key}")
        return self._embedding_cache.get(text_key)

    async def _set_cached_embedding(self, text_key: str, embedding: list[float]):
        logger.debug(f"Caching embedding for key: {text_key}")
        if len(self._embedding_cache) > 1000:
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        self._embedding_cache[text_key] = embedding

    async def _generate_embeddings(self, chunk_id: str, text_data: str) -> list[float] | None:
        """
        Simulates generating embeddings for text data. Uses caching.
        """
        logger.debug(f"Generating embeddings for chunk ID: {chunk_id} (stubbed)")
        if not text_data: # Handle empty string case
            logger.warning(f"Empty text_data for embedding generation for chunk ID: {chunk_id}")
            return [0.0] * 10 # Return a default zero vector or similar

        text_key = self._get_text_key(text_data)
        cached_embedding = await self._get_cached_embedding(text_key)
        if cached_embedding:
            logger.debug(f"Found cached embedding for chunk ID: {chunk_id}")
            return cached_embedding

        await asyncio.sleep(0.03) # Simulate embedding generation time
        embedding = [float(ord(c)) / 256.0 for c in text_data[:10]] # Simple char-based embedding for stub
        embedding.extend([0.0] * (10 - len(embedding))) # Pad to fixed length 10

        await self._set_cached_embedding(text_key, embedding)
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
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- GraphIndexer Example Usage ---")

    graph_db = GraphDB(neo4j_uri="bolt://stubhost:7687")
    await graph_db.boot()

    indexer = GraphIndexer(graph_db_instance=graph_db)
    await indexer.boot()

    chunk1_id = "mem_chunk_001"
    chunk1_data = "Alice works at Wonderland Inc. Bob is her manager. They are working on the Context Kernel project which involves AI."

    chunk2_id = "mem_chunk_002"
    chunk2_data = {
        "document_id": "doc_xyz",
        "page_number": 5,
        "text_content": "The Context Kernel aims to improve memory systems in AI. Alice is a key contributor to the AI field.",
        "author": "SystemReport"
    }

    chunk3_id = "mem_chunk_003" # Test with empty string
    chunk3_data = ""


    result1 = await indexer.process_memory_chunk(chunk1_id, chunk1_data)
    # logger.info(f"Result for {chunk1_id}: {json.dumps(result1, indent=2)}") # Already logged in process_memory_chunk

    result2 = await indexer.process_memory_chunk(chunk2_id, chunk2_data)
    # logger.info(f"Result for {chunk2_id}: {json.dumps(result2, indent=2)}")

    result3 = await indexer.process_memory_chunk(chunk3_id, chunk3_data)


    logger.info(f"--- Reprocessing {chunk1_id} to test idempotency ---")
    result1_reprocessed = await indexer.process_memory_chunk(chunk1_id, chunk1_data)
    # logger.info(f"Result for reprocessed {chunk1_id}: {json.dumps(result1_reprocessed, indent=2)}")

    logger.info("--- Inspecting GraphDB (sample) ---")
    alice_node = await graph_db.get_node("entity_person_alice")
    logger.info(f"Alice node: {json.dumps(alice_node, indent=2)}")

    ck_node = await graph_db.get_node("entity_project_context_kernel")
    logger.info(f"Context Kernel node: {json.dumps(ck_node, indent=2)}")

    chunk1_node = await graph_db.get_node("chunk_mem_chunk_001")
    logger.info(f"Chunk1 node properties: {json.dumps(chunk1_node.get('properties') if chunk1_node else None, indent=2)}")

    edge_alice_mentioned = await graph_db.get_edge("chunk_mem_chunk_001", "entity_person_alice", "MENTIONS_ENTITY")
    logger.info(f"Chunk1 MENTIONS_ENTITY Alice: {json.dumps(edge_alice_mentioned, indent=2)}")

    await indexer.shutdown()
    await graph_db.shutdown()

    logger.info("--- GraphIndexer Example Usage Complete ---")

if __name__ == "__main__":
    # It's good practice to have basicConfig here if the module can be run directly
    # and might not have had logging configured by an importing module.
    # However, since it's at the top, this is mostly for clarity or if top one is removed.
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
