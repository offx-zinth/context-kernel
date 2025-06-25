import asyncio
import logging
from typing import List, Dict, Any, Optional # Added for type hints

# Assuming StructuredInsight and related types are accessible
# If they are in llm_listener.py, adjust the import path accordingly.
# For now, let's assume they might be moved to a common types module or defined here if small.
# Placeholder imports, adjust based on actual location of these types:
from contextkernel.core_logic.llm_listener import StructuredInsight, Summary, Entity, Relation
from contextkernel.core_logic.exceptions import MemoryAccessError

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, graph_db: Any, ltm: Any, stm: Any, raw_cache: Any): # Added type hints for components
        self.graph_db = graph_db
        self.ltm = ltm
        self.stm = stm
        self.raw_cache = raw_cache
        logger.info("MemoryManager initialized with DB instances.")

    async def store(self, structured_data: StructuredInsight) -> None:
        logger.info(f"Persisting structured insights (Raw ID: {structured_data.raw_data_id or 'N/A'})")
        doc_id_base = structured_data.raw_data_id or f"doc_{structured_data.created_at.isoformat()}"

        memory_ops = []

        # 1. Save raw content to RawCache (This step was part of _generate_insights in LLMListener,
        # but the problem description implies MemoryManager's store method should handle it)
        # Assuming structured_data.source_data_preview holds the raw content or a reference to it.
        # If raw content is directly available in structured_data, use it.
        # For now, let's assume raw_data_id implies it's already in RawCache by the time store is called.
        # If MemoryManager is also responsible for putting the *initial* raw data into RawCache,
        # this method would need the raw data itself, not just StructuredInsight.
        # The prompt says: "Saving the raw content to RawCache." - this implies this method should do it.
        # Let's assume `structured_data.source_data_preview` might be the raw content for this example.
        # A field like `raw_content` in StructuredInsight would be clearer.

        # Re-interpreting: llm_listener.py's `process_data` calls `rc.store` for raw data *before* `_write_to_memory`.
        # So, `_write_to_memory` (and thus this `store` method) doesn't save the *initial* raw content.
        # It *does* link the RawCacheEntry in the graph.

        # 2. Save summary to STM
        if structured_data.summary and self.stm:
            logger.debug(f"Queueing STM save for summary of {doc_id_base}")
            stm_summary_id = f"{doc_id_base}_summary"
            # Ensure summary_obj is what STM expects, e.g., the Summary object or its text
            summary_object_to_store = structured_data.summary # Storing the Summary Pydantic model
            if hasattr(structured_data.summary, 'text'): # Or just the text if STM expects that
                 summary_object_to_store = structured_data.summary.text

            memory_ops.append(self.stm.save_summary( # Assuming save_summary is an async method
                summary_id=stm_summary_id,
                summary_obj=summary_object_to_store, # Use the Pydantic model directly if STM handles it
                metadata={
                    "doc_id_base": doc_id_base,
                    "raw_data_id": structured_data.raw_data_id,
                    "source_preview": structured_data.source_data_preview,
                    "type": "summary"
                }
            ))

        # 3. Save document/embedding to LTM
        # Determine what content to save in LTM. Original logic used summary text if available.
        ltm_content_to_store = structured_data.source_data_preview # Default to preview
        if structured_data.summary and structured_data.summary.text:
            ltm_content_to_store = structured_data.summary.text

        if self.ltm and ltm_content_to_store:
            logger.debug(f"Queueing LTM save for document {doc_id_base}")
            ltm_doc_id = f"{doc_id_base}_ltm_doc"
            memory_ops.append(self.ltm.save_document( # Assuming save_document is an async method
                doc_id=ltm_doc_id,
                text_content=ltm_content_to_store,
                embedding=structured_data.content_embedding or [], # Ensure embedding is not None
                metadata={
                    "doc_id_base": doc_id_base,
                    "raw_data_id": structured_data.raw_data_id,
                    "original_data_type": structured_data.original_data_type,
                    "type": "document_content"
                }
            ))

        # 4. Enrich GraphDB
        if self.graph_db:
            logger.info(f"Starting graph enrichment for SourceDocument ID: {doc_id_base}")

            source_doc_props = {
                "raw_data_id": structured_data.raw_data_id,
                "preview": structured_data.source_data_preview,
                "original_data_type": structured_data.original_data_type,
                "created_at": structured_data.created_at.isoformat(),
                "updated_at": structured_data.updated_at.isoformat()
            }

            try:
                # Ensure SourceDocument node is created first.
                # This needs to be awaited if subsequent operations depend on its immediate existence
                # or if graph_db methods are not internally queuing but executing directly.
                # The original code awaits ensure_source_document_node.
                await self.graph_db.ensure_source_document_node(document_id=doc_id_base, properties=source_doc_props)
                logger.info(f"SourceDocument node ensured for ID: {doc_id_base}")

                # Link STM entry if summary exists
                if structured_data.summary and self.stm:
                    stm_fragment_id = f"{doc_id_base}_summary"
                    stm_node_props = {
                        "id": stm_fragment_id,
                        "text": structured_data.summary.text,
                        "source_document_id": doc_id_base,
                        "type": "summary",
                        "created_at": structured_data.summary.created_at.isoformat()
                    }
                    memory_ops.append(self.graph_db.add_memory_fragment_link(
                        document_id=doc_id_base,
                        fragment_id=stm_fragment_id,
                        fragment_main_label="STMEntry",
                        relationship_type="HAS_STM_REPRESENTATION",
                        fragment_properties=stm_node_props
                    ))
                    logger.debug(f"Queueing STMEntry link for {stm_fragment_id} to {doc_id_base}")

                # Link LTM entry
                if self.ltm and ltm_content_to_store:
                    ltm_fragment_id = f"{doc_id_base}_ltm_doc"
                    ltm_node_props = {
                        "id": ltm_fragment_id,
                        "text_preview": ltm_content_to_store[:255],
                        "has_embedding": bool(structured_data.content_embedding),
                        "source_document_id": doc_id_base,
                        "type": "ltm_document_content",
                        "created_at": structured_data.created_at.isoformat()
                    }
                    memory_ops.append(self.graph_db.add_memory_fragment_link(
                        document_id=doc_id_base,
                        fragment_id=ltm_fragment_id,
                        fragment_main_label="LTMLogEntry",
                        relationship_type="HAS_LTM_REPRESENTATION",
                        fragment_properties=ltm_node_props
                    ))
                    logger.debug(f"Queueing LTMLogEntry link for {ltm_fragment_id} to {doc_id_base}")

                # Link Raw Cache entry if raw_data_id exists
                # This assumes raw_cache itself is not directly managed by MemoryManager.store,
                # but its existence is recorded in the graph.
                if structured_data.raw_data_id and self.raw_cache: # Check self.raw_cache to align with others
                    raw_cache_node_props = {
                        "id": structured_data.raw_data_id,
                        "type": "raw_data_log",
                        "source_document_id": doc_id_base,
                        "created_at": structured_data.created_at.isoformat()
                    }
                    memory_ops.append(self.graph_db.add_memory_fragment_link(
                        document_id=doc_id_base,
                        fragment_id=structured_data.raw_data_id, # This is the ID of the raw cache entry
                        fragment_main_label="RawCacheEntry",
                        relationship_type="REFERENCES_RAW_CACHE",
                        fragment_properties=raw_cache_node_props
                    ))
                    logger.debug(f"Queueing RawCacheEntry link for {structured_data.raw_data_id} to {doc_id_base}")

                # Add entities, linking them to doc_id_base
                if structured_data.entities:
                    entities_as_dicts = [entity.model_dump(exclude_none=True) for entity in structured_data.entities]
                    # Assuming add_entities_to_document is async
                    memory_ops.append(self.graph_db.add_entities_to_document(document_id=doc_id_base, entities=entities_as_dicts))
                    logger.debug(f"Queueing {len(entities_as_dicts)} entities for document {doc_id_base}")

                # Add relations, linking them to doc_id_base
                if structured_data.relations:
                    relations_as_dicts = [relation.model_dump(exclude_none=True) for relation in structured_data.relations]
                    # Assuming add_relations_to_document is async
                    memory_ops.append(self.graph_db.add_relations_to_document(document_id=doc_id_base, relations=relations_as_dicts))
                    logger.debug(f"Queueing {len(relations_as_dicts)} relations for document {doc_id_base}")

            except Exception as e_graph_init:
                logger.error(f"GraphDB: Critical error during graph node setup for {doc_id_base}: {e_graph_init}", exc_info=True)
                # This was a direct await, so if it fails, we might want to stop before gathering other ops.
                raise MemoryAccessError(f"GraphDB initial node setup failed for {doc_id_base}") from e_graph_init

        # Execute all queued memory operations
        if memory_ops:
            results = await asyncio.gather(*memory_ops, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error(f"Memory operation {i} failed: {res}", exc_info=True)
                    # Collect all errors or raise on first? Original code raises on first.
                    raise MemoryAccessError(f"At least one memory operation failed: {res}") from res
            logger.info(f"All memory operations for {doc_id_base} (STM, LTM, GraphDB links/nodes) completed successfully.")
        else:
            logger.info(f"No memory operations were queued for {doc_id_base} (aside from initial graph node).")

        # The requirement "Saving the raw content to RawCache" needs clarification.
        # If it means `MemoryManager.store` should take raw_content and save it, the signature and logic need change.
        # The current implementation follows `_write_to_memory` which assumes raw content is already in RawCache,
        # and `structured_data.raw_data_id` is its key.
        # If `structured_data` needs to *also* carry the raw text for *this method* to save it,
        # then an additional step for `self.raw_cache.store(...)` would be needed here.
        # Given the prompt "This store method will contain the logic previously in llm_listener.py's _write_to_memory function",
        # and _write_to_memory does NOT save to raw_cache (it's done before in `process_data`),
        # I will assume for now that saving to raw_cache is handled *before* calling MemoryManager.store.
        # The graph linking for RawCacheEntry *is* included.
