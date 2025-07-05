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

    async def edit(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Edits a specific memory item based on its ID.
        The memory_id format will determine which system (STM, LTM, Graph node) to target.
        Example memory_id formats:
        - "doc123_summary" -> STM
        - "doc1_chunk2_summary" -> STM (for chunk-specific summary)
        - "doc123_ltm_doc" -> LTM
        - "entity_person_alice" -> GraphDB node (direct entity edit)
        - "doc_xyz123" -> GraphDB SourceDocument node properties
        Args:
            memory_id: The unique identifier of the memory item to edit.
            updates: A dictionary containing the fields and their new values.
        Returns:
            True if the edit was successful, False otherwise.
        """
        logger.info(f"Attempting to edit memory_id: '{memory_id}' with updates: {updates}")

        # ID Parsing Logic (example, can be made more robust)
        # This is a simplified example. A more robust system might involve a prefix or a lookup.
        if memory_id.endswith("_summary") and self.stm:
            # Target STM: memory_id is summary_id
            try:
                # Assuming stm has an `update_summary` method
                # (to be added to STMInterface and implementations)
                if not hasattr(self.stm, 'update_summary'):
                    logger.error(f"STM interface does not have 'update_summary' method.")
                    return False
                await self.stm.update_summary(summary_id=memory_id, updates=updates)
                logger.info(f"Successfully updated STM entry: {memory_id}")

                # Optionally, update corresponding graph node if one exists and is linked
                # This requires graph_db to have a method to find and update nodes by properties or specific ID pattern.
                # For example, find node with id=memory_id or type="STMEntry" and source_document_id derived from memory_id
                # This part can be complex and depends on graph structure.
                # For now, focusing on direct DB update.

                return True
            except Exception as e:
                logger.error(f"Failed to update STM entry {memory_id}: {e}", exc_info=True)
                raise MemoryAccessError(f"Failed to update STM entry {memory_id}") from e

        elif memory_id.endswith("_ltm_doc") and self.ltm:
            # Target LTM: memory_id is doc_id for LTM
            try:
                # Assuming ltm has an `update_document` or `update_embedding` method
                # (to be added to LTMInterface and implementations)
                if not hasattr(self.ltm, 'update_document'): # or a more generic update_embedding
                    logger.error(f"LTM interface does not have 'update_document' method.")
                    return False
                await self.ltm.update_document(doc_id=memory_id, updates=updates) # Or update_embedding
                logger.info(f"Successfully updated LTM entry: {memory_id}")
                # Similar to STM, graph node linkage could be updated here.
                return True
            except Exception as e:
                logger.error(f"Failed to update LTM entry {memory_id}: {e}", exc_info=True)
                raise MemoryAccessError(f"Failed to update LTM entry {memory_id}") from e

        elif self.graph_db: # General case: assume it's a graph node ID
            # This could be a SourceDocument ID, an entity ID, or any other graph node ID.
            # The `updates` dict would apply to the node's properties.
            try:
                # Assuming graph_db has a generic `update_node_properties` or similar
                if not hasattr(self.graph_db, 'update_node'): # Using generic update_node from plan
                    logger.error(f"GraphDB interface does not have 'update_node' method.")
                    return False
                await self.graph_db.update_node(node_id=memory_id, data=updates)
                logger.info(f"Successfully updated GraphDB node: {memory_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to update GraphDB node {memory_id}: {e}", exc_info=True)
                raise MemoryAccessError(f"Failed to update GraphDB node {memory_id}") from e

        else:
            logger.warning(f"Could not determine target memory system for ID: {memory_id} or system not available.")
            return False

    async def forget(self, memory_id: str) -> bool:
        """
        Performs a cascading delete of a memory item and its related data.
        - GraphDB: Delete the main node and its relationships.
        - LTM: Delete associated embeddings.
        - STM: Delete associated summaries.
        - RawCache: Delete associated raw data (if applicable and ID mappable).
        Args:
            memory_id: The unique identifier of the main memory item to forget
                       (typically a SourceDocument ID in the graph).
        Returns:
            True if the forget operation was successful, False otherwise.
        """
        logger.info(f"Attempting to 'forget' memory_id: '{memory_id}' (typically a SourceDocument ID)")

        # For a cascading delete, the graph is often the source of truth for linked items.
        # We'll primarily use memory_id as a graph node ID (e.g., SourceDocument).

        # Step 1: Delete from GraphDB and retrieve linked item IDs before deletion
        # This is conceptual. The actual implementation depends heavily on graph_db capabilities.
        # graph_db.delete_node might not return linked IDs. We might need a separate query.

        # Let's assume memory_id is the SourceDocument node ID in the graph.
        # We need to find associated STM summaries, LTM documents, and RawCache entries.
        # One way is to query the graph for nodes connected to `memory_id` with specific relationships or types.

        # Placeholder: Assume we can derive associated IDs or they are stored as properties on the main node.
        # This part is complex and needs robust ID management or graph querying.
        # Example: if memory_id = "doc_xyz123"
        stm_id_to_delete = f"{memory_id}_summary"
        ltm_id_to_delete = f"{memory_id}_ltm_doc"
        # raw_cache_id_to_delete = memory_id # If raw_cache_id is the same as source_doc_id, or derived
        # This needs to be confirmed by how raw_data_id is set in StructuredInsight and used in store()
        # From store(), structured_data.raw_data_id is used. If memory_id is doc_id_base, then it's the same.
        raw_cache_id_to_delete = memory_id # Assuming memory_id passed to forget is the base doc_id / raw_data_id
                                           # This means memory_id is the ID for the RawCacheEntry.
                                           # And also the `doc_id_base` for STM/LTM entries.

        delete_ops = []
        success_flags = []

        # Delete from GraphDB (main node and relationships)
        if self.graph_db:
            try:
                # `delete_node` should ideally handle cascading deletes of relationships.
                # If not, relationships must be deleted manually first.
                if not hasattr(self.graph_db, 'delete_node'):
                    logger.error("GraphDB interface does not have 'delete_node' method.")
                    # Consider this a partial failure or handle as per requirements
                else:
                    # This should delete the SourceDocument node and all its incident relationships.
                    # It might also need to delete nodes that are exclusively linked FROM this node
                    # (e.g., Entity nodes that only belong to this document). This depends on graph model.
                    # For now, assume delete_node(memory_id) handles the primary node and its direct links.
                    await self.graph_db.delete_node(node_id=memory_id)
                    logger.info(f"Successfully deleted GraphDB node and its relationships: {memory_id}")
                    success_flags.append(True)
                    # If there are "fragment" nodes in the graph (STMEntry, LTMLogEntry, RawCacheEntry nodes)
                    # linked to this SourceDocument, delete_node might need to handle them, or we do it explicitly.
                    # The `add_memory_fragment_link` implies these are separate nodes.
                    # A robust graph_db.delete_node for a SourceDocument should ideally find and remove these.
                    # Or, we query for them first, then delete them, then delete the SourceDocument node.
                    # For now, we assume graph_db.delete_node(memory_id) is smart or we handle fragments below.

                    # Explicitly delete fragment nodes if not handled by graph_db.delete_node cascade:
                    # await self.graph_db.delete_node(node_id=stm_id_to_delete) # If STMEntry is a graph node
                    # await self.graph_db.delete_node(node_id=ltm_id_to_delete) # If LTMLogEntry is a graph node
                    # await self.graph_db.delete_node(node_id=raw_cache_id_to_delete) # If RawCacheEntry is a graph node

            except Exception as e:
                logger.error(f"Failed to delete GraphDB node {memory_id}: {e}", exc_info=True)
                success_flags.append(False)
                # Decide if we should proceed with other deletions or stop. For 'forget', try to clean up as much as possible.

        # Delete from LTM
        if self.ltm:
            try:
                # Assuming ltm has `delete_embedding` or a more general `delete_document`
                # (to be added to LTMInterface and implementations)
                if not hasattr(self.ltm, 'delete_embedding'):
                     logger.error(f"LTM interface does not have 'delete_embedding' method.")
                else:
                    await self.ltm.delete_embedding(memory_id=ltm_id_to_delete) # LTM uses its own doc_id
                    logger.info(f"Successfully deleted LTM entry: {ltm_id_to_delete}")
                    success_flags.append(True)
            except Exception as e:
                logger.error(f"Failed to delete LTM entry {ltm_id_to_delete}: {e}", exc_info=True)
                success_flags.append(False)

        # Delete from STM
        if self.stm:
            try:
                # Assuming stm has `delete_summary`
                # (to be added to STMInterface and implementations)
                if not hasattr(self.stm, 'delete_summary'):
                    logger.error(f"STM interface does not have 'delete_summary' method.")
                else:
                    await self.stm.delete_summary(summary_id=stm_id_to_delete) # STM uses its own summary_id
                    logger.info(f"Successfully deleted STM entry: {stm_id_to_delete}")
                    success_flags.append(True)
            except Exception as e:
                logger.error(f"Failed to delete STM entry {stm_id_to_delete}: {e}", exc_info=True)
                success_flags.append(False)

        # Delete from RawCache
        # This assumes that `memory_id` passed to `forget` is the `raw_data_id` used for RawCache.
        if self.raw_cache:
            try:
                if not hasattr(self.raw_cache, 'delete_raw_data'): # Assuming a method name
                    logger.error(f"RawCache interface does not have 'delete_raw_data' method.")
                else:
                    await self.raw_cache.delete_raw_data(doc_id=raw_cache_id_to_delete)
                    logger.info(f"Successfully deleted RawCache entry: {raw_cache_id_to_delete}")
                    success_flags.append(True)
            except Exception as e:
                logger.error(f"Failed to delete RawCache entry {raw_cache_id_to_delete}: {e}", exc_info=True)
                success_flags.append(False)

        # If any operation failed, the overall 'forget' might be considered partial or failed.
        # For now, return True if at least one operation succeeded or if no relevant systems were available.
        # A more stringent check would be `all(success_flags)` if success_flags is not empty.
        if not success_flags: # No systems to delete from or all configured systems failed to provide methods
            logger.warning(f"No delete operations performed or all failed for memory_id {memory_id}. Check system configurations and errors.")
            return False # Or True if "nothing to do" is success. Let's say False if nothing was done.

        return any(success_flags) # Return True if at least one delete was attempted and potentially succeeded.
                                 # Or change to all(s for s in success_flags if s is not None) for stricter success.
                                 # Given the cascading nature, if graph deletion fails, subsequent ones might be less meaningful.
                                 # For now, `any` reflects a "best effort" cleanup.
