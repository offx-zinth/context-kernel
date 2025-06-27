import asyncio
import logging
from typing import Any, Optional, List, Dict

from contextkernel.memory_system.graph_db import GraphDB
from contextkernel.memory_system.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class RetrievalComponent:
    def __init__(self,
                 memory_manager: MemoryManager,
                 input_queue: asyncio.Queue,
                 output_queue: asyncio.Queue,
                 retrieval_config: Optional[Dict[str, Any]] = None):

        self.memory_manager = memory_manager
        self.graph_db = memory_manager.graph_db
        if not self.graph_db or not isinstance(self.graph_db, GraphDB):
            raise ValueError("RetrievalComponent requires a valid GraphDB instance via MemoryManager.")

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = retrieval_config or {}

        self.top_k_results = self.config.get("top_k", 5)
        self.embedding_client = self.config.get("embedding_client", None)


        self._running = False
        self.retrieval_loop_task: Optional[asyncio.Task] = None
        logger.info("RetrievalComponent initialized.")

    async def _get_embedding_for_cue(self, cue: str) -> Optional[List[float]]:
        if not self.embedding_client:
            logger.warning("Embedding client not available for RetrievalComponent, cannot generate cue embeddings.")
            return None

        try:
            if hasattr(self.embedding_client, 'generate_embedding') and \
               asyncio.iscoroutinefunction(self.embedding_client.generate_embedding):
                embedding = await self.embedding_client.generate_embedding(cue)
                logger.debug(f"Generated embedding for cue: '{cue}'")
                return embedding
            elif hasattr(self.embedding_client, 'encode'):
                 embedding = self.embedding_client.encode(cue).tolist()
                 logger.debug(f"Generated embedding for cue (sync encode): '{cue}'")
                 return embedding
            else:
                logger.warning(f"Embedding client for RetrievalComponent does not have a recognized embedding method.")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding for cue '{cue}': {e}", exc_info=True)
            return None


    async def _search_memory(self, cue: str) -> List[Dict[str, Any]]:
        retrieved_items = []

        try:
            logger.debug(f"Performing graph keyword search for cue: '{cue}'")
            graph_results = await self.graph_db.search(query_text=cue, top_k=self.top_k_results)
            if graph_results:
                logger.info(f"Graph keyword search returned {len(graph_results)} results for cue '{cue}'.")
                retrieved_items.extend(graph_results)
            else:
                logger.info(f"Graph keyword search for '{cue}' returned no results.")
        except Exception as e:
            logger.error(f"Error during graph keyword search for cue '{cue}': {e}", exc_info=True)

        if self.config.get("enable_vector_search", False):
            cue_embedding = await self._get_embedding_for_cue(cue)
            if cue_embedding:
                try:
                    logger.debug(f"Performing vector search for cue: '{cue}'")
                    vector_index_name = self.config.get("vector_search_index_name", "node_embedding_index")
                    vector_results_raw = await self.graph_db.vector_search(
                        embedding=cue_embedding,
                        top_k=self.top_k_results,
                        index_name=vector_index_name
                    )
                    if vector_results_raw:
                        logger.info(f"Vector search returned {len(vector_results_raw)} raw results for cue '{cue}'.")
                        for res in vector_results_raw:
                            node_data = res.get("data", {})
                            content_preview = node_data.get("text", str(node_data))[:200] + "..."

                            retrieved_items.append({
                                "id": res.get("node_id") or node_data.get("node_id"),
                                "content": content_preview,
                                "source": f"vector_search_{vector_index_name}",
                                "score": res.get("score"),
                                "metadata": {"raw_node_properties": node_data}
                            })
                    else:
                        logger.info(f"Vector search for '{cue}' (index: {vector_index_name}) returned no results.")
                except Exception as e:
                    logger.error(f"Error during vector search for cue '{cue}': {e}", exc_info=True)
            else:
                logger.debug(f"Skipping vector search for cue '{cue}' as embedding could not be generated or client not available.")
        else:
            logger.debug("Vector search disabled by configuration.")

        final_results = []
        seen_ids = set()
        for item in retrieved_items:
            item_id = item.get("id")
            if item_id and item_id not in seen_ids:
                final_results.append(item)
                seen_ids.add(item_id)
            elif not item_id:
                logger.warning(f"Retrieved item has no ID: {item.get('content', '')[:50]}")
                final_results.append(item)

        final_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return final_results[:self.top_k_results]


    async def _retrieval_loop(self):
        logger.info("RetrievalComponent loop started.")
        while self._running:
            try:
                cue_item = await self.input_queue.get()
                if cue_item is None:
                    logger.info("RetrievalComponent received None, loop stopping.")
                    break

                cue_query = str(cue_item)
                logger.info(f"RetrievalComponent processing cue: {cue_query[:100]}...")

                retrieved_context_list = await self._search_memory(cue_query)

                if retrieved_context_list:
                    logger.info(f"Retrieved {len(retrieved_context_list)} items for cue '{cue_query}'. Pushing to output queue.")
                    output_package = {
                        "cue": cue_query,
                        "retrieved_items": retrieved_context_list,
                        "timestamp": asyncio.get_running_loop().time()
                    }
                    await self.output_queue.put(output_package)
                else:
                    logger.info(f"No context retrieved for cue: '{cue_query}'.")

                self.input_queue.task_done()

            except asyncio.CancelledError:
                logger.info("RetrievalComponent loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in RetrievalComponent loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        logger.info("RetrievalComponent loop finished.")

    async def start(self):
        if self._running:
            logger.warning("RetrievalComponent is already running.")
            return
        self._running = True
        logger.info("RetrievalComponent starting...")
        self.retrieval_loop_task = asyncio.create_task(self._retrieval_loop())
        logger.info("RetrievalComponent background loop initiated.")

    async def stop(self):
        if not self._running:
            logger.warning("RetrievalComponent is not running.")
            return

        logger.info("RetrievalComponent stopping...")
        self._running = False

        if self.input_queue:
            await self.input_queue.put(None)

        if self.retrieval_loop_task:
            self.retrieval_loop_task.cancel()
            try:
                await self.retrieval_loop_task
            except asyncio.CancelledError:
                logger.info("Retrieval loop task successfully cancelled.")

        logger.info("RetrievalComponent stopped.")
