import asyncio
import logging
from typing import Any, Optional, Dict

from contextkernel.core_logic.chunker import SemanticChunker
from contextkernel.core_logic.hallucination_detector import HallucinationDetector
from contextkernel.core_logic.llm_listener import LLMListener, StructuredInsight
from contextkernel.memory_system.memory_manager import MemoryManager
from contextkernel.utils.config import NLPConfig

logger = logging.getLogger(__name__)

class ContextAgent:
    def __init__(self,
                 nlp_config: NLPConfig,
                 memory_manager: MemoryManager,
                 llm_listener: LLMListener,
                 hallucination_detector: HallucinationDetector,
                 chunker: SemanticChunker,
                 input_queue: asyncio.Queue,
                 # For RetrievalComponent integration:
                 retrieval_input_queue: asyncio.Queue,
                 retrieved_context_output_queue: asyncio.Queue,
                 retriever_component: Optional[Any] = None, # Type hint later with actual RetrieverComponent
                 embedding_client: Optional[Any] = None # Added for retriever's vector search
                 ):
        self.nlp_config = nlp_config
        self.memory_manager = memory_manager
        self.llm_listener = llm_listener
        self.hallucination_detector = hallucination_detector
        self.chunker = chunker
        self.input_queue = input_queue # Main data input for Write & Verify

        # Queues for Retrieval Component
        self.retrieval_input_queue = retrieval_input_queue # Agent sends cues here
        self.retrieved_context_output_queue = retrieved_context_output_queue # Agent gets context from here
        self.retriever_component = retriever_component # Will be set properly after RetrieverComponent is defined

        self.embedding_client = embedding_client # Needed for RetrieverComponent if it does vector search

        self._running = False
        self.write_verify_loop_task: Optional[asyncio.Task] = None
        self.context_usage_loop_task: Optional[asyncio.Task] = None # For consuming retrieved context

        logger.info("ContextAgent initialized.")

    def set_retriever_component(self, retriever_component: Any):
        """Allows setting the retriever component after ContextAgent initialization."""
        self.retriever_component = retriever_component
        logger.info(f"RetrieverComponent set for ContextAgent.")


    async def _process_incoming_data(self, data_item: Any):
        text_content = str(data_item) # Assuming data_item can be stringified
        logger.debug(f"Processing data item for Write & Verify: {text_content[:100]}...")

        # Potential cue for retrieval based on incoming data
        # This is a simple way to trigger retrieval; could be more sophisticated
        await self.retrieval_input_queue.put(text_content[:500]) # Send a snippet as a cue

        chunks = self.chunker.split_text(text_content, max_tokens=self.nlp_config.chunker_max_tokens or 200)
        logger.info(f"Split content into {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}: \"{chunk[:100]}...\"")
            validation_result = self.hallucination_detector.detect(chunk)

            if not validation_result.is_valid:
                logger.warning(f"Chunk {i+1} failed validation: {validation_result.explanation}")
                continue
            logger.info(f"Chunk {i+1} validated successfully.")

            try:
                insights_dict = await self.llm_listener._generate_insights(
                    data=chunk,
                    instructions={"summarize": True, "extract_entities": True, "extract_relations": True},
                    raw_id=None
                )
                if insights_dict:
                    insights_dict["original_data"] = chunk
                    structured_insight = await self.llm_listener._structure_data(insights_dict)
                    if structured_insight:
                        logger.debug(f"Generated structured insight for chunk {i+1}.")
                        await self.memory_manager.store(structured_insight)
                        logger.info(f"Stored structured insight for chunk {i+1} via MemoryManager.")
                    else:
                        logger.warning(f"Could not structure insights for chunk {i+1}. Structured data was None.")
                else:
                    logger.warning(f"Could not generate insights dictionary for chunk {i+1}. Insights dict was None.")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1} for insight generation or storage: {e}", exc_info=True)

    async def _write_verify_loop(self):
        logger.info("Write & Verify loop started.")
        while self._running:
            try:
                data_item = await self.input_queue.get()
                if data_item is None:
                    logger.info("Received None in input_queue, Write & Verify loop stopping.")
                    if self.retrieval_input_queue: # Also signal retrieval input if it's shared or linked
                        await self.retrieval_input_queue.put(None)
                    break
                await self._process_incoming_data(data_item)
                self.input_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Write & Verify loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in Write & Verify loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        logger.info("Write & Verify loop finished.")

    async def _context_usage_loop(self):
        """
        Listens to the retrieved_context_output_queue and processes retrieved context.
        Placeholder for how the agent uses this context.
        """
        logger.info("Context Usage loop started (listening for retrieved context).")
        while self._running:
            try:
                retrieved_package = await self.retrieved_context_output_queue.get()
                if retrieved_package is None: # Shutdown signal
                    logger.info("Received None in retrieved_context_output_queue, Context Usage loop stopping.")
                    break

                cue = retrieved_package.get("cue")
                items = retrieved_package.get("retrieved_items", [])
                logger.info(f"ContextAgent received {len(items)} retrieved items for cue: '{cue[:100]}...'")
                # --- How to "inject" or use this context? ---
                # 1. Log it for now.
                # 2. Add to a short-term buffer for upcoming LLM calls.
                # 3. Use it to modify ongoing task processing.
                # This is a key design area for "continuous cognition".
                for item in items:
                    logger.debug(f"  - Retrieved: {item.get('id')}, score: {item.get('score')}, content: {item.get('content', '')[:100]}...")

                # Example: If there's an active LLM task, this context could be added to its prompt.
                # This requires more state management within the agent.

                self.retrieved_context_output_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Context Usage loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in Context Usage loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        logger.info("Context Usage loop finished.")

    async def start(self):
        if self._running:
            logger.warning("ContextAgent is already running.")
            return

        self._running = True
        logger.info("ContextAgent starting...")

        self.write_verify_loop_task = asyncio.create_task(self._write_verify_loop())
        self.context_usage_loop_task = asyncio.create_task(self._context_usage_loop()) # Start consumer for retrieved context

        if self.retriever_component and hasattr(self.retriever_component, 'start'):
            logger.info("Starting associated RetrieverComponent...")
            await self.retriever_component.start() # Start the retriever component's loop
            logger.info("RetrieverComponent started by ContextAgent.")
        else:
            logger.warning("RetrieverComponent not set or does not have a start method. Retrieval will not be active.")

        logger.info("ContextAgent background loops initiated.")

    async def stop(self):
        if not self._running:
            logger.warning("ContextAgent is not running.")
            return

        logger.info("ContextAgent stopping...")
        self._running = False # Signal loops to stop

        # Stop the RetrieverComponent first
        if self.retriever_component and hasattr(self.retriever_component, 'stop'):
            logger.info("Stopping associated RetrieverComponent...")
            await self.retriever_component.stop()
            logger.info("RetrieverComponent stopped by ContextAgent.")

        # Signal main input queue
        if self.input_queue:
            await self.input_queue.put(None)

        # Signal retrieved context output queue (which context_usage_loop listens to)
        if self.retrieved_context_output_queue:
            await self.retrieved_context_output_queue.put(None)

        tasks_to_wait_for = []
        if self.write_verify_loop_task:
            self.write_verify_loop_task.cancel()
            tasks_to_wait_for.append(self.write_verify_loop_task)
        if self.context_usage_loop_task:
            self.context_usage_loop_task.cancel()
            tasks_to_wait_for.append(self.context_usage_loop_task)

        if tasks_to_wait_for:
            await asyncio.gather(*tasks_to_wait_for, return_exceptions=True)

        logger.info("ContextAgent stopped.")

    async def push_data_to_kernel(self, data: Any, is_retrieval_cue: bool = False):
        if not self._running:
            logger.warning("ContextAgent not running. Cannot push data.")
            return

        if is_retrieval_cue:
            if self.retrieval_input_queue:
                await self.retrieval_input_queue.put(data)
                logger.debug(f"Data pushed to ContextAgent retrieval_input_queue: {str(data)[:100]}")
            else:
                logger.error("Retrieval input queue not initialized.")
        else:
            if self.input_queue:
                await self.input_queue.put(data)
                logger.debug(f"Data pushed to ContextAgent main input_queue: {str(data)[:100]}")
            else:
                logger.error("Main input queue not initialized.")


    async def handle_chat(self, chat_message: Any, session_id: Optional[str], state_manager: Any) -> Any:
        logger.info(f"ContextAgent: Handling chat: {chat_message.message}. Pushing to input queue and as retrieval cue.")
        chat_data_to_process = {
            "type": "chat_message",
            "user_id": chat_message.user_id,
            "session_id": session_id,
            "message": chat_message.message,
            "timestamp": asyncio.get_running_loop().time()
        }
        # Process as regular input (for storage, etc.)
        await self.push_data_to_kernel(str(chat_data_to_process), is_retrieval_cue=False)
        # Also use chat message as a cue for retrieval
        await self.push_data_to_kernel(chat_message.message, is_retrieval_cue=True)

        from contextkernel.interfaces.api import ContextResponse
        return ContextResponse(context_id=session_id or "N/A", data={"reply": "Message received for processing and retrieval by ContextAgent."})

    async def ingest_data(self, data: Any, settings: Any) -> Any:
        content_to_ingest = data.content if data.content else data.source_uri
        document_id = data.document_id or f"ingest_{abs(hash(str(content_to_ingest)))}"

        logger.info(f"ContextAgent: Ingesting data (doc_id: {document_id}): {str(content_to_ingest)[:100]}. Pushing to input queue and as retrieval cue.")

        data_to_process = {
            "type": "data_ingestion",
            "document_id": document_id,
            "source_uri": data.source_uri,
            "content": content_to_ingest,
            "metadata": data.metadata if hasattr(data, 'metadata') else None,
            "timestamp": asyncio.get_running_loop().time()
        }
        # Process as regular input
        await self.push_data_to_kernel(str(data_to_process), is_retrieval_cue=False)
        # Also use content as a cue for retrieval (maybe a summary or keywords would be better)
        await self.push_data_to_kernel(str(content_to_ingest)[:500], is_retrieval_cue=True)

        from contextkernel.interfaces.api import IngestResponse
        return IngestResponse(document_id=document_id, status="processing", message="Data queued for processing and retrieval cue by ContextAgent.")

    async def get_context_details(self, context_id: str, state_manager: Any) -> Optional[Any]:
        logger.warning("ContextAgent.get_context_details needs full implementation using MemoryManager retrieval.")
        # This should ideally query the memory system for a specific context_id or related info.
        # For now, it's a placeholder. Could trigger a retrieval for `context_id` if it's a query string.
        await self.push_data_to_kernel(context_id, is_retrieval_cue=True)
        from contextkernel.interfaces.api import ContextResponse
        return ContextResponse(context_id=context_id, data={"message": f"Retrieval cued for '{context_id}'. Monitor logs or context usage loop."})
