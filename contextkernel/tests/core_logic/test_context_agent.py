import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from contextkernel.core_logic.context_agent import ContextAgent
from contextkernel.core_logic.chunker import SemanticChunker
from contextkernel.core_logic.hallucination_detector import HallucinationDetector, ValidationResult
from contextkernel.core_logic.llm_listener import LLMListener, StructuredInsight, Summary # Assuming Summary is needed for StructuredInsight
from contextkernel.core_logic.retriever_component import RetrievalComponent
from contextkernel.memory_system.memory_manager import MemoryManager
from pydantic import BaseModel # Import BaseModel for NLPConfig if not already available globally for tests

# Helper to run async tests
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# Define a simple NLPConfig for testing if not available from contextkernel.utils.config directly for tests
class TestNLPConfig(BaseModel):
    chunker_max_tokens: int = 50
    # Add other fields expected by ContextAgent's nlp_config if any
    # For example, if retriever config is part of NLPConfig:
    retriever_top_k: int = 3
    retriever_enable_vector_search: bool = False
    retriever_vector_index_name: str = "test_nlp_config_index"


class TestContextAgent(unittest.TestCase):

    def setUp(self):
        self.nlp_config = TestNLPConfig()
        self.mock_memory_manager = AsyncMock(spec=MemoryManager)
        self.mock_llm_listener = AsyncMock(spec=LLMListener)
        # Mock internal methods of llm_listener that agent calls
        self.mock_llm_listener._generate_insights = AsyncMock(return_value={
            "summary": "Generated summary", "entities": [], "relations": [], "original_data": "chunk1" # Ensure original_data is present
        })
        self.sample_summary_obj = Summary(text="Generated summary") # Create Summary object
        self.sample_insight = StructuredInsight(summary=self.sample_summary_obj, original_data_type="text", source_data_preview="chunk1") # Ensure all required fields
        self.mock_llm_listener._structure_data = AsyncMock(return_value=self.sample_insight)

        self.mock_hallucination_detector = MagicMock(spec=HallucinationDetector)
        self.mock_chunker = MagicMock(spec=SemanticChunker)

        self.agent_input_q = asyncio.Queue()
        self.retrieval_input_q = asyncio.Queue()
        self.retrieved_context_q = asyncio.Queue()

        self.mock_retriever_component = AsyncMock(spec=RetrievalComponent)
        self.mock_embedding_client = AsyncMock()

        self.agent = ContextAgent(
            nlp_config=self.nlp_config,
            memory_manager=self.mock_memory_manager,
            llm_listener=self.mock_llm_listener,
            hallucination_detector=self.mock_hallucination_detector,
            chunker=self.mock_chunker,
            input_queue=self.agent_input_q,
            retrieval_input_queue=self.retrieval_input_q,
            retrieved_context_output_queue=self.retrieved_context_q,
            retriever_component=self.mock_retriever_component,
            embedding_client=self.mock_embedding_client
        )

        # Default mock behaviors
        self.mock_chunker.split_text.return_value = ["chunk1", "chunk2"]
        self.mock_hallucination_detector.detect.return_value = ValidationResult(is_valid=True, explanation="")
        self.mock_memory_manager.store.return_value = None


    @async_test
    async def test_write_verify_loop_processes_data(self):
        test_data = "This is a test document."

        await self.agent.start()
        await self.agent_input_q.put(test_data)

        await asyncio.sleep(0.1)

        self.mock_chunker.split_text.assert_called_once_with(test_data, max_tokens=self.nlp_config.chunker_max_tokens)
        self.assertEqual(self.mock_hallucination_detector.detect.call_count, 2)
        self.mock_hallucination_detector.detect.assert_any_call("chunk1")
        self.mock_hallucination_detector.detect.assert_any_call("chunk2")

        self.assertEqual(self.mock_llm_listener._generate_insights.call_count, 2)
        self.mock_llm_listener._generate_insights.assert_any_call(data="chunk1", instructions=unittest.mock.ANY, raw_id=None)

        # Check that 'original_data' was added to insights_dict before _structure_data
        # This requires inspecting the actual call_args if _generate_insights doesn't include it.
        # The current agent code adds it: insights_dict["original_data"] = chunk

        self.assertEqual(self.mock_llm_listener._structure_data.call_count, 2)
        # Example: Check args of _structure_data for chunk1
        args_chunk1, _ = self.mock_llm_listener._structure_data.call_args_list[0] # Assuming chunk1 is first
        self.assertEqual(args_chunk1[0].get("original_data"), "chunk1")


        self.assertEqual(self.mock_memory_manager.store.call_count, 2)
        self.mock_memory_manager.store.assert_any_call(self.sample_insight)

        retrieval_cue_sent = await asyncio.wait_for(self.retrieval_input_q.get(), timeout=0.1)
        self.assertEqual(retrieval_cue_sent, test_data[:500])
        self.retrieval_input_q.task_done()

        await self.agent.stop()


    @async_test
    async def test_write_verify_loop_handles_invalid_chunk(self):
        self.mock_hallucination_detector.detect.side_effect = [
            ValidationResult(is_valid=True, explanation=""),
            ValidationResult(is_valid=False, explanation="chunk2 is bad")
        ]

        await self.agent.start()
        await self.agent_input_q.put("Test data with one invalid chunk.")
        await asyncio.sleep(0.1)

        self.assertEqual(self.mock_llm_listener._generate_insights.call_count, 1)
        self.mock_llm_listener._generate_insights.assert_called_once_with(data="chunk1", instructions=unittest.mock.ANY, raw_id=None)
        self.assertEqual(self.mock_memory_manager.store.call_count, 1)

        await self.agent.stop()

    @async_test
    async def test_context_usage_loop_processes_retrieved_context(self):
        retrieved_package = {
            "cue": "test cue",
            "retrieved_items": [{"id": "ctx1", "content": "Retrieved context content"}]
        }

        await self.agent.start()
        await self.retrieved_context_q.put(retrieved_package)

        with patch('logging.Logger.info') as mock_log_info:
            await asyncio.sleep(0.1)

            found_log = False
            for call_arg_tuple in mock_log_info.call_args_list:
                log_message = call_arg_tuple[0][0] # The first argument of the call
                if f"ContextAgent received 1 retrieved items for cue: '{retrieved_package['cue']}" in log_message:
                    found_log = True
                    break
            self.assertTrue(found_log, "Log message for received context not found.")

        await self.agent.stop()

    @async_test
    async def test_agent_start_starts_retriever_component(self):
        await self.agent.start()
        self.mock_retriever_component.start.assert_called_once()
        await self.agent.stop()

    @async_test
    async def test_agent_stop_stops_retriever_and_loops(self):
        await self.agent.start()
        self.assertTrue(self.agent._running)
        self.assertIsNotNone(self.agent.write_verify_loop_task)
        self.assertIsNotNone(self.agent.context_usage_loop_task)
        self.mock_retriever_component.start.assert_called_once()

        self.assertFalse(self.agent.write_verify_loop_task.done())

        await self.agent.stop()

        self.assertFalse(self.agent._running)
        self.mock_retriever_component.stop.assert_called_once()

        await asyncio.sleep(0.01)
        if self.agent.write_verify_loop_task:
             self.assertTrue(self.agent.write_verify_loop_task.done() or self.agent.write_verify_loop_task.cancelled())
        if self.agent.context_usage_loop_task:
             self.assertTrue(self.agent.context_usage_loop_task.done() or self.agent.context_usage_loop_task.cancelled())


    @async_test
    async def test_push_data_to_kernel_main_queue(self):
        data = "some important data"
        await self.agent.start()
        await self.agent.push_data_to_kernel(data, is_retrieval_cue=False)

        queued_item = await asyncio.wait_for(self.agent_input_q.get(), timeout=0.1)
        self.assertEqual(queued_item, data)
        self.agent_input_q.task_done()

        await self.agent.stop()

    @async_test
    async def test_push_data_to_kernel_retrieval_queue(self):
        cue = "find info about this"
        await self.agent.start()
        await self.agent.push_data_to_kernel(cue, is_retrieval_cue=True)

        queued_cue = await asyncio.wait_for(self.retrieval_input_q.get(), timeout=0.1)
        self.assertEqual(queued_cue, cue)
        self.retrieval_input_q.task_done()

        await self.agent.stop()

if __name__ == '__main__':
    unittest.main()
