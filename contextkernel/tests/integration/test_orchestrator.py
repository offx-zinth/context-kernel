import asyncio
import logging
import unittest
import datetime # Added for timestamping in potential future structured logging

# Import project modules (adjust paths if necessary based on actual structure)
from contextkernel.core_logic.context_agent import ContextAgent
from contextkernel.core_logic.summarizer import Summarizer
from contextkernel.core_logic.llm_listener import LLMListener
from contextkernel.core_logic.llm_retriever import LLMRetriever

# Import memory system modules
from contextkernel.memory_system.stm import STM # Assuming STM class
from contextkernel.memory_system.ltm import LTM # Assuming LTM class
from contextkernel.memory_system.graph_db import GraphDB # Assuming GraphDB class
from contextkernel.memory_system.raw_cache import RawCache # Assuming RawCache class

# Import mock dependencies
from contextkernel.tests.mocks.mock_llm import MockLLM
from contextkernel.tests.mocks.mock_redis import MockRedis
from contextkernel.tests.mocks.mock_vector_db import MockVectorDB

# Configure basic logging for the test run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TestIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        logger.info("Setting up test environment for TestIntegration...")
        # Initialize Mocks
        self.mock_llm = MockLLM()
        # self.mock_redis_instance = MockRedis() # Not directly injected into LTM/GraphDB/RawCache current stubs
        # self.mock_vector_db_instance = MockVectorDB() # Not directly injected into LTM current stubs

        # Initialize Memory Systems
        # These memory systems currently use internal stubs or simple in-memory dicts/deques.
        # The mock instances (MockRedis, MockVectorDB) are not directly injected into them
        # unless their constructors are changed to accept and use client instances.

        self.ltm = LTM() # Uses internal _vector_db_stub, _raw_store_stub
        self.graph_db = GraphDB() # Uses internal _nodes, _edges
        self.raw_cache = RawCache() # Uses internal _cache_stub (dict-based)

        # STM requires an LTM instance for flushing
        self.stm = STM(ltm_instance=self.ltm, redis_config={'type': 'mock_in_memory_deque'}) # STM uses internal deque

        self.memory_systems = {
            "stm": self.stm,
            "ltm": self.ltm,
            "graph_db": self.graph_db,
            "raw_cache": self.raw_cache
        }

        # Initialize Core Logic components
        self.llm_listener = LLMListener(
            llm_config={"stub_config": True},
            memory_systems=self.memory_systems,
            llm_client=self.mock_llm # Inject MockLLM into LLMListener
        )

        self.llm_retriever = LLMRetriever(
            ltm_interface=self.ltm, # LTM instance (which uses its internal stubs)
            stm_interface=self.stm, # STM instance (which uses its internal deque)
            graphdb_interface=self.graph_db, # GraphDB instance (uses internal stubs)
            embedding_model=self.mock_llm # MockLLM provides generate_embedding for the retriever
        )

        # Summarizer uses internal stubs for LLM calls; does not take an llm_client.
        self.summarizer = Summarizer()

        # Mock for the 'llm_service' expected by ContextAgent's current constructor
        # This mock will now delegate to the actual llm_listener instance for summarization tasks.
        class MockContextAgentLLMService:
            def __init__(self, llm_listener_instance): # llm_listener_instance to call process_data
                self.last_called_method = None
                self.last_params = None
                self.llm_listener = llm_listener_instance

            async def retrieve(self, params):
                logger.info(f"MockContextAgentLLMService.retrieve called with {params}")
                self.last_called_method = "retrieve"
                self.last_params = params
                return f"Mock retrieved data for: {params.get('query', 'N/A')}"

            async def process(self, params):
                logger.info(f"MockContextAgentLLMService.process called with {params}")
                self.last_called_method = "process"
                self.last_params = params

                # If this is a summarization call routed by ContextAgent,
                # params should contain 'raw_data' and 'context_instructions'.
                if params and "raw_data" in params and "context_instructions" in params:
                    # Call the actual LLMListener's process_data method.
                    # LLMListener.process_data itself doesn't return the summary directly,
                    # it writes to memory and its internal _generate_insights creates a summary.
                    # For the test, we want the summary that was generated.
                    # LLMListener._generate_insights is where the summary is created.
                    # LLMListener.process_data -> _preprocess_data -> _generate_insights -> _structure_data -> _write_to_memory

                    # Let's get the summary from LLMListener by calling its insight generation.
                    # Note: process_data doesn't return the summary.
                    # The summary is part of the StructuredInsight.
                    # For the purpose of this mock, we want to return the summary string.

                    # The llm_listener is now equipped with MockLLM, so its _call_llm_summarize will use MockLLM.
                    # We need what _call_llm_summarize would return.
                    insights = await self.llm_listener._generate_insights(
                        data=params["raw_data"],
                        context_instructions=params["context_instructions"]
                    )
                    # The 'insights' dict contains 'summary', 'entities', 'relations'.
                    # The summary here will be from MockLLM via LLMListener.
                    return insights.get("summary", "Failed to get summary from LLMListener insights")
                else:
                    # Fallback for other "process" calls if any (e.g. simple save)
                    return f"Mock processed data (non-summarization): {params.get('data', 'N/A')}"

        self.mock_context_agent_llm_service = MockContextAgentLLMService(llm_listener_instance=self.llm_listener)

        self.context_agent = ContextAgent(
            llm_service=self.mock_context_agent_llm_service,
            memory_system=self.memory_systems, # Passing the dict of memory systems
            config={} # Placeholder config
        )
        logger.info("Test environment setup complete.")

    async def asyncTearDown(self):
        logger.info("Tearing down test environment...")
        # No explicit clear_all_data needed for mock_redis_instance or mock_vector_db_instance
        # as they are not directly managing the state of LTM, GraphDB, RawCache in this setup.
        # The memory system instances themselves will be garbage collected with the test class instance.
        # If MockRedis/MockVectorDB were used by components, clearing would be:
        # if hasattr(self.mock_redis_instance, 'clear_all_data'):
        #     self.mock_redis_instance.clear_all_data()
        # if hasattr(self.mock_vector_db_instance, 'clear_all_data'):
        #     self.mock_vector_db_instance.clear_all_data()
        logger.info("Test environment teardown complete.")

    async def test_dummy_orchestration(self):
        logger.info("Starting dummy orchestration test...")
        self.assertTrue(True)
        logger.info("Dummy orchestration test completed.")

    async def test_simple_conversation_flow(self):
        logger.info("Starting test_simple_conversation_flow...")

        conversation_id = "test_conv_001"
        # Changed input to trigger a known intent in ContextAgent's stubbed logic
        user_input_1 = "search for information about dogs"

        # Simulate ContextAgent handling the first user input
        logger.info(f"test_simple_conversation_flow: Raw input for ContextAgent: '{user_input_1}', Conversation ID: '{conversation_id}'")
        # The ContextAgent.handle_request is an async method
        task_result_1 = await self.context_agent.handle_request(
            raw_input=user_input_1,
            conversation_id=conversation_id
        )
        logger.info(f"test_simple_conversation_flow: TaskResult from ContextAgent: {task_result_1.dict()}")

        self.assertIsNotNone(task_result_1, "TaskResult from ContextAgent should not be None.")
        self.assertEqual(task_result_1.status, "success", f"ContextAgent task did not succeed. Message: {task_result_1.message}")

        # Verify that the correct method on MockContextAgentLLMService was called based on routing
        self.assertEqual(self.mock_context_agent_llm_service.last_called_method, "retrieve",
                         "ContextAgent should have called 'retrieve' on llm_service for a 'search_info' intent.")
        self.assertIsNotNone(self.mock_context_agent_llm_service.last_params)
        actual_query_param = self.mock_context_agent_llm_service.last_params.get('query')
        self.assertEqual(actual_query_param, "for information about dogs")

        # Assert the content of task_result.data
        expected_retrieval_output = f"Mock retrieved data for: {actual_query_param}"
        self.assertEqual(task_result_1.data, expected_retrieval_output,
                         "TaskResult data for retrieval does not match expected mock output.")

        # --- Verify STM interaction (Conceptual - STM is mostly an in-memory deque for now) ---
        # The ContextAgent itself doesn't directly interact with STM in the provided snippet.
        # STM interactions would typically happen via LLMListener (saving summaries/turns) or LLMRetriever (getting context).
        # For this flow, let's assume LLMListener might be called by ContextAgent (though current stub doesn't show that explicitly)
        # or that some component processed and stored to STM.
        # For now, we'll check if the LLMListener processed something if the intent was 'save_info'.
        # Given the input, intent is likely 'unknown_intent', so LLMListener wouldn't be called by ContextAgent's dispatch.

        # We can manually add to STM for testing other components if needed, but for pure ContextAgent flow:
        # If ContextAgent's dispatch were to call LLMListener.process_data with the input:
        # If the intent was 'save_info', we would check:
        # self.assertEqual(self.mock_context_agent_llm_service.last_called_method, "process")
        # current_task_params = self.mock_context_agent_llm_service.last_params
        # self.assertEqual(current_task_params.get('data'), "data to save")


        # --- Further conceptual checks for STM/LTM (if ContextAgent directly used LLMListener/Retriever instances) ---
        # These are commented out because ContextAgent currently uses the llm_service facade.
        # If ContextAgent.dispatch_task directly called self.llm_listener.process_data or self.llm_retriever.retrieve,
        # then we could spy on those specific component methods.

        # Example (if ContextAgent directly used LLMListener for a 'save' intent):
        # if self.mock_context_agent_llm_service.last_called_method == "process":
            # This implies LLMListener logic should have been triggered if ContextAgent was more direct.
            # We would then need to mock/spy LLMListener's methods to see if it interacted with STM/LTM.
            # logger.info("Conceptual: Verifying data was processed by LLMListener and potentially stored in STM/LTM.")
            # For instance, if LLMListener.process_data was called and it stores to STM:
            # stm_content = await self.stm.get_recent_turns(conversation_id, num_turns=1)
            # self.assertTrue(len(stm_content) > 0, "Data should be in STM if LLMListener was called.")

        # Example (if ContextAgent directly used LLMRetriever for a 'search' intent):
        # if self.mock_context_agent_llm_service.last_called_method == "retrieve":
            # This implies LLMRetriever logic.
            # logger.info("Conceptual: Verifying LLMRetriever was called and might have queried LTM.")
            # Test actual LTM retrieval via self.llm_retriever (which uses a stubbed LTM)
            # query_used_by_agent = self.mock_context_agent_llm_service.last_params.get('query')
            # if query_used_by_agent:
            #     retriever_results = await self.llm_retriever.retrieve(query=query_used_by_agent, top_k=1)
            #     self.assertTrue(len(retriever_results.items) >= 0) # Check if LTM stub returned something

        logger.info("test_simple_conversation_flow completed successfully.")

    async def test_summarization_flow(self):
        logger.info("Starting test_summarization_flow...")

        transcript = "Alice: Hello everyone. Bob: Hi Alice. Today's topic is project X. Alice: Great, I have updates. Charlie: Me too."
        user_input = f"summarize this meeting transcript: {transcript}"
        conversation_id = "conv_summarize_001"

        logger.info(f"test_summarization_flow: Raw input for ContextAgent: '{user_input}', Conversation ID: '{conversation_id}'")
        task_result = await self.context_agent.handle_request(
            raw_input=user_input,
            conversation_id=conversation_id
        )
        logger.info(f"test_summarization_flow: TaskResult from ContextAgent: {task_result.dict()}")

        self.assertIsNotNone(task_result, "TaskResult from ContextAgent should not be None.")
        self.assertEqual(task_result.status, "success", f"Summarization task did not succeed. Message: {task_result.message}")

        self.assertEqual(self.mock_context_agent_llm_service.last_called_method, "process",
                         "ContextAgent should have called 'process' on llm_service for a 'summarization_intent'.")

        # Determine what ContextAgent logic would extract as text_to_summarize
        # user_input = f"summarize this meeting transcript: {transcript}"
        # processed_user_input = user_input.lower() # Done by ContextAgent.process_input
        # text_extracted_by_agent = processed_user_input.split("summarize ", 1)[1]
        # This is what should be in last_params.get('raw_data')

        expected_text_for_summarizer = ("this meeting transcript: " + transcript).lower()


        self.assertIsNotNone(self.mock_context_agent_llm_service.last_params)
        actual_raw_data_in_params = self.mock_context_agent_llm_service.last_params.get('raw_data')
        self.assertEqual(actual_raw_data_in_params, expected_text_for_summarizer)
        self.assertTrue(self.mock_context_agent_llm_service.last_params.get('context_instructions', {}).get('process_for_summarization'))

        # The task_result.data should contain the summary from MockLLM via LLMListener,
        # based on the text_extracted_by_agent.
        expected_summary_prefix = f"Mock summary for text: '{expected_text_for_summarizer[:50]}...'" # MockLLM.summarize format
        self.assertIn(expected_summary_prefix, task_result.data,
                      f"Expected summary prefix not found in task result. Got: {task_result.data}. Expected prefix: {expected_summary_prefix}")
        self.assertIn("(max_length: 100)", task_result.data) # MockLLM.summarize format

        # Conceptual: Check if summary was stored in LTM by LLMListener
        # This would require inspecting LTM or mocking/spying on LTM.store_memory_chunk
        # For now, we confirmed the summary was generated and returned.

        logger.info("test_summarization_flow completed successfully.")


if __name__ == '__main__':
    unittest.main()
