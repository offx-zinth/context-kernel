import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest # For skipping tests

# ContextAgent is removed. NLPConfig, TaskResult, IntentExtractionResult, RoutingDecision are now in nlp_utils / core_logic
from contextkernel.core_logic import NLPConfig, TaskResult, IntentExtractionResult, RoutingDecision, process_input, initialize_matcher, keyword_end_token_index, detect_intent
from contextkernel.core_logic.exceptions import ConfigurationError

@pytest.mark.skip(reason="ContextAgent class removed, all tests in this suite need complete refactoring or removal.")
class TestContextAgentProactiveFeatures(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_llm_service = MagicMock()
        self.mock_memory_system = MagicMock()
        self.mock_state_manager = AsyncMock()

        # Default config with proactive features disabled
        self.default_config = NLPConfig() # Was ContextAgentConfig

        # Config with proactive features enabled
        # proactive_enabled is not part of NLPConfig anymore.
        # For the purpose of this refactor, we'll remove it.
        # If these tests were to be refactored, this logic would need rethinking.
        self.proactive_config = NLPConfig() # Was ContextAgentConfig(proactive_enabled=True)

        # Mock NLP components for ContextAgent initialization to avoid real model loading
        self.mock_spacy_nlp = MagicMock()
        self.mock_spacy_matcher = MagicMock()

        # Patch spacy.load and Matcher to return our mocks
        self.patcher_spacy_load = patch('spacy.load', return_value=self.mock_spacy_nlp)
        self.patcher_matcher = patch('spacy.matcher.Matcher', return_value=self.mock_spacy_matcher)

        self.mock_spacy_load = self.patcher_spacy_load.start()
        self.mock_matcher_constructor = self.patcher_matcher.start()

        # Mock intent classifier pipeline if transformers is available
        # Check if the class attribute 'intent_classifier' (the actual pipeline) would be loaded,
        # not the config 'intent_classifier_model'. The agent tries to load it if model name is present in config.
        # So, we base our patching decision on whether the agent *would attempt* to load it.
        if self.default_config.intent_classifier_model: # Check if a model is configured
            # Try to patch 'transformers.pipeline' only if it's expected to be called.
            try:
                self.patcher_pipeline = patch('transformers.pipeline', return_value=AsyncMock())
                self.mock_pipeline = self.patcher_pipeline.start()
            except ImportError: # If transformers itself is not installed, pipeline won't be available.
                self.patcher_pipeline = None
                self.mock_pipeline = None
        else:
            self.patcher_pipeline = None
            self.mock_pipeline = None


    def tearDown(self):
        self.patcher_spacy_load.stop()
        self.patcher_matcher.stop()
        if self.patcher_pipeline: # Only stop if it was started
            self.patcher_pipeline.stop()


    async def test_handle_request_proactive_disabled(self):
        agent = ContextAgent(
            llm_service=self.mock_llm_service,
            memory_system=self.mock_memory_system,
            agent_config=self.default_config, # Proactive disabled
            state_manager=self.mock_state_manager
        )

        # Mock standard methods
        agent.process_input = MagicMock(return_value="processed input")
        agent.detect_intent = AsyncMock(return_value=IntentExtractionResult(intent="test_intent", confidence=0.9, original_input="processed input"))
        agent.decide_route = MagicMock(return_value=RoutingDecision(target_module="TestModule", task_parameters={}))
        agent.dispatch_task = AsyncMock(return_value=TaskResult(status="success", data="task data"))

        # Spy on proactive methods
        agent._detect_latent_intent = AsyncMock()
        agent._proactively_check_memory = AsyncMock()
        agent._trigger_listener_for_memory_creation = AsyncMock()
        agent._inject_proactive_context = AsyncMock()

        raw_input = "test input"
        await agent.handle_request(raw_input)

        agent._detect_latent_intent.assert_not_called()
        agent._proactively_check_memory.assert_not_called()
        agent._trigger_listener_for_memory_creation.assert_not_called()
        agent._inject_proactive_context.assert_not_called()
        self.mock_state_manager.save_state.assert_called_once()


    async def test_handle_request_proactive_enabled_memory_found(self):
        agent = ContextAgent(
            llm_service=self.mock_llm_service,
            memory_system=self.mock_memory_system,
            agent_config=self.proactive_config, # Proactive enabled
            state_manager=self.mock_state_manager
        )

        # Mock standard methods
        agent.process_input = MagicMock(return_value="processed input")
        agent.detect_intent = AsyncMock(return_value=IntentExtractionResult(intent="test_intent", confidence=0.9, original_input="processed input"))
        agent.decide_route = MagicMock(return_value=RoutingDecision(target_module="TestModule", task_parameters={}))
        agent.dispatch_task = AsyncMock(return_value=TaskResult(status="success", data="task data"))

        # Mock proactive methods' behavior
        agent._detect_latent_intent = AsyncMock(return_value="detected_latent_intent")
        agent._proactively_check_memory = AsyncMock(return_value={"retrieved": "proactive data"})
        agent._inject_proactive_context = AsyncMock(return_value={"proactive_context": {"retrieved": "proactive data"}})
        agent._trigger_listener_for_memory_creation = AsyncMock() # Should not be called

        raw_input = "test input with latent need"
        initial_context = {"user_id": "123"}

        # IMPORTANT: The agent modifies the context in-place if it's passed.
        # To check the final state of current_context, we can inspect it after the call.
        # However, handle_request is designed to work with its own copy if None is passed,
        # or modify the passed dict.

        # Let's test the scenario where current_context is modified.
        current_context_to_pass = initial_context.copy()
        await agent.handle_request(raw_input, current_context=current_context_to_pass)


        agent._detect_latent_intent.assert_called_once_with(raw_input, env_signals=None)
        agent._proactively_check_memory.assert_called_once_with("detected_latent_intent")
        agent._inject_proactive_context.assert_called_once_with({"retrieved": "proactive data"})
        agent._trigger_listener_for_memory_creation.assert_not_called()

        # Verify that the passed current_context was updated
        self.assertIn("proactive_context", current_context_to_pass)
        self.assertEqual(current_context_to_pass["proactive_context"], {"retrieved": "proactive data"})
        self.mock_state_manager.save_state.assert_called_once()


    async def test_handle_request_proactive_enabled_memory_not_found_trigger_creation(self):
        agent = ContextAgent(
            llm_service=self.mock_llm_service,
            memory_system=self.mock_memory_system,
            agent_config=self.proactive_config, # Proactive enabled
            state_manager=self.mock_state_manager
        )

        agent.process_input = MagicMock(return_value="processed input")
        agent.detect_intent = AsyncMock(return_value=IntentExtractionResult(intent="test_intent", confidence=0.9, original_input="processed input"))
        agent.decide_route = MagicMock(return_value=RoutingDecision(target_module="TestModule", task_parameters={}))
        agent.dispatch_task = AsyncMock(return_value=TaskResult(status="success", data="task data"))

        agent._detect_latent_intent = AsyncMock(return_value="another_latent_intent")
        agent._proactively_check_memory = AsyncMock(return_value=None) # Memory not found
        agent._trigger_listener_for_memory_creation = AsyncMock(return_value={"status": "simulated_creation"})
        agent._inject_proactive_context = AsyncMock() # Should not be called

        raw_input = "test input for memory creation"
        await agent.handle_request(raw_input)

        agent._detect_latent_intent.assert_called_once_with(raw_input, env_signals=None)
        agent._proactively_check_memory.assert_called_once_with("another_latent_intent")
        agent._trigger_listener_for_memory_creation.assert_called_once_with("another_latent_intent", raw_input)
        agent._inject_proactive_context.assert_not_called()

    async def test_handle_request_proactive_enabled_no_specific_latent_intent(self):
        agent = ContextAgent(
            llm_service=self.mock_llm_service,
            memory_system=self.mock_memory_system,
            agent_config=self.proactive_config, # Proactive enabled
            state_manager=self.mock_state_manager
        )

        agent.process_input = MagicMock(return_value="processed input")
        agent.detect_intent = AsyncMock(return_value=IntentExtractionResult(intent="test_intent", confidence=0.9, original_input="processed input"))
        agent.decide_route = MagicMock(return_value=RoutingDecision(target_module="TestModule", task_parameters={}))
        agent.dispatch_task = AsyncMock(return_value=TaskResult(status="success", data="task data"))

        agent._detect_latent_intent = AsyncMock(return_value="latent_intent_placeholder") # Placeholder means no specific intent
        agent._proactively_check_memory = AsyncMock()
        agent._trigger_listener_for_memory_creation = AsyncMock()
        agent._inject_proactive_context = AsyncMock()

        raw_input = "general input"
        await agent.handle_request(raw_input)

        agent._detect_latent_intent.assert_called_once_with(raw_input, env_signals=None)
        agent._proactively_check_memory.assert_not_called()
        agent._trigger_listener_for_memory_creation.assert_not_called()
        agent._inject_proactive_context.assert_not_called()

    # Basic tests for the placeholder methods themselves
    async def test_proactive_method_placeholders(self):
        # Need to use a config where proactive features would be enabled to ensure methods are covered
        # if they had internal checks based on config (they don't currently, but good practice)
        agent = ContextAgent(
            llm_service=self.mock_llm_service,
            memory_system=self.mock_memory_system,
            agent_config=self.proactive_config, # Proactive enabled
            state_manager=self.mock_state_manager
        )

        # _detect_latent_intent
        self.assertEqual(await agent._detect_latent_intent("input", None), "latent_intent_placeholder")
        self.assertEqual(await agent._detect_latent_intent("remember this important fact", None), "latent_intent_remember_info")

        # _proactively_check_memory
        self.assertIsNone(await agent._proactively_check_memory("some_intent"))
        self.assertIsNone(await agent._proactively_check_memory("latent_intent_remember_info")) # Current placeholder returns None

        # _trigger_listener_for_memory_creation
        # Mock the llm_service.listener part if it's not already sufficiently mocked by self.mock_llm_service
        mock_listener_instance = AsyncMock()
        mock_listener_instance.process_data = AsyncMock(return_value="processed_by_listener")
        agent.llm_service.listener = mock_listener_instance # Attach it to the agent's llm_service mock

        creation_result = await agent._trigger_listener_for_memory_creation("intent", "input")
        self.assertIsNotNone(creation_result)
        self.assertEqual(creation_result["status"], "simulated_memory_created")
        # The actual call to llm_service.listener.process_data is commented out in the placeholder,
        # so we can't assert_called_with on mock_listener_instance.process_data yet.
        # If it were active, we would:
        # mock_listener_instance.process_data.assert_called_once_with(raw_data="input", context_instructions={"latent_intent": "intent", "action": "store_proactively"})

        # _inject_proactive_context
        self.assertEqual(await agent._inject_proactive_context("memory_string"), {"proactive_context": {"data": "memory_string"}})
        self.assertEqual(await agent._inject_proactive_context({"key": "val"}), {"proactive_context": {"key": "val"}})

if __name__ == '__main__':
    unittest.main()


class TestNLPUtils(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.nlp_config = NLPConfig() # Default config

    def setUp(self):
        self.nlp_config = NLPConfig() # Default config
        # Load a real (but small) spacy model for some tests
        try:
            self.nlp = spacy.load(self.nlp_config.spacy_model_name)
        except OSError:
            # Fallback if model not available (e.g. in some CI environments without full setup)
            # This might make some tests less effective but allows the suite to run.
            # Consider requiring the model for full test runs.
            self.nlp = None
            print(f"Warning: spaCy model '{self.nlp_config.spacy_model_name}' not found. Some NLP tests will be limited.")


    def test_process_input(self):
        self.assertEqual(process_input("  Test Input  "), "test input")
        self.assertEqual(process_input("TEST InPuT"), "test input")
        self.assertEqual(process_input(123), "123")
        self.assertEqual(process_input(None), "")
        self.assertEqual(process_input(""), "")

    def test_keyword_end_token_index(self):
        if not self.nlp:
            self.skipTest(f"spaCy model '{self.nlp_config.spacy_model_name}' not available, skipping keyword_end_token_index test.")

        doc1 = self.nlp("search for cat pictures")
        # Matcher usually provides (match_id, start_token_idx, end_token_idx)
        # "search" is token 0. "for" is token 1.
        match1 = (123, 0, 4) # (match_id, start_idx of "search", end_idx of "pictures")
        self.assertEqual(keyword_end_token_index(doc1, match1), 1) # Should point to index of "for"

        doc2 = self.nlp("look up the weather")
        match2 = (456, 0, 4) # "look" is 0, "up" is 1, "the" is 2
        self.assertEqual(keyword_end_token_index(doc2, match2), 2) # Should point to index of "the"

        doc3 = self.nlp("summarize this document please")
        match3 = (789, 0, 4) # "summarize" is 0, "this" is 1
        self.assertEqual(keyword_end_token_index(doc3, match3), 1) # Should point to index of "this"

        doc4 = self.nlp("just some random text") # No keyword
        match4 = (101, 0, 4)
        self.assertEqual(keyword_end_token_index(doc4, match4), 4) # Should return original end if no keyword match

    def test_initialize_matcher(self):
        if not self.nlp:
            self.skipTest(f"spaCy model '{self.nlp_config.spacy_model_name}' not available, skipping initialize_matcher test.")

        matcher = initialize_matcher(self.nlp)
        self.assertIsNotNone(matcher)

        doc_search = self.nlp("search for information")
        matches_search = matcher(doc_search)
        self.assertTrue(len(matches_search) > 0)
        self.assertEqual(self.nlp.vocab.strings[matches_search[0][0]], "search_info")

        doc_save = self.nlp("remember this note")
        matches_save = matcher(doc_save)
        self.assertTrue(len(matches_save) > 0)
        self.assertEqual(self.nlp.vocab.strings[matches_save[0][0]], "save_info")

        doc_summarize = self.nlp("summarize the article")
        matches_summarize = matcher(doc_summarize)
        self.assertTrue(len(matches_summarize) > 0)
        self.assertEqual(self.nlp.vocab.strings[matches_summarize[0][0]], "summarization_intent")


    # More tests will be added here for detect_intent
