import unittest
from unittest.mock import patch, MagicMock, ANY
import logging
import spacy # Import spacy for type hinting and potential real object creation if not fully mocked

# Assuming ContextAgent, ContextAgentConfig, IntentExtractionResult, RoutingDecision, TaskResult are in context_agent.py
from contextkernel.core_logic.context_agent import ContextAgent, ContextAgentConfig, IntentExtractionResult, RoutingDecision, TaskResult

# Disable logging for tests unless specifically needed
logging.disable(logging.CRITICAL)

class TestContextAgentProduction(unittest.TestCase):

    def setUp(self):
        """Set up common resources for tests."""
        self.mock_llm_service = MagicMock()
        self.mock_memory_system = MagicMock()
        self.mock_state_manager = MagicMock()

        # Create a default config for the agent
        self.agent_config = ContextAgentConfig(
            spacy_model_name="en_core_web_sm_test", # Use a distinct name for test config if needed
            low_confidence_threshold=0.6,
            default_intent_confidence=0.5,
            high_confidence_threshold=0.85 # Consistent with agent's use
        )

        # Patch spacy.load for all tests in this class to avoid actual model loading
        # and to control the returned nlp object.
        self.patcher_spacy_load = patch('spacy.load')
        self.mock_spacy_load = self.patcher_spacy_load.start()

        self.mock_nlp = MagicMock()
        self.mock_nlp.vocab = MagicMock() # Mock vocab attribute
        self.mock_nlp.vocab.strings = MagicMock() # Mock strings attribute on vocab
        self.mock_spacy_load.return_value = self.mock_nlp

        # Mock Matcher separately if its direct instantiation or methods are used outside nlp pipeline
        self.patcher_matcher = patch('spacy.matcher.Matcher')
        self.mock_matcher_class = self.patcher_matcher.start()
        self.mock_matcher_instance = MagicMock()
        self.mock_matcher_class.return_value = self.mock_matcher_instance

        # Instantiate the ContextAgent
        # We pass the real config, but spacy.load is mocked.
        self.agent = ContextAgent(
            llm_service=self.mock_llm_service,
            memory_system=self.mock_memory_system,
            agent_config=self.agent_config,
            state_manager=self.mock_state_manager
        )
        # Ensure the mocked nlp is assigned if spacy.load was called in init
        self.agent.nlp = self.mock_nlp
        self.agent.matcher = self.mock_matcher_instance

    def tearDown(self):
        """Clean up resources after tests."""
        self.patcher_spacy_load.stop()
        self.patcher_matcher.stop()

    # 1. Initialization Tests
    def test_initialization_with_config(self):
        self.assertEqual(self.agent.agent_config, self.agent_config)
        self.mock_spacy_load.assert_called_once_with(self.agent_config.spacy_model_name)
        self.assertIsNotNone(self.agent.nlp)
        self.assertIsNotNone(self.agent.matcher)
        # Check if _initialize_matchers was called (indirectly, by checking if add was called on matcher)
        # This assumes _initialize_matchers calls self.matcher.add
        self.assertTrue(self.mock_matcher_instance.add.called, "Matcher.add should have been called during initialization.")

    @patch('spacy.load')
    def test_initialization_spacy_load_error(self, mock_spacy_load_error):
        mock_spacy_load_error.side_effect = Exception("spaCy model loading failed")

        agent = ContextAgent(
            llm_service=self.mock_llm_service,
            memory_system=self.mock_memory_system,
            agent_config=self.agent_config,
            state_manager=self.mock_state_manager
        )
        self.assertIsNone(agent.nlp)
        self.assertIsNone(agent.matcher)
        # In a real scenario, you might check for a log message here
        # self.mock_logger.error.assert_called_with(...)

    # 2. Input Processing Tests (process_input)
    def test_process_input_string(self):
        raw_input = "  Test Input String  "
        expected_output = "test input string"
        processed_input = self.agent.process_input(raw_input)
        self.assertEqual(processed_input, expected_output)

    def test_process_input_non_string(self):
        raw_input = 12345
        expected_output = "12345"
        processed_input = self.agent.process_input(raw_input)
        self.assertEqual(processed_input, expected_output)

    def test_process_input_empty(self):
        raw_input = ""
        expected_output = ""
        processed_input = self.agent.process_input(raw_input)
        self.assertEqual(processed_input, expected_output)

    def test_process_input_none(self):
        raw_input = None
        # According to current implementation, str(None) is "none"
        expected_output = "none"
        processed_input = self.agent.process_input(raw_input)
        self.assertEqual(processed_input, expected_output)


    @patch('contextkernel.core_logic.context_agent.ContextAgent.process_input', side_effect=Exception("Conversion error"))
    def test_process_input_conversion_error_direct_mock(self, mock_process_input_method):
        # This test mocks the method itself to simulate an internal error during processing
        # More targeted would be to mock `str()` if that's the specific failure point.
        # However, the current implementation's try/except is broad.
        raw_input = MagicMock() # An object that might cause str() to fail

        # To test the str() conversion error, we'd need to make raw_input.__str__ raise error
        # For this setup, let's assume the error happens within the method's try block
        # The current method structure returns "" on any exception.

        # Re-instantiate agent for this specific test if setUp's agent is too broad
        # Or, ensure the mocked method is on the instance
        self.agent.process_input = mock_process_input_method

        result = self.agent.process_input(raw_input)
        self.assertEqual(result, "")
        # Add log check here if possible: self.agent.logger.error.assert_called_once()

    # 3. Intent Detection Tests (detect_intent)
    async def test_detect_intent_spacy_match_search(self):
        processed_input = "search for cats"
        mock_doc = MagicMock()
        self.mock_nlp.return_value = mock_doc # nlp(text) returns doc

        # Simulate a matcher result
        # match_id, start_token, end_token
        mock_match = (self.mock_nlp.vocab.strings.add("search_info"), 0, 3)
        self.mock_matcher_instance.return_value = [mock_match]

        # Mocking the span and its text attribute
        mock_span = MagicMock()
        mock_span.text = "search for cats"
        mock_doc.__getitem__.return_value = mock_span # doc[start:end]

        # Mocking token-level details for entity extraction if needed by the method
        mock_token_search = MagicMock(); mock_token_search.lower_ = "search"
        mock_token_for = MagicMock(); mock_token_for.lower_ = "for"
        mock_token_cats = MagicMock(); mock_token_cats.lower_ = "cats"
        # Simulate doc[start], doc[start+1] etc.
        def doc_getitem_side_effect(key):
            if isinstance(key, slice): # For spans like doc[0:3]
                return mock_span
            # For individual tokens like doc[0]
            token_map = {0: mock_token_search, 1: mock_token_for, 2: mock_token_cats}
            return token_map.get(key, MagicMock()) # Return a generic mock if key not in map
        mock_doc.__getitem__.side_effect = doc_getitem_side_effect

        # For entity extraction: doc[keyword_end_token_index:end].text
        mock_entity_span = MagicMock()
        mock_entity_span.text = "cats"
        # This needs to be more dynamic based on actual slicing in detect_intent
        # For "search for cats", keyword_end_token_index is 1 (after "search")
        # So, doc[1:3] should be the entity span
        def dynamic_entity_span_side_effect(key):
            if key.start == 1 and key.stop == 3: # Specific to "search for cats"
                return mock_entity_span
            return mock_span # Default fallback
        mock_doc.__getitem__.side_effect = dynamic_entity_span_side_effect


        result = await self.agent.detect_intent(processed_input)

        self.assertEqual(result.intent, "search_info")
        self.agent.logger.debug(f"Search entities: {result.entities}")
        self.assertEqual(result.entities.get("query"), "cats")
        self.assertEqual(result.confidence, self.agent_config.high_confidence_threshold)
        self.assertEqual(result.original_input, processed_input)
        self.assertIsNotNone(result.spacy_doc)
        self.assertTrue(len(result.matched_patterns) > 0)
        self.assertEqual(result.matched_patterns[0]["pattern_name"], "search_info")
        self.assertEqual(result.matched_patterns[0]["matched_text"], "search for cats")

    async def test_detect_intent_spacy_match_save(self):
        processed_input = "remember to buy milk"
        mock_doc = MagicMock(name="doc_save")
        self.mock_nlp.return_value = mock_doc

        mock_match = (self.mock_nlp.vocab.strings.add("save_info"), 0, 4) # "remember to buy milk"
        self.mock_matcher_instance.return_value = [mock_match]

        mock_span_full = MagicMock(name="span_full_save")
        mock_span_full.text = "remember to buy milk"

        mock_token_remember = MagicMock(lower_="remember")
        mock_token_to = MagicMock(lower_="to")
        mock_token_buy = MagicMock(lower_="buy")
        mock_token_milk = MagicMock(lower_="milk")

        mock_entity_span = MagicMock(name="entity_span_save")
        mock_entity_span.text = "to buy milk"

        def doc_getitem_side_effect_save(key):
            if isinstance(key, slice):
                if key.start == 0 and key.stop == 4: # Full span
                    return mock_span_full
                elif key.start == 1 and key.stop == 4: # Entity span "to buy milk"
                    return mock_entity_span
                return MagicMock(text="some slice text") # Default for other slices
            # For individual tokens
            token_map = {0: mock_token_remember, 1: mock_token_to, 2: mock_token_buy, 3: mock_token_milk}
            return token_map.get(key, MagicMock())
        mock_doc.__getitem__.side_effect = doc_getitem_side_effect_save

        result = await self.agent.detect_intent(processed_input)

        self.assertEqual(result.intent, "save_info")
        self.assertEqual(result.entities.get("data"), "to buy milk")
        self.assertEqual(result.confidence, self.agent_config.high_confidence_threshold)
        self.assertEqual(result.original_input, processed_input)

    async def test_detect_intent_spacy_match_summarize(self):
        processed_input = "summarize the article about birds"
        mock_doc = MagicMock(name="doc_summarize")
        self.mock_nlp.return_value = mock_doc

        # "summarize the article about birds" -> summarize:0, the:1, article:2, about:3, birds:4
        mock_match = (self.mock_nlp.vocab.strings.add("summarization_intent"), 0, 5)
        self.mock_matcher_instance.return_value = [mock_match]

        mock_span_full = MagicMock(name="span_full_summarize")
        mock_span_full.text = "summarize the article about birds"

        mock_token_summarize = MagicMock(lower_="summarize")
        # ... other tokens can be mocked if needed for more complex logic

        mock_entity_span = MagicMock(name="entity_span_summarize")
        mock_entity_span.text = "the article about birds"

        def doc_getitem_side_effect_summarize(key):
            if isinstance(key, slice):
                if key.start == 0 and key.stop == 5: return mock_span_full # Full span
                elif key.start == 1 and key.stop == 5: return mock_entity_span # Entity span
                return MagicMock(text="some slice text")
            if key == 0: return mock_token_summarize
            return MagicMock() # Default for other tokens
        mock_doc.__getitem__.side_effect = doc_getitem_side_effect_summarize

        result = await self.agent.detect_intent(processed_input)

        self.assertEqual(result.intent, "summarization_intent")
        self.assertEqual(result.entities.get("text_to_summarize"), "the article about birds")
        self.assertEqual(result.confidence, self.agent_config.high_confidence_threshold)

    async def test_detect_intent_no_spacy_match_fallback_keyword(self):
        processed_input = "can you look up status of order 123" # Assume no spaCy rule matches this
        mock_doc = MagicMock()
        self.mock_nlp.return_value = mock_doc
        self.mock_matcher_instance.return_value = [] # No matches from Matcher

        # Fallback logic in detect_intent uses simple "in" check
        # "search" is not in "can you look up status of order 123"
        # "save" is not in "can you look up status of order 123"
        # "summarize" is not in "can you look up status of order 123"
        # So it should become "unknown_intent"

        result = await self.agent.detect_intent(processed_input)

        self.assertEqual(result.intent, "unknown_intent")
        self.assertEqual(result.confidence, self.agent_config.default_intent_confidence)
        self.assertEqual(result.original_input, processed_input)

    async def test_detect_intent_no_spacy_match_fallback_keyword_search(self):
        # Test fallback to keyword "search" when spaCy match fails
        processed_input = "i want to search for books on AI"
        mock_doc = MagicMock()
        self.mock_nlp.return_value = mock_doc
        self.mock_matcher_instance.return_value = [] # No spaCy matches

        result = await self.agent.detect_intent(processed_input)

        self.assertEqual(result.intent, "search_info") # Fallback to keyword
        self.assertEqual(result.confidence, self.agent_config.default_intent_confidence)


    async def test_detect_intent_spacy_load_failure_fallback(self):
        self.agent.nlp = None # Simulate spaCy load failure
        self.agent.matcher = None
        processed_input = "search for weather"

        result = await self.agent.detect_intent(processed_input)

        self.assertEqual(result.intent, "search_info") # Basic keyword fallback
        self.assertEqual(result.entities.get("error_message"), "spaCy model not available")
        self.assertEqual(result.confidence, self.agent_config.default_intent_confidence)

    async def test_detect_intent_spacy_processing_error(self):
        processed_input = "some troublesome input"
        self.mock_nlp.return_value = MagicMock()
        self.mock_matcher_instance.side_effect = Exception("Matcher failed spectacularly")

        result = await self.agent.detect_intent(processed_input)

        self.assertEqual(result.intent, "intent_detection_error")
        self.assertIn("Matcher failed spectacularly", result.entities.get("error_message", ""))
        self.assertEqual(result.confidence, 0.0)


    # 4. Routing Decision Tests (decide_route)
    def test_decide_route_intent_detection_error(self):
        intent_res = IntentExtractionResult(
            intent="intent_detection_error",
            entities={"error_message": "Failed!"},
            confidence=0.0,
            original_input="test"
        )
        routing_decision = self.agent.decide_route(intent_res)
        self.assertEqual(routing_decision.target_module, "ErrorHandler")
        self.assertIn("Intent detection failed", routing_decision.task_parameters.get("error_message", ""))

    def test_decide_route_low_confidence(self):
        intent_res = IntentExtractionResult(
            intent="search_info",
            entities={"query": "something"},
            confidence=self.agent_config.low_confidence_threshold - 0.1, # Below threshold
            original_input="search something"
        )
        routing_decision = self.agent.decide_route(intent_res)
        self.assertEqual(routing_decision.target_module, "ErrorHandler")
        self.assertIn("Intent unclear or confidence too low", routing_decision.task_parameters.get("error_message", ""))

    def test_decide_route_search_info_sufficient_confidence(self):
        intent_res = IntentExtractionResult(
            intent="search_info",
            entities={"query": "weather in london"},
            confidence=self.agent_config.high_confidence_threshold, # Sufficient
            original_input="search weather in london"
        )
        routing_decision = self.agent.decide_route(intent_res)
        self.assertEqual(routing_decision.target_module, "LLMRetriever")
        self.assertEqual(routing_decision.task_parameters.get("query"), "weather in london")

    def test_decide_route_save_info_sufficient_confidence(self):
        intent_res = IntentExtractionResult(
            intent="save_info",
            entities={"data": "my note"},
            confidence=self.agent_config.high_confidence_threshold,
            original_input="save my note"
        )
        routing_decision = self.agent.decide_route(intent_res)
        self.assertEqual(routing_decision.target_module, "LLMListener")
        self.assertEqual(routing_decision.task_parameters.get("data"), "my note")

    def test_decide_route_summarization_intent_sufficient_confidence(self):
        intent_res = IntentExtractionResult(
            intent="summarization_intent",
            entities={"text_to_summarize": "long article text..."},
            confidence=self.agent_config.high_confidence_threshold,
            original_input="summarize long article text..."
        )
        routing_decision = self.agent.decide_route(intent_res)
        self.assertEqual(routing_decision.target_module, "LLMListener")
        self.assertEqual(routing_decision.task_parameters.get("raw_data"), "long article text...")
        self.assertIn("process_for_summarization", routing_decision.task_parameters.get("context_instructions", {}))

    def test_decide_route_unknown_intent(self):
        intent_res = IntentExtractionResult(
            intent="unknown_intent",
            entities={},
            confidence=self.agent_config.default_intent_confidence, # Assume it passed confidence check somehow
            original_input="gibberish input"
        )
        routing_decision = self.agent.decide_route(intent_res)
        self.assertEqual(routing_decision.target_module, "ErrorHandler")
        self.assertIn("Unknown intent detected", routing_decision.task_parameters.get("error_message", ""))

    def test_decide_route_unrecognized_high_confidence_intent(self):
        intent_res = IntentExtractionResult(
            intent="some_custom_future_intent", # Not one of the known ones
            entities={"detail": "abc"},
            confidence=self.agent_config.high_confidence_threshold,
            original_input="do custom abc"
        )
        routing_decision = self.agent.decide_route(intent_res)
        self.assertEqual(routing_decision.target_module, "ErrorHandler")
        self.assertIn("Unrecognized high-confidence intent string: some_custom_future_intent", routing_decision.task_parameters.get("error_message", ""))

    # 5. Task Dispatching Tests (dispatch_task)
    async def test_dispatch_task_llm_retriever_success(self):
        route = RoutingDecision(
            target_module="LLMRetriever",
            task_parameters={"query": "what is AI?"},
            original_intent=MagicMock() # Mocked original intent
        )
        self.mock_llm_service.retrieve.return_value = "AI is artificial intelligence." # Successful async result

        task_result = await self.agent.dispatch_task(route)

        self.mock_llm_service.retrieve.assert_called_once_with(query="what is AI?")
        self.assertEqual(task_result.status, "success")
        self.assertEqual(task_result.data, "AI is artificial intelligence.")
        self.assertEqual(task_result.message, "LLMRetriever processed successfully.")

    async def test_dispatch_task_llm_retriever_missing_query(self):
        route = RoutingDecision(
            target_module="LLMRetriever",
            task_parameters={"some_other_param": "value"}, # Query is missing
            original_intent=MagicMock()
        )
        task_result = await self.agent.dispatch_task(route)
        self.assertEqual(task_result.status, "error")
        self.assertIn("Missing 'query' parameter for LLMRetriever", task_result.message)
        self.mock_llm_service.retrieve.assert_not_called()

    async def test_dispatch_task_llm_retriever_service_exception(self):
        route = RoutingDecision(
            target_module="LLMRetriever",
            task_parameters={"query": "what is AI?"},
            original_intent=MagicMock()
        )
        self.mock_llm_service.retrieve.side_effect = Exception("LLM service unavailable")

        task_result = await self.agent.dispatch_task(route)

        self.mock_llm_service.retrieve.assert_called_once_with(query="what is AI?")
        self.assertEqual(task_result.status, "error")
        self.assertIn("Error during LLMRetriever execution: LLM service unavailable", task_result.message)
        self.assertIn("LLM service unavailable", task_result.error_details.get("exception_message"))

    async def test_dispatch_task_llm_listener_success(self):
        route = RoutingDecision(
            target_module="LLMListener",
            task_parameters={"raw_data": "some text to process", "context_instructions": {"do_x": True}},
            original_intent=MagicMock()
        )
        self.mock_llm_service.process.return_value = "Processing complete."

        task_result = await self.agent.dispatch_task(route)

        self.mock_llm_service.process.assert_called_once_with(
            data="some text to process",
            context_instructions={"do_x": True}
        )
        self.assertEqual(task_result.status, "success")
        self.assertEqual(task_result.data, "Processing complete.")
        self.assertEqual(task_result.message, "LLMListener processed successfully.")

    async def test_dispatch_task_llm_listener_service_exception(self):
        route = RoutingDecision(
            target_module="LLMListener",
            task_parameters={"raw_data": "some text"},
            original_intent=MagicMock()
        )
        self.mock_llm_service.process.side_effect = Exception("LLM processing error")

        task_result = await self.agent.dispatch_task(route)

        self.mock_llm_service.process.assert_called_once_with(data="some text", context_instructions=None)
        self.assertEqual(task_result.status, "error")
        self.assertIn("Error during LLMListener execution: LLM processing error", task_result.message)
        self.assertIn("LLM processing error", task_result.error_details.get("exception_message"))

    async def test_dispatch_task_llm_service_none(self):
        self.agent.llm_service = None
        route = RoutingDecision(target_module="LLMRetriever", task_parameters={"query": "test"})
        task_result = await self.agent.dispatch_task(route)
        self.assertEqual(task_result.status, "error")
        self.assertIn("LLMRetriever service not configured", task_result.message)

    async def test_dispatch_task_llm_service_method_missing(self):
        self.agent.llm_service = MagicMock() # Fresh mock without 'retrieve' or 'process'
        del self.agent.llm_service.retrieve # Ensure method is missing

        route = RoutingDecision(target_module="LLMRetriever", task_parameters={"query": "test"})
        task_result = await self.agent.dispatch_task(route)
        self.assertEqual(task_result.status, "error")
        self.assertIn("LLMRetriever service method 'retrieve' not found", task_result.message)

    async def test_dispatch_task_error_handler(self):
        error_info = {"error_message": "Something went very wrong.", "code": 500}
        route = RoutingDecision(
            target_module="ErrorHandler",
            task_parameters=error_info, # ErrorHandler expects the error details in task_parameters
            original_intent=MagicMock()
        )
        task_result = await self.agent.dispatch_task(route)
        self.assertEqual(task_result.status, "error")
        self.assertEqual(task_result.message, "Something went very wrong.")
        self.assertEqual(task_result.error_details, error_info)

    async def test_dispatch_task_unknown_module(self):
        route = RoutingDecision(
            target_module="UnknownModuleXYZ",
            task_parameters={"data": "test"},
            original_intent=MagicMock()
        )
        task_result = await self.agent.dispatch_task(route)
        self.assertEqual(task_result.status, "error")
        self.assertEqual(task_result.message, "Unknown module: UnknownModuleXYZ")
        self.assertEqual(task_result.error_details["reason"], "Unknown target module: UnknownModuleXYZ")

    # 6. Overall Request Handling Tests (handle_request)
    @patch.object(ContextAgent, 'process_input', new_callable=MagicMock)
    @patch.object(ContextAgent, 'detect_intent', new_callable=MagicMock)
    @patch.object(ContextAgent, 'decide_route', new_callable=MagicMock)
    @patch.object(ContextAgent, 'dispatch_task', new_callable=MagicMock)
    async def test_handle_request_successful_flow(
        self, mock_dispatch_task, mock_decide_route, mock_detect_intent, mock_process_input
    ):
        raw_input_text = "Search for news about AI"
        processed_text = "search for news about ai"
        intent_result_mock = IntentExtractionResult(
            intent="search_info",
            entities={"query": "news about ai"},
            confidence=0.9,
            original_input=processed_text
        )
        routing_decision_mock = RoutingDecision(
            target_module="LLMRetriever",
            task_parameters={"query": "news about ai"},
            original_intent=intent_result_mock
        )
        final_task_result_mock = TaskResult(
            status="success",
            data="AI news data here",
            message="LLMRetriever processed successfully."
        )

        # Configure mocks
        mock_process_input.return_value = processed_text
        mock_detect_intent.return_value = intent_result_mock
        mock_decide_route.return_value = routing_decision_mock
        mock_dispatch_task.return_value = final_task_result_mock

        # Call the method
        final_result = await self.agent.handle_request(raw_input_text, conversation_id="conv123")

        # Assertions
        mock_process_input.assert_called_once_with(raw_input_text)
        mock_detect_intent.assert_called_once_with(processed_text)
        mock_decide_route.assert_called_once_with(intent_result_mock)
        mock_dispatch_task.assert_called_once_with(routing_decision_mock)
        self.assertEqual(final_result, final_task_result_mock)
        self.assertEqual(final_result.status, "success")
        self.assertEqual(final_result.data, "AI news data here")

    @patch.object(ContextAgent, 'process_input', new_callable=MagicMock)
    @patch.object(ContextAgent, 'detect_intent', new_callable=MagicMock)
    @patch.object(ContextAgent, 'decide_route', new_callable=MagicMock)
    @patch.object(ContextAgent, 'dispatch_task', new_callable=MagicMock)
    async def test_handle_request_intent_detection_failure(
        self, mock_dispatch_task, mock_decide_route, mock_detect_intent, mock_process_input
    ):
        raw_input_text = "gibberish that fails intent"
        processed_text = "gibberish that fails intent"
        intent_error_result_mock = IntentExtractionResult(
            intent="intent_detection_error",
            entities={"error_message": "spaCy error"},
            confidence=0.0,
            original_input=processed_text
        )
        # Route decision when intent detection fails (usually to ErrorHandler)
        routing_to_error_handler_mock = RoutingDecision(
            target_module="ErrorHandler",
            task_parameters={"error_message": "Intent detection failed.", "original_intent_info": intent_error_result_mock.dict()},
            original_intent=intent_error_result_mock
        )
        # Task result from ErrorHandler
        error_handler_task_result_mock = TaskResult(
            status="error",
            message="Intent detection failed.",
            error_details=routing_to_error_handler_mock.task_parameters
        )

        # Configure mocks
        mock_process_input.return_value = processed_text
        mock_detect_intent.return_value = intent_error_result_mock
        mock_decide_route.return_value = routing_to_error_handler_mock
        mock_dispatch_task.return_value = error_handler_task_result_mock

        # Call the method
        final_result = await self.agent.handle_request(raw_input_text)

        # Assertions
        mock_process_input.assert_called_once_with(raw_input_text)
        mock_detect_intent.assert_called_once_with(processed_text)
        mock_decide_route.assert_called_once_with(intent_error_result_mock)
        mock_dispatch_task.assert_called_once_with(routing_to_error_handler_mock)
        self.assertEqual(final_result, error_handler_task_result_mock)
        self.assertEqual(final_result.status, "error")
        self.assertIn("Intent detection failed", final_result.message)

    @patch('contextkernel.core_logic.context_agent.ContextAgent.process_input', side_effect=Exception("Input processing critical error"))
    async def test_handle_request_critical_failure_in_process_input(self, mock_process_input_error):
        # This tests a scenario where process_input itself raises an unhandled exception
        # which should be caught by the outermost try-except in handle_request.
        self.agent.process_input = mock_process_input_error # Assign the problematic mock

        raw_input_text = "some input"
        final_result = await self.agent.handle_request(raw_input_text)

        self.assertEqual(final_result.status, "error")
        self.assertIn("Critical error during request handling: Input processing critical error", final_result.message)
        self.assertEqual(final_result.error_details["exception_type"], "Exception")


if __name__ == '__main__':
    unittest.main()
