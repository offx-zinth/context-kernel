import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Modules to test
from contextkernel.core_logic.context_agent import (
    ContextAgentConfig,
    ContextAgent,
    IntentExtractionResult,
    RoutingDecision,
    TaskResult
)
from contextkernel.utils.state_manager import InMemoryStateManager, RedisStateManager, AbstractStateManager


# --- Global Mocks ---
@pytest.fixture(autouse=True)
def mock_external_libs_for_agent(monkeypatch):
    """Mocks external libraries used by ContextAgent globally."""

    # Mock Hugging Face transformers.pipeline
    mock_pipeline_instance = MagicMock()
    # Default behavior for zero-shot classification
    mock_pipeline_instance.return_value = {
        'sequence': 'test input',
        'labels': ['search information'],
        'scores': [0.9]
    }
    mock_hf_pipeline_constructor = MagicMock(return_value=mock_pipeline_instance)
    monkeypatch.setattr("contextkernel.core_logic.context_agent.pipeline", mock_hf_pipeline_constructor)

    # Mock spaCy
    mock_spacy_nlp_instance = MagicMock()
    mock_spacy_doc_instance = MagicMock()
    mock_spacy_doc_instance.ents = [] # No entities by default
    mock_spacy_nlp_instance.return_value = mock_spacy_doc_instance # nlp("text") returns a Doc
    mock_spacy_load = MagicMock(return_value=mock_spacy_nlp_instance)
    monkeypatch.setattr("spacy.load", mock_spacy_load)

    # Mock Matcher (though its methods are usually called on the instance)
    # We need to ensure that Matcher can be instantiated.
    mock_matcher_instance = MagicMock()
    mock_matcher_constructor = MagicMock(return_value=mock_matcher_instance)
    monkeypatch.setattr("spacy.matcher.Matcher", mock_matcher_constructor)


    # Mock RedisStateManager's dependency if Redis is chosen
    mock_redis_instance_for_agent = AsyncMock()
    mock_aioredis_from_url = MagicMock(return_value=mock_redis_instance_for_agent)
    # Patch where aioredis is imported in state_manager.py
    monkeypatch.setattr("contextkernel.utils.state_manager.aioredis.from_url", mock_aioredis_from_url)


# --- Fixtures for ContextAgent Tests ---
@pytest.fixture
def default_agent_config():
    """Returns a default ContextAgentConfig."""
    return ContextAgentConfig(
        intent_classifier_model="mock-intent-classifier", # Ensure it uses mocked pipeline
        spacy_model_name="mock-spacy-model" # Ensure it uses mocked spacy
    )

@pytest.fixture
def mock_llm_service():
    """Mocks the LLMService, expecting retriever and listener components."""
    service = MagicMock()
    service.retriever = AsyncMock(spec_set=['retrieve']) # Mock the retriever component with specific methods
    service.listener = AsyncMock(spec_set=['process_data'])  # Mock the listener component

    # Set up default return values for the mocked methods
    service.retriever.retrieve = AsyncMock(return_value=TaskResult(status="success", data={"retrieved": "data"}))
    service.listener.process_data = AsyncMock(return_value=None)
    return service

@pytest.fixture
def mock_memory_system():
    """Mocks the MemorySystem (not directly used by ContextAgent but passed)."""
    return MagicMock()

@pytest.fixture
def mock_state_manager_fixture(monkeypatch): # Renamed to avoid conflict with class name
    """Mocks a generic AbstractStateManager and patches its constructor calls if needed."""
    manager_instance = AsyncMock(spec=AbstractStateManager)
    manager_instance.get_state = AsyncMock(return_value=None)
    manager_instance.save_state = AsyncMock()
    manager_instance.delete_state = AsyncMock()

    # If ContextAgent instantiates a StateManager directly, you might need to patch those specific classes
    # For now, this fixture provides an instance that can be injected if ContextAgent takes one.
    return manager_instance


class TestContextAgentInit:
    def test_init_in_memory_state_manager(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch):
        """Tests initialization with default InMemoryStateManager."""
        default_agent_config.state_manager_type = "in_memory"

        # Constructors are mocked by the autouse fixture 'mock_external_libs_for_agent'
        hf_pipeline_constructor_mock = contextkernel.core_logic.context_agent.pipeline
        spacy_load_mock = spacy.load

        agent = ContextAgent(
            llm_service=mock_llm_service,
            memory_system=mock_memory_system,
            agent_config=default_agent_config
            # state_manager is not injected, so it should create one.
        )
        assert isinstance(agent.state_manager, InMemoryStateManager)
        assert agent.intent_classifier is not None
        assert agent.nlp is not None
        spacy_load_mock.assert_called_once_with(default_agent_config.spacy_model_name)
        hf_pipeline_constructor_mock.assert_called_with(
            "zero-shot-classification",
            model=default_agent_config.intent_classifier_model
        )

    def test_init_redis_state_manager_success(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch):
        """Tests initialization with RedisStateManager."""
        default_agent_config.state_manager_type = "redis"
        default_agent_config.redis_host = "testredis"
        default_agent_config.redis_port = 1234

        aioredis_from_url_mock = contextkernel.utils.state_manager.aioredis.from_url # From autouse fixture
        spacy_load_mock = spacy.load # From autouse fixture

        agent = ContextAgent(
            llm_service=mock_llm_service,
            memory_system=mock_memory_system,
            agent_config=default_agent_config
        )
        assert isinstance(agent.state_manager, RedisStateManager)
        aioredis_from_url_mock.assert_called_once_with(f"redis://{default_agent_config.redis_host}:{default_agent_config.redis_port}/0")
        spacy_load_mock.assert_called_with(default_agent_config.spacy_model_name)


    def test_init_redis_state_manager_import_error_fallback(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch, caplog):
        """Tests fallback to InMemoryStateManager if Redis client fails to import."""
        default_agent_config.state_manager_type = "redis"
        monkeypatch.setattr("contextkernel.utils.state_manager.aioredis", None) # Specifically break aioredis for this test

        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        assert isinstance(agent.state_manager, InMemoryStateManager)
        assert "Failed to initialize RedisStateManager due to ImportError" in caplog.text

    def test_init_intent_classifier_failure(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch, caplog):
        """Tests that ContextAgent handles intent classifier loading failure."""
        # The global mock_hf_pipeline_constructor is already set up by mock_external_libs_for_agent.
        # We need to make its side_effect specific for this test.
        hf_pipeline_constructor_mock = contextkernel.core_logic.context_agent.pipeline
        hf_pipeline_constructor_mock.side_effect = Exception("Classifier load failed!")

        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        assert agent.intent_classifier is None
        assert "Failed to load intent classifier model" in caplog.text
        assert "Classifier load failed!" in caplog.text

    def test_init_spacy_failure(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch, caplog):
        """Tests that ContextAgent handles spaCy loading failure."""
        spacy_load_mock = MagicMock(side_effect=Exception("spaCy load failed!"))
        monkeypatch.setattr("spacy.load", spacy_load_mock)

        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        assert agent.nlp is None
        assert agent.matcher is None # Matcher initialization depends on nlp.vocab
        assert "Error loading spaCy model" in caplog.text
        assert "spaCy load failed!" in caplog.text

    def test_init_with_injected_state_manager(self, default_agent_config, mock_llm_service, mock_memory_system, mock_state_manager_fixture):
        """Tests initialization with an injected StateManager."""
        agent = ContextAgent(
            llm_service=mock_llm_service,
            memory_system=mock_memory_system,
            agent_config=default_agent_config,
            state_manager=mock_state_manager_fixture # Inject the mock
        )
        assert agent.state_manager == mock_state_manager_fixture


class TestContextAgentDetectIntent:
    @pytest.mark.asyncio
    async def test_detect_intent_spacy_matcher_first_success(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch):
        """Tests intent detection using spaCy Matcher as primary."""
        default_agent_config.use_spacy_matcher_first = True

        mock_nlp_instance = MagicMock()
        mock_doc_instance = MagicMock()
        mock_doc_instance.ents = [MagicMock(text="matched_entity_text", label_="TEST_ENTITY")]
        mock_nlp_instance.return_value = mock_doc_instance

        mock_matcher_instance = MagicMock()
        # Simulate a spaCy match: (match_id, start_token, end_token)
        # map 'search_info_id' to "search_info" string
        search_info_id = 12345

        # spaCy.load is mocked by autouse fixture. Get the nlp_instance it returns.
        nlp_instance_mock = spacy.load.return_value # type: ignore
        nlp_instance_mock.vocab.strings.__getitem__.side_effect = lambda x: "search_info" if x == search_info_id else str(x)

        matcher_instance_mock = spacy.matcher.Matcher.return_value # type: ignore
        matcher_instance_mock.return_value = [(search_info_id, 0, 2)] # Simulate a match

        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        # agent.nlp and agent.matcher will be set up using the globally mocked instances

        processed_input = "search for context kernel"
        # To make keyword_end_token_index work as expected for this test:
        mock_doc_instance.__getitem__.side_effect = lambda x: MagicMock(lower_="search") if x==0 else MagicMock(lower_="for")

        intent_result = await agent.detect_intent(processed_input)

        assert intent_result.intent == "search_info"
        assert intent_result.confidence == default_agent_config.high_confidence_threshold
        assert "query" in intent_result.entities # Based on simple entity logic in detect_intent
        assert "test_entity" in intent_result.entities # From doc.ents
        assert intent_result.entities["test_entity"] == "matched_entity_text"
        assert len(intent_result.matched_patterns) == 1
        assert intent_result.matched_patterns[0]["pattern_name"] == "search_info"
        # The zero-shot classifier should not have been called if spaCy match is high-confidence
        if agent.intent_classifier: # intent_classifier can be None if model load fails
             agent.intent_classifier.assert_not_called()


    @pytest.mark.asyncio
    async def test_detect_intent_zero_shot_classifier(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch):
        """Tests intent detection using zero-shot classifier."""
        default_agent_config.use_spacy_matcher_first = False # Force zero-shot

        mock_nlp_instance = MagicMock()
        mock_doc_instance = MagicMock()
        mock_doc_instance.ents = [MagicMock(text="entity from spacy", label_="ORG")]

        # The HF pipeline constructor is mocked by autouse fixture. Get its return_value (the pipeline instance).
        hf_pipeline_instance_mock = contextkernel.core_logic.context_agent.pipeline.return_value
        hf_pipeline_instance_mock.return_value = { # Simulate zero-shot output
            'sequence': 'some query about summarization',
            'labels': ['summarize text', 'search information'],
            'scores': [0.95, 0.05]
        }

        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        # agent.nlp is already the globally mocked one

        processed_input = "summarize the document for me"
        intent_result = await agent.detect_intent(processed_input)

        assert intent_result.intent == "summarize text"
        assert intent_result.confidence == 0.95
        assert "org" in intent_result.entities # From doc.ents
        assert intent_result.entities["org"] == "entity from spacy"
        hf_pipeline_mock.return_value.assert_called_once_with(
            processed_input, default_agent_config.intent_candidate_labels, multi_label=False
        )

    @pytest.mark.asyncio
    async def test_detect_intent_spacy_no_match_fallback_to_zero_shot(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch):
        """Tests fallback to zero-shot if spaCy Matcher finds no matches."""
        default_agent_config.use_spacy_matcher_first = True

        mock_nlp_instance = MagicMock()
        mock_doc_instance = MagicMock()
        mock_doc_instance.ents = []

        matcher_instance_mock = spacy.matcher.Matcher.return_value # type: ignore
        matcher_instance_mock.return_value = [] # No spaCy matches

        hf_pipeline_instance_mock = contextkernel.core_logic.context_agent.pipeline.return_value
        hf_pipeline_instance_mock.return_value = {'labels': ['general question'], 'scores': [0.88]}

        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        # agent.nlp and agent.matcher are from global mocks

        processed_input = "tell me a joke"
        intent_result = await agent.detect_intent(processed_input)

        assert intent_result.intent == "general question"
        assert intent_result.confidence == 0.88
        mock_matcher_instance.assert_called_once()
        mock_classifier_instance.assert_called_once()


    @pytest.mark.asyncio
    async def test_detect_intent_no_spacy_model(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch, caplog):
        """Tests behavior when spaCy model is not available."""
        monkeypatch.setattr("spacy.load", MagicMock(side_effect=OSError("spaCy model not found")))

        # Agent init will log error, nlp will be None
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        assert agent.nlp is None

        processed_input = "search for something"
        intent_result = await agent.detect_intent(processed_input)

        assert "spaCy nlp model not initialized" in caplog.text
        assert intent_result.intent == "search_info" # Basic keyword fallback
        assert intent_result.entities == {"error": "spaCy model unavailable"}
        assert intent_result.confidence == 0.1 # Default low confidence for this case


    @pytest.mark.asyncio
    async def test_detect_intent_no_classifier_and_no_spacy_match(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch, caplog):
        """Tests fallback when no spaCy match and classifier is unavailable."""
        default_agent_config.use_spacy_matcher_first = True

        mock_nlp_instance = MagicMock()
        mock_doc_instance = MagicMock()
        mock_doc_instance.ents = []
        mock_nlp_instance.return_value = mock_doc_instance
        monkeypatch.setattr("spacy.load", MagicMock(return_value=mock_nlp_instance))

        mock_matcher_instance = MagicMock(return_value=[]) # No spaCy matches
        monkeypatch.setattr("spacy.matcher.Matcher", MagicMock(return_value=mock_matcher_instance))

        # Simulate intent classifier not being available
        monkeypatch.setattr("contextkernel.core_logic.context_agent.pipeline", MagicMock(side_effect=Exception("Classifier init failed")))

        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        agent.matcher = mock_matcher_instance
        agent.nlp = mock_nlp_instance
        assert agent.intent_classifier is None # Ensure it failed to load

        processed_input = "a very generic statement"
        intent_result = await agent.detect_intent(processed_input)

        assert "No spaCy match and no intent classifier available" in caplog.text
        assert intent_result.intent == "general_question"
        assert intent_result.confidence == default_agent_config.default_intent_confidence


class TestContextAgentRoutingAndDispatch:
    def test_decide_route_search_info(self, default_agent_config, mock_llm_service, mock_memory_system):
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        intent_res = IntentExtractionResult(
            intent="search_info",
            entities={"query": "context kernel details"},
            confidence=0.9
        )
        routing_decision = agent.decide_route(intent_res)
        assert routing_decision.target_module == "LLMRetriever"
        assert routing_decision.task_parameters == {"query": "context kernel details"}

    def test_decide_route_save_info(self, default_agent_config, mock_llm_service, mock_memory_system):
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        intent_res = IntentExtractionResult(
            intent="save_info",
            entities={"data": "CK is cool"},
            confidence=0.9
        )
        routing_decision = agent.decide_route(intent_res)
        assert routing_decision.target_module == "LLMListener"
        assert routing_decision.task_parameters == {"data": "CK is cool"}

    def test_decide_route_summarize_text(self, default_agent_config, mock_llm_service, mock_memory_system):
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        intent_res = IntentExtractionResult(
            intent="summarization_intent",
            entities={"text_to_summarize": "A long document..."},
            confidence=0.9
        )
        routing_decision = agent.decide_route(intent_res)
        assert routing_decision.target_module == "LLMListener"
        assert routing_decision.task_parameters["raw_data"] == "A long document..."
        assert routing_decision.task_parameters["context_instructions"]["summarize"] is True

    def test_decide_route_low_confidence(self, default_agent_config, mock_llm_service, mock_memory_system):
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        intent_res = IntentExtractionResult(intent="search_info", confidence=0.4) # Below threshold
        routing_decision = agent.decide_route(intent_res)
        assert routing_decision.target_module == "ErrorHandler"
        assert "Intent unclear or confidence too low" in routing_decision.task_parameters["error_message"]

    @pytest.mark.asyncio
    async def test_dispatch_task_to_retriever(self, default_agent_config, mock_llm_service, mock_memory_system):
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        route = RoutingDecision(
            target_module="LLMRetriever",
            task_parameters={"query": "what is CK?", "top_k": 3}
        )
        task_result = await agent.dispatch_task(route)

        mock_llm_service.retriever.retrieve.assert_called_once_with(query="what is CK?", top_k=3)
        assert task_result.status == "success" # Based on mock_llm_service setup
        assert task_result.data == {"retrieved": "data"}

    @pytest.mark.asyncio
    async def test_dispatch_task_to_listener(self, default_agent_config, mock_llm_service, mock_memory_system):
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        route = RoutingDecision(
            target_module="LLMListener",
            task_parameters={"raw_data": "some text", "context_instructions": {"process": True}}
        )
        task_result = await agent.dispatch_task(route)

        mock_llm_service.listener.process_data.assert_called_once_with(
            raw_data="some text", context_instructions={"process": True}
        )
        assert task_result.status == "success" # Based on mock_llm_service setup

    @pytest.mark.asyncio
    async def test_dispatch_task_to_errorhandler(self, default_agent_config, mock_llm_service, mock_memory_system):
        agent = ContextAgent(default_agent_config, mock_llm_service, mock_memory_system)
        error_details = {"error_message": "Test error"}
        route = RoutingDecision(target_module="ErrorHandler", task_parameters=error_details)
        task_result = await agent.dispatch_task(route)

        assert task_result.status == "error"
        assert task_result.message == "Test error"
        assert task_result.error_details == error_details


class TestContextAgentHandleRequest:
    @pytest.mark.asyncio
    async def test_handle_request_e2e_with_state(self, default_agent_config, mock_llm_service, mock_memory_system, monkeypatch):
        """End-to-end test for handle_request including state management."""

        # Mock StateManager and inject it
        mock_state_mgr_instance = AsyncMock(spec=InMemoryStateManager) # Use spec of a concrete class
        mock_state_mgr_instance.get_state = AsyncMock(return_value={"previous_turn": "something"})
        mock_state_mgr_instance.save_state = AsyncMock()

        # We need to patch the constructor that ContextAgent calls if state_manager is not injected
        # OR inject the mock_state_mgr_instance. Let's try injection.

        agent = ContextAgent(
            llm_service=mock_llm_service,
            memory_system=mock_memory_system,
            agent_config=default_agent_config,
            state_manager=mock_state_mgr_instance # Injecting the mock
        )

        # Mock internal methods of ContextAgent for focused test
        agent.process_input = MagicMock(return_value="processed query: search for cars")
        mock_intent_result = IntentExtractionResult(intent="search_info", confidence=0.9, entities={"query":"cars"})
        agent.detect_intent = AsyncMock(return_value=mock_intent_result)

        mock_routing_decision = RoutingDecision(target_module="LLMRetriever", task_parameters={"query":"cars"})
        agent.decide_route = MagicMock(return_value=mock_routing_decision)

        # dispatch_task is already tested for retriever, relies on mock_llm_service
        # Here, we ensure it's called by handle_request
        # mock_llm_service.retriever.retrieve is already an AsyncMock returning success

        conversation_id = "conv123"
        raw_input = "Search for cars"
        current_context = {"user_id": "user1"}

        task_result = await agent.handle_request(raw_input, conversation_id, current_context)

        agent.process_input.assert_called_once_with(raw_input)
        mock_state_mgr_instance.get_state.assert_called_once_with(conversation_id)
        agent.detect_intent.assert_called_once_with("processed query: search for cars")
        agent.decide_route.assert_called_once_with(mock_intent_result)

        # Check that dispatch_task was called via the llm_service retriever mock
        mock_llm_service.retriever.retrieve.assert_called_once_with(query="cars")

        mock_state_mgr_instance.save_state.assert_called_once()
        # Inspect the state saved
        saved_state_args = mock_state_mgr_instance.save_state.call_args[0]
        assert saved_state_args[0] == conversation_id
        assert saved_state_args[1]["last_intent_detected"] == "search_info"
        assert saved_state_args[1]["last_task_status"] == "success" # from mock_llm_service

        assert task_result.status == "success"
        assert task_result.data == {"retrieved": "data"}
