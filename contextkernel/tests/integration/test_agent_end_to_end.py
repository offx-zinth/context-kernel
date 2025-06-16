# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from types import SimpleNamespace # For creating a facade llm_service object

# Configuration
from contextkernel.utils.config import load_config, AppConfig

# Core Logic Components
from contextkernel.core_logic.summarizer import Summarizer, SummarizerConfig
from contextkernel.core_logic.llm_retriever import LLMRetriever, HuggingFaceEmbeddingModel, StubLTM, StubGraphDB, RetrievalResponse
from contextkernel.core_logic.llm_listener import LLMListener, LLMListenerConfig
from contextkernel.core_logic.context_agent import ContextAgent, ContextAgentConfig, TaskResult

# Stubs for memory systems not fully implemented in retriever/listener stubs
# Re-using test stubs from listener tests for RawCache and STM
# Ensure these stubs are importable or define them here if simpler for standalone run.
# For this subtask, assuming they are made available in the test environment.
# If not, they might need to be copied into this file or a shared test utils.

# A simple version of stubs for this integration test:
class SimpleStubRawCache:
    async def store(self, doc_id: str, data: Any): return doc_id
    async def load(self, doc_id: str): return None

class SimpleStubSTM:
    async def save_summary(self, summary_id: str, summary_obj: Any, metadata: dict): pass
    async def load_summary(self, summary_id: str): return None


# State Management
from contextkernel.utils.state_manager import InMemoryStateManager

# Custom Exceptions (to ensure they are handled if they propagate)
from contextkernel.core_logic.exceptions import ConfigurationError


# Fixture to provide AppConfig
@pytest.fixture(scope="module")
def app_config_integration():
    # Using more specific names for integration test config to avoid .env conflicts if running locally
    # These should point to very small, fast models or be entirely mockable for CI.
    # For a true integration test, some real model interaction is good, but keep it minimal.

    # It's better to directly instantiate AppConfig with test-specific overrides
    # to avoid reliance on .env files during automated testing.
    cfg = AppConfig(
        summarizer=SummarizerConfig(hf_abstractive_model_name="hf-internal-testing/tiny-random-BartForConditionalGeneration", hf_tokenizer_name="hf-internal-testing/tiny-random-BartForConditionalGeneration"),
        retriever=LLMRetrieverConfig(embedding_model_name="hf-internal-testing/tiny-random-SentenceTransformer", cross_encoder_model_name=None),
        listener=LLMListenerConfig(
            entity_extraction_model_name="hf-internal-testing/tiny-random-DistilBertForTokenClassification",
            relation_extraction_model_name=None,
            general_llm_for_re_model_name="hf-internal-testing/tiny-random-GPT2LMHeadModel",
            embedding_model_name="hf-internal-testing/tiny-random-SentenceTransformer"
        ),
        agent=ContextAgentConfig(
            intent_classifier_model="hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
            spacy_model_name="en_core_web_sm"
        ),
        log_level="DEBUG"
    )
    return cfg

@pytest.mark.asyncio
async def test_handle_search_request_flow(app_config_integration: AppConfig, monkeypatch):
    """
    Tests a basic end-to-end flow for a search request through the ContextAgent.
    Uses actual core logic components with stubbed memory and potentially mocked LLM calls.
    """
    app_config = app_config_integration # Use the fixture

    # 1. Mock Hugging Face pipeline calls to ensure test speed and avoid downloads during test.
    # The app_config_integration fixture already sets tiny models that might not need network if cached,
    # but explicit mocking is safer for CI.

    mock_pipeline_instance = MagicMock()
    def pipeline_side_effect(task, model, **kwargs):
        if task == "zero-shot-classification":
            return MagicMock(return_value={'sequence': 'test input', 'labels': ['search information'], 'scores': [0.99]})
        elif task == "ner":
            return MagicMock(return_value=[]) # No entities for simplicity
        elif task == "text2text-generation" or task == "text-generation":
             # For Summarizer, RE LLM - ensure output matches what pipeline would give
            return MagicMock(return_value=[{'summary_text': 'mocked summary', 'generated_text': 'mocked generated text'}])
        return MagicMock() # Default mock for any other pipeline

    mock_hf_pipeline_constructor = MagicMock(side_effect=pipeline_side_effect)

    monkeypatch.setattr("contextkernel.core_logic.summarizer.pipeline", mock_hf_pipeline_constructor)
    monkeypatch.setattr("contextkernel.core_logic.llm_listener.pipeline", mock_hf_pipeline_constructor)
    monkeypatch.setattr("contextkernel.core_logic.context_agent.pipeline", mock_hf_pipeline_constructor)

    # Mock SentenceTransformer and CrossEncoder constructors to return mocks
    # This avoids actual model loading for embedding/cross-encoding if "hf-internal-testing" models are not sufficient
    mock_st_model_instance = MagicMock()
    mock_st_model_instance.encode.return_value = [[0.1]*384] # Ensure correct embedding dimension
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.SentenceTransformer", MagicMock(return_value=mock_st_model_instance))

    mock_ce_model_instance = MagicMock()
    mock_ce_model_instance.predict.return_value = [0.5]
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.CrossEncoder", MagicMock(return_value=mock_ce_model_instance))


    # 2. Instantiate components
    try:
        embedding_model = HuggingFaceEmbeddingModel(model_name=app_config.retriever.embedding_model_name, device=app_config.retriever.embedding_device)
        summarizer = Summarizer(default_config=app_config.summarizer)
    except (ConfigurationError, EmbeddingError) as e:
        pytest.skip(f"Skipping integration test: Core component init failed - {e}")

    ltm_stub = StubLTM(retriever_config=app_config.retriever)
    graph_db_stub = StubGraphDB() # No config path needed for in-memory
    raw_cache_stub = SimpleStubRawCache() # Using the simplified test stub
    stm_stub = SimpleStubSTM()

    memory_systems = {"ltm": ltm_stub, "graph_db": graph_db_stub, "raw_cache": raw_cache_stub, "stm": stm_stub}

    listener = LLMListener(listener_config=app_config.listener, memory_systems=memory_systems)
    listener.embedding_model = embedding_model
    listener.summarizer = summarizer

    retriever = LLMRetriever(retriever_config=app_config.retriever, ltm_interface=ltm_stub, stm_interface=stm_stub, graphdb_interface=graph_db_stub)
    retriever.embedding_model = embedding_model

    llm_service_facade = SimpleNamespace(retriever=retriever, listener=listener)
    state_manager = InMemoryStateManager()

    # Mock spaCy load if en_core_web_sm is not guaranteed in test env
    mock_spacy_nlp = MagicMock()
    mock_spacy_doc = MagicMock(); mock_spacy_doc.ents = []
    mock_spacy_nlp.return_value = mock_spacy_doc
    monkeypatch.setattr("spacy.load", MagicMock(return_value=mock_spacy_nlp))

    agent = ContextAgent(llm_service=llm_service_facade, memory_system=SimpleNamespace(**memory_systems), agent_config=app_config.agent, state_manager=state_manager)
    # Ensure agent's pipelines are the mocked ones if they were created by global patch
    if hasattr(agent, 'intent_classifier') and agent.intent_classifier is not None:
        agent.intent_classifier = mock_hf_pipeline_constructor("zero-shot-classification", model=app_config.agent.intent_classifier_model)


    # 3. Pre-populate LTM with some data
    doc_id_test = "doc_test_1"
    text_content_test = "Context Kernel is an advanced AI framework for building context-aware applications."
    embedding_test = await embedding_model.generate_embedding(text_content_test)
    await ltm_stub.add_document(doc_id=doc_id_test, text_content=text_content_test, embedding=embedding_test, metadata={"source": "test_doc"})

    # 4. Call handle_request
    user_query = "search for information about context kernel"
    session_id = "test_session_integration_123"
    task_result = await agent.handle_request(user_query, session_id)

    # 5. Assertions
    assert task_result is not None, "handle_request should return a TaskResult"
    assert task_result.status == "success", f"Request failed: {task_result.message} - {task_result.error_details}"

    assert isinstance(task_result.data, RetrievalResponse), "Data from a search should be RetrievalResponse"
    retrieval_response: RetrievalResponse = task_result.data # type: ignore

    assert len(retrieval_response.items) > 0, "Expected some items to be retrieved"
    # The content might not be exactly the same due to mock embedding/search score, focus on source and doc_id.
    assert any(item.metadata.get("doc_id") == doc_id_test for item in retrieval_response.items), \
           f"Expected retrieved items to include test doc '{doc_id_test}'"

    saved_state = await state_manager.get_state(session_id)
    assert saved_state is not None
    assert saved_state["last_intent_detected"] == "search information"
    assert saved_state["last_task_status"] == "success"

    print(f"Integration test 'test_handle_search_request_flow' passed. Result: {task_result.message}")
