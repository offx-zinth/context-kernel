import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List, Optional

# Modules to test
from contextkernel.core_logic.llm_listener import (
    LLMListenerConfig,
    LLMListener,
    StructuredInsight,
    # Interfaces (for type hinting and spec for mocks)
    RawCacheInterface,
    STMInterface,
    LTMInterface,
    GraphDBInterface,
)
from contextkernel.core_logic.summarizer import Summarizer, SummarizerConfig
from contextkernel.core_logic.llm_retriever import HuggingFaceEmbeddingModel, StubLTM as RetrieverStubLTM, StubGraphDB as RetrieverStubGraphDB

# --- Re-define simple Stubs for RawCache and STM as inner classes or here for test context ---
class TestStubRawCache(RawCacheInterface):
    def __init__(self):
        super().__init__()
        self.cache: Dict[str, Any] = {}

    async def store(self, doc_id: str, data: Any) -> Optional[str]:
        self.logger.info(f"TestStubRawCache storing data with doc_id: {doc_id}.")
        self.cache[doc_id] = data
        return doc_id

    async def load(self, doc_id: str) -> Optional[Any]:
        self.logger.info(f"TestStubRawCache loading data with doc_id: {doc_id}.")
        return self.cache.get(doc_id)

class TestStubSTM(STMInterface):
    def __init__(self):
        super().__init__()
        self.cache: Dict[str, Any] = {}

    async def save_summary(self, summary_id: str, summary_obj: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.logger.info(f"TestStubSTM saving summary with summary_id: {summary_id}.")
        self.cache[summary_id] = {"summary": summary_obj, "metadata": metadata or {}}

    async def load_summary(self, summary_id: str) -> Optional[Any]:
        self.logger.info(f"TestStubSTM loading summary with summary_id: {summary_id}.")
        return self.cache.get(summary_id)


# Global Mocks for external libraries (Hugging Face Transformers Pipeline)
@pytest.fixture(autouse=True)
def mock_hf_pipelines(monkeypatch):
    """Mocks Hugging Face transformers.pipeline for all tests."""
    mock_pipeline_instance = MagicMock()
    # Default behavior for a pipeline call; can be customized per test
    mock_pipeline_instance.return_value = [{"generated_text": "mocked pipeline output"}]

    mock_pipeline_constructor = MagicMock(return_value=mock_pipeline_instance)
    monkeypatch.setattr("contextkernel.core_logic.llm_listener.pipeline", mock_pipeline_constructor)
    # To access the constructor mock later for assertions:
    # contextkernel.core_logic.llm_listener.pipeline
    return mock_pipeline_constructor


@pytest.fixture
def default_listener_config():
    """Returns a default LLMListenerConfig for tests."""
    return LLMListenerConfig(
        summarizer_config=SummarizerConfig(hf_abstractive_model_name="mock-summarizer"),
        entity_extraction_model_name="mock-ner-model",
        relation_extraction_model_name=None, # Test with general LLM for RE by default
        general_llm_for_re_model_name="mock-re-llm",
        embedding_model_name="mock-embedding-model"
    )

@pytest.fixture
def mock_memory_systems():
    """Provides a dictionary of mocked memory system interfaces."""
    return {
        "raw_cache": AsyncMock(spec=TestStubRawCache), # Use our test stubs as spec
        "stm": AsyncMock(spec=TestStubSTM),
        "ltm": AsyncMock(spec=RetrieverStubLTM), # Use the actual StubLTM from retriever for interface matching
        "graph_db": AsyncMock(spec=RetrieverStubGraphDB)
    }

@pytest.fixture
def llm_listener(default_listener_config, mock_memory_systems, monkeypatch):
    """Fixture to create an LLMListener instance with mocked dependencies."""
    # Mock Summarizer and HuggingFaceEmbeddingModel constructors
    mock_summarizer_instance = AsyncMock(spec=Summarizer)
    mock_summarizer_instance.summarize = AsyncMock(return_value="Mocked summary.")
    mock_summarizer_constructor = MagicMock(return_value=mock_summarizer_instance)
    monkeypatch.setattr("contextkernel.core_logic.llm_listener.Summarizer", mock_summarizer_constructor)

    mock_hf_embedding_instance = AsyncMock(spec=HuggingFaceEmbeddingModel)
    mock_hf_embedding_instance.model = MagicMock() # Simulate a loaded model within the embedding model
    mock_hf_embedding_instance.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_hf_embedding_constructor = MagicMock(return_value=mock_hf_embedding_instance)
    monkeypatch.setattr("contextkernel.core_logic.llm_listener.HuggingFaceEmbeddingModel", mock_hf_embedding_constructor)

    listener = LLMListener(
        listener_config=default_listener_config,
        memory_systems=mock_memory_systems,
    )
    # Attach mocks for easy access in tests if needed, though direct patching is often cleaner
    listener.summarizer = mock_summarizer_instance
    listener.embedding_model = mock_hf_embedding_instance
    return listener


class TestLLMListenerInit:
    def test_init_successful_all_models(self, default_listener_config, mock_memory_systems, mock_hf_pipelines, monkeypatch):
        """Tests successful initialization of LLMListener with all models configured."""

        mock_sum_constructor = MagicMock(return_value=AsyncMock(spec=Summarizer))
        monkeypatch.setattr("contextkernel.core_logic.llm_listener.Summarizer", mock_sum_constructor)

        mock_emb_constructor = MagicMock(return_value=AsyncMock(spec=HuggingFaceEmbeddingModel, model=MagicMock()))
        monkeypatch.setattr("contextkernel.core_logic.llm_listener.HuggingFaceEmbeddingModel", mock_emb_constructor)

        listener = LLMListener(default_listener_config, mock_memory_systems)

        mock_sum_constructor.assert_called_once_with(default_listener_config.summarizer_config)
        mock_emb_constructor.assert_called_once_with(model_name=default_listener_config.embedding_model_name)

        # Check NER pipeline call
        # mock_hf_pipelines is the constructor for `pipeline`
        mock_hf_pipelines.assert_any_call(
            "ner",
            model=default_listener_config.entity_extraction_model_name,
            tokenizer=default_listener_config.entity_extraction_model_name
        )
        # Check RE LLM pipeline call (since relation_extraction_model_name is None by default)
        mock_hf_pipelines.assert_any_call(
            "text-generation",
            model=default_listener_config.general_llm_for_re_model_name,
            tokenizer=default_listener_config.general_llm_for_re_model_name
        )
        assert listener.summarizer is not None
        assert listener.embedding_model is not None
        assert listener.ner_pipeline is not None
        assert listener.re_llm_pipeline is not None
        assert listener.re_pipeline is None # Dedicated RE model was None in default_listener_config


    def test_init_dedicated_re_model(self, mock_memory_systems, mock_hf_pipelines, monkeypatch):
        config = LLMListenerConfig(
            relation_extraction_model_name="mock-dedicated-re",
            general_llm_for_re_model_name=None # Disable general RE LLM
        )
        mock_sum_constructor = MagicMock(return_value=AsyncMock(spec=Summarizer))
        monkeypatch.setattr("contextkernel.core_logic.llm_listener.Summarizer", mock_sum_constructor)
        mock_emb_constructor = MagicMock(return_value=AsyncMock(spec=HuggingFaceEmbeddingModel, model=MagicMock()))
        monkeypatch.setattr("contextkernel.core_logic.llm_listener.HuggingFaceEmbeddingModel", mock_emb_constructor)

        listener = LLMListener(config, mock_memory_systems)

        mock_hf_pipelines.assert_any_call(
            "text2text-generation", # Default task for dedicated RE in __init__
            model=config.relation_extraction_model_name,
            tokenizer=config.relation_extraction_model_name
        )
        assert listener.re_pipeline is not None
        assert listener.re_llm_pipeline is None

    def test_init_pipeline_creation_failure(self, default_listener_config, mock_memory_systems, mock_hf_pipelines, caplog, monkeypatch):
        """Tests that pipeline creation failures are logged and pipelines are None."""
        mock_hf_pipelines.side_effect = Exception("Pipeline creation failed!")

        mock_sum_constructor = MagicMock(return_value=AsyncMock(spec=Summarizer))
        monkeypatch.setattr("contextkernel.core_logic.llm_listener.Summarizer", mock_sum_constructor)
        mock_emb_constructor = MagicMock(return_value=AsyncMock(spec=HuggingFaceEmbeddingModel, model=MagicMock()))
        monkeypatch.setattr("contextkernel.core_logic.llm_listener.HuggingFaceEmbeddingModel", mock_emb_constructor)

        listener = LLMListener(default_listener_config, mock_memory_systems)

        assert listener.ner_pipeline is None
        assert listener.re_llm_pipeline is None
        assert "Failed to initialize NER pipeline" in caplog.text
        assert "Failed to initialize general LLM for RE pipeline" in caplog.text


class TestLLMListenerCallLLMMethods:
    @pytest.mark.asyncio
    async def test_call_llm_summarize(self, llm_listener):
        text_content = "This is a long text to summarize."
        custom_instructions = {"desired_length_type": "words", "desired_length_value": 50}

        # The llm_listener fixture already has summarizer mocked
        expected_summary = "Mocked summary for test."
        llm_listener.summarizer.summarize = AsyncMock(return_value=expected_summary)

        summary = await llm_listener._call_llm_summarize(text_content, instructions=custom_instructions)

        assert summary == expected_summary
        # Check if summarize was called with text and a SummarizerConfig object
        llm_listener.summarizer.summarize.assert_called_once()
        call_args = llm_listener.summarizer.summarize.call_args
        assert call_args[0][0] == text_content
        assert isinstance(call_args[1]['config'], SummarizerConfig)
        assert call_args[1]['config'].desired_length_type == "words"
        assert call_args[1]['config'].desired_length_value == 50


    @pytest.mark.asyncio
    async def test_call_llm_extract_entities_success(self, llm_listener, mock_hf_pipelines):
        text_content = "Dr. Alice Smith works at Google in New York."
        # Configure the global pipeline mock for NER for this test
        mock_ner_output = [
            {'entity_group': 'PER', 'word': 'Alice Smith', 'start': 4, 'end': 15, 'score': 0.99},
            {'entity_group': 'ORG', 'word': 'Google', 'start': 25, 'end': 31, 'score': 0.98},
            {'entity_group': 'LOC', 'word': 'New York', 'start': 35, 'end': 43, 'score': 0.97}
        ]
        # The llm_listener's ner_pipeline is already a MagicMock due to mock_hf_pipelines
        llm_listener.ner_pipeline.return_value = mock_ner_output

        entities = await llm_listener._call_llm_extract_entities(text_content)

        assert entities is not None
        assert len(entities) == 3
        assert entities[0]['text'] == 'Alice Smith'
        assert entities[0]['type'] == 'PER'
        assert entities[1]['text'] == 'Google'
        assert entities[1]['type'] == 'ORG'
        llm_listener.ner_pipeline.assert_called_once_with(text_content)

    @pytest.mark.asyncio
    async def test_call_llm_extract_entities_pipeline_unavailable(self, llm_listener, caplog):
        llm_listener.ner_pipeline = None # Simulate pipeline failure
        entities = await llm_listener._call_llm_extract_entities("Some text")
        assert entities == []
        assert "NER pipeline not available" in caplog.text


    @pytest.mark.asyncio
    async def test_call_llm_extract_relations_dedicated_re_pipeline(self, llm_listener, mock_hf_pipelines):
        text_content = "Alice works for Bob."
        # Assume listener is configured with a dedicated RE model
        llm_listener.re_pipeline = MagicMock(return_value=[ # Simulate dedicated RE model output
            {"subject": "Alice", "relation": "works_for", "object": "Bob"}
        ])
        llm_listener.re_llm_pipeline = None # Ensure general LLM RE is not used

        relations = await llm_listener._call_llm_extract_relations(text_content, entities=[]) # Entities optional here

        assert relations is not None
        assert len(relations) == 1
        assert relations[0]['subject'] == "Alice"
        assert relations[0]['verb'] == "works_for"
        assert relations[0]['object'] == "Bob"
        llm_listener.re_pipeline.assert_called_once_with(text_content)

    @pytest.mark.asyncio
    async def test_call_llm_extract_relations_general_llm_re_pipeline(self, llm_listener, monkeypatch):
        text_content = "Carol manages David."
        # Mock the tokenizer used by the general RE LLM pipeline for max_length calculation
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1] * len(text_content.split()) # Dummy token IDs

        # Assume listener is configured with a general LLM for RE
        llm_listener.re_pipeline = None # Ensure dedicated RE is not used
        # The re_llm_pipeline is already a MagicMock from mock_hf_pipelines in llm_listener fixture
        llm_listener.re_llm_pipeline.tokenizer = mock_tokenizer # Assign mock tokenizer

        # Simulate output from general LLM pipeline
        # Prompt is constructed internally, then text is appended. We need to mock the final output.
        # Expected prompt structure: "Entities found: ... Context text: \"...\" Extracted Relations:"
        # For simplicity, we'll assume the prompt is roughly X tokens and output is Y tokens.
        # The important part is that the mock returns text that includes the prompt.

        # Construct part of the expected prompt to simulate its length for the mock output
        mock_prompt_prefix = "Entities found: \nGiven the text below, extract relations ... Extracted Relations:\n"
        llm_generated_text = "(Carol; manages; David)\n(Another; relation; example)" # What the LLM "generates"

        # The pipeline mock needs to return this structure: [{'generated_text': 'full_output_incl_prompt'}]
        llm_listener.re_llm_pipeline.return_value = [{'generated_text': mock_prompt_prefix + llm_generated_text}]


        entities_for_prompt = [{"text": "Carol"}, {"text": "David"}] # Example entities
        relations = await llm_listener._call_llm_extract_relations(text_content, entities=entities_for_prompt)

        assert relations is not None
        assert len(relations) == 2
        assert relations[0]['subject'] == "Carol"
        assert relations[0]['verb'] == "manages"
        assert relations[0]['object'] == "David"
        assert relations[1]['subject'] == "Another"

        llm_listener.re_llm_pipeline.assert_called_once()
        # We can also check the prompt passed to the pipeline if needed by inspecting call_args

    @pytest.mark.asyncio
    async def test_call_llm_extract_relations_no_pipeline(self, llm_listener, caplog):
        llm_listener.re_pipeline = None
        llm_listener.re_llm_pipeline = None
        relations = await llm_listener._call_llm_extract_relations("Some text", entities=[])
        assert relations == []
        assert "No RE pipeline available or configured" in caplog.text


class TestLLMListenerInsightProcessing:
    @pytest.mark.asyncio
    async def test_generate_insights(self, llm_listener):
        raw_data = "Some important text data."
        context_instructions = {"summarize": True, "extract_entities": True, "extract_relations": True}
        raw_data_doc_id = "raw_doc_123"

        # Mock the sub-methods that _generate_insights calls
        llm_listener._call_llm_summarize = AsyncMock(return_value="Generated Summary")
        llm_listener._call_llm_extract_entities = AsyncMock(return_value=[{"text": "Entity1"}])
        llm_listener._call_llm_extract_relations = AsyncMock(return_value=[{"subject": "Entity1"}])
        # llm_listener.embedding_model.generate_embedding is already an AsyncMock from the fixture

        insights = await llm_listener._generate_insights(raw_data, context_instructions, raw_data_doc_id)

        llm_listener._call_llm_summarize.assert_called_once_with(raw_data, instructions=None)
        llm_listener._call_llm_extract_entities.assert_called_once_with(raw_data, instructions=None)
        # Relations called with entities from _call_llm_extract_entities
        llm_listener._call_llm_extract_relations.assert_called_once_with(raw_data, [{"text": "Entity1"}], instructions=None)
        llm_listener.embedding_model.generate_embedding.assert_called_once_with(raw_data)

        assert insights["summary"] == "Generated Summary"
        assert insights["entities"] == [{"text": "Entity1"}]
        assert insights["relations"] == [{"subject": "Entity1"}]
        assert insights["original_data"] == raw_data
        assert insights["raw_data_doc_id"] == raw_data_doc_id
        assert insights["content_embedding"] == [0.1, 0.2, 0.3] # From mock_hf_embedding_instance

    @pytest.mark.asyncio
    async def test_structure_data(self, llm_listener):
        insights_dict = {
            "summary": "A summary.",
            "entities": [{"text": "Paris", "type": "LOC"}],
            "relations": [{"subject": "Paris", "verb": "isCapitalOf", "object": "France"}],
            "original_data": "Paris is the capital of France.",
            "raw_data_doc_id": "doc_raw_456",
            "content_embedding": [0.5, 0.4, 0.3]
        }
        structured_insight = await llm_listener._structure_data(insights_dict)

        assert isinstance(structured_insight, StructuredInsight)
        assert structured_insight.summary.text == "A summary."
        assert len(structured_insight.entities) == 1
        assert structured_insight.entities[0].text == "Paris"
        assert structured_insight.raw_data_id == "doc_raw_456"
        assert structured_insight.content_embedding == [0.5, 0.4, 0.3]

    @pytest.mark.asyncio
    async def test_write_to_memory(self, llm_listener, mock_memory_systems):
        structured_data = StructuredInsight(
            source_data_preview="France...",
            summary={"text": "Summary about France."}, # Needs to be Summary object
            entities=[{"text": "France", "type": "LOC"}], # Needs to be Entity object
            relations=[{"subject": "France", "verb": "is", "object": "Country"}], # Relation object
            raw_data_id="raw1",
            content_embedding=[0.1,0.1]
        )
        # For the test, we need to ensure the objects are of the Pydantic types
        # This is normally handled by _structure_data
        from contextkernel.core_logic.llm_listener import Summary, Entity, Relation
        structured_data.summary = Summary(text="Summary about France.")
        structured_data.entities = [Entity(text="France", type="LOC")]
        structured_data.relations = [Relation(subject="France", verb="is", object="Country")]


        await llm_listener._write_to_memory(structured_data)

        # STM: save_summary
        mock_memory_systems["stm"].save_summary.assert_called_once()
        stm_call_args = mock_memory_systems["stm"].save_summary.call_args
        assert stm_call_args[1]['summary_id'].startswith(structured_data.raw_data_id if structured_data.raw_data_id else "")
        assert stm_call_args[1]['summary_obj'] == structured_data.summary

        # LTM: save_document
        mock_memory_systems["ltm"].save_document.assert_called_once()
        ltm_call_args = mock_memory_systems["ltm"].save_document.call_args
        assert ltm_call_args[1]['doc_id'].startswith(structured_data.raw_data_id if structured_data.raw_data_id else "")
        assert ltm_call_args[1]['text_content'] == structured_data.source_data_preview
        assert ltm_call_args[1]['embedding'] == structured_data.content_embedding

        # GraphDB: add_entities, add_relations
        mock_memory_systems["graph_db"].add_entities.assert_called_once_with(
            entities=structured_data.entities, document_id=mock.ANY, metadata=mock.ANY
        )
        mock_memory_systems["graph_db"].add_relations.assert_called_once_with(
            relations=structured_data.relations, document_id=mock.ANY, metadata=mock.ANY
        )

    @pytest.mark.asyncio
    async def test_process_data_end_to_end(self, llm_listener, mock_memory_systems):
        raw_data = "Live test data for processing."
        context_instructions = {"summarize": True, "extract_entities": True, "extract_relations": True}

        # Mock all sub-methods that process_data calls internally
        llm_listener._preprocess_data = AsyncMock(side_effect=lambda x: x) # passthrough
        llm_listener._generate_insights = AsyncMock(return_value={
            "summary": "Processed summary", "entities": [], "relations": [],
            "original_data": raw_data, "raw_data_doc_id": "test_raw_id_1",
            "content_embedding": [0.7,0.8]
        })
        llm_listener._structure_data = AsyncMock(
            # side_effect=lambda insights: StructuredInsight(**insights) # simplified
            return_value = StructuredInsight( # More explicit for clarity
                summary=Summary(text="Processed summary"),
                entities=[], relations=[], original_data_type=type(raw_data).__name__,
                source_data_preview=raw_data[:100]+"...", raw_data_id="test_raw_id_1",
                content_embedding=[0.7,0.8]
            )
        )
        llm_listener._write_to_memory = AsyncMock()

        await llm_listener.process_data(raw_data, context_instructions)

        mock_memory_systems["raw_cache"].store.assert_called_once() # Check raw_cache was called
        llm_listener._preprocess_data.assert_called_once_with(raw_data)
        llm_listener._generate_insights.assert_called_once()
        llm_listener._structure_data.assert_called_once()
        llm_listener._write_to_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_data_error_handling(self, llm_listener, caplog):
        """Tests that an error in a step of process_data is logged."""
        raw_data = "Data that will cause an error."
        llm_listener._preprocess_data = AsyncMock(side_effect=Exception("Preprocessing failed!"))

        await llm_listener.process_data(raw_data)

        assert "Error during data processing pipeline: Preprocessing failed!" in caplog.text
