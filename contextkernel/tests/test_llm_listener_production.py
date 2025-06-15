import unittest
from unittest.mock import patch, MagicMock, ANY, mock_open
import logging
import sys # For mocking imports

# Assuming LLMListener, LLMListenerConfig, StructuredInsight, Summary, Entity, Relation are in llm_listener.py
from contextkernel.core_logic.llm_listener import (
    LLMListener,
    LLMListenerConfig,
    StructuredInsight,
    Summary,
    Entity,
    Relation,
    BaseMemorySystem # Import BaseMemorySystem for mocking
)

# Disable logging for tests unless specifically needed
logging.disable(logging.CRITICAL)

# Mock the pipeline class from transformers if transformers is installed, otherwise it's already None
try:
    from transformers import Pipeline as TransformersPipeline # type: ignore
except ImportError:
    TransformersPipeline = None # type: ignore

class TestLLMListenerProduction(unittest.IsolatedAsyncioTestCase): # For async methods

    def setUp(self):
        """Set up common resources for tests."""
        self.listener_config = LLMListenerConfig(
            summarization_model_name="test-summarizer",
            entity_extraction_model_name="test-ner-model",
            enable_stub_relation_extraction=True, # Enable for some tests
            default_summarization_min_length=20,
            default_summarization_max_length=100
        )

        self.mock_stm = MagicMock(spec=BaseMemorySystem)
        self.mock_ltm = MagicMock(spec=BaseMemorySystem)
        self.mock_graph_db = MagicMock(spec=BaseMemorySystem)
        self.mock_raw_cache = MagicMock(spec=BaseMemorySystem)

        self.memory_systems = {
            "stm": self.mock_stm,
            "ltm": self.mock_ltm,
            "graph_db": self.mock_graph_db,
            "raw_cache": self.mock_raw_cache,
        }
        self.mock_llm_client = MagicMock()

    def create_listener_with_patched_pipeline(self, mock_pipeline_func):
        """Helper to create listener with a specific pipeline mock."""
        with patch('contextkernel.core_logic.llm_listener.pipeline', mock_pipeline_func):
            listener = LLMListener(
                listener_config=self.listener_config,
                memory_systems=self.memory_systems,
                llm_client=self.mock_llm_client
            )
        return listener

    # 1. Initialization Tests
    def test_initialization_successful_pipelines(self):
        mock_pipeline_instance_summarizer = MagicMock(spec=TransformersPipeline if TransformersPipeline else MagicMock)
        mock_pipeline_instance_ner = MagicMock(spec=TransformersPipeline if TransformersPipeline else MagicMock)

        def mock_pipeline_side_effect(task, model, tokenizer):
            if task == "summarization":
                self.assertEqual(model, self.listener_config.summarization_model_name)
                return mock_pipeline_instance_summarizer
            elif task == "ner":
                self.assertEqual(model, self.listener_config.entity_extraction_model_name)
                return mock_pipeline_instance_ner
            raise ValueError(f"Unexpected task for pipeline: {task}")

        mock_pipeline_func = MagicMock(side_effect=mock_pipeline_side_effect)

        listener = self.create_listener_with_patched_pipeline(mock_pipeline_func)

        self.assertEqual(listener.listener_config, self.listener_config)
        self.assertIsNotNone(listener.summarization_pipeline)
        self.assertIsNotNone(listener.ner_pipeline)
        self.assertEqual(mock_pipeline_func.call_count, 2) # Called for summarization and NER

    def test_initialization_transformers_not_available(self):
        # Simulate 'from transformers import pipeline' failing
        with patch.dict('sys.modules', {'transformers': None}):
            # Need to re-patch 'pipeline' inside llm_listener specifically for this test
            # as it might have been imported by the test module itself already.
            with patch('contextkernel.core_logic.llm_listener.pipeline', None):
                listener = LLMListener(
                    listener_config=self.listener_config,
                    memory_systems=self.memory_systems,
                    llm_client=self.mock_llm_client
                )
                self.assertIsNone(listener.summarization_pipeline)
                self.assertIsNone(listener.ner_pipeline)
                # Add log check here if possible

    def test_initialization_summarization_pipeline_fails(self):
        mock_pipeline_instance_ner = MagicMock(spec=TransformersPipeline if TransformersPipeline else MagicMock)
        def mock_pipeline_side_effect(task, model, tokenizer):
            if task == "summarization":
                raise Exception("Summarization model load failed")
            elif task == "ner":
                return mock_pipeline_instance_ner
            return MagicMock()

        mock_pipeline_func = MagicMock(side_effect=mock_pipeline_side_effect)
        listener = self.create_listener_with_patched_pipeline(mock_pipeline_func)

        self.assertIsNone(listener.summarization_pipeline)
        self.assertIsNotNone(listener.ner_pipeline)

    def test_initialization_ner_pipeline_fails(self):
        mock_pipeline_instance_summarizer = MagicMock(spec=TransformersPipeline if TransformersPipeline else MagicMock)
        def mock_pipeline_side_effect(task, model, tokenizer):
            if task == "summarization":
                return mock_pipeline_instance_summarizer
            elif task == "ner":
                raise Exception("NER model load failed")
            return MagicMock()

        mock_pipeline_func = MagicMock(side_effect=mock_pipeline_side_effect)
        listener = self.create_listener_with_patched_pipeline(mock_pipeline_func)

        self.assertIsNotNone(listener.summarization_pipeline)
        self.assertIsNone(listener.ner_pipeline)

    # 2. Preprocessing Tests (_preprocess_data)
    async def test_preprocess_data_simple_text(self):
        # Current _preprocess_data is a passthrough, so this test is simple
        listener = self.create_listener_with_patched_pipeline(MagicMock()) # Pipeline mock not important here
        raw_text = "  Some Text with spaces  "
        processed_text = await listener._preprocess_data(raw_text)
        self.assertEqual(processed_text, raw_text) # As it's a passthrough

    # 3. Relation Extraction (Placeholder) Tests (_call_llm_extract_relations)
    async def test_call_llm_extract_relations_stub_enabled_with_entities(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.listener_config.enable_stub_relation_extraction = True # Ensure enabled

        sample_entities = [
            {"text": "Entity1", "type": "PERSON"},
            {"text": "Entity2", "type": "LOCATION"}
        ]
        text_content = "Entity1 visited Entity2."
        relations = await listener._call_llm_extract_relations(text_content, sample_entities)

        self.assertIsNotNone(relations)
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["subject"], "Entity1")
        self.assertEqual(relations[0]["verb"], "is_related_to_stub")
        self.assertEqual(relations[0]["object"], "Entity2")

    async def test_call_llm_extract_relations_stub_disabled(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.listener_config.enable_stub_relation_extraction = False # Ensure disabled

        sample_entities = [{"text": "Entity1", "type": "PERSON"}, {"text": "Entity2", "type": "LOCATION"}]
        text_content = "Entity1 visited Entity2."
        relations = await listener._call_llm_extract_relations(text_content, sample_entities)

        self.assertEqual(relations, [])

    async def test_call_llm_extract_relations_not_enough_entities(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.listener_config.enable_stub_relation_extraction = True

        sample_entities = [{"text": "Entity1", "type": "PERSON"}] # Only one entity
        text_content = "Entity1 exists."
        relations = await listener._call_llm_extract_relations(text_content, sample_entities)

        self.assertEqual(relations, [])

    # 4. LLM Calls Tests
    # 4.1 _call_llm_summarize
    async def test_call_llm_summarize_pipeline_success(self):
        mock_summarizer_pipeline = MagicMock(return_value=[{'summary_text': 'Test summary'}])
        listener = self.create_listener_with_patched_pipeline(lambda task, model, tokenizer: mock_summarizer_pipeline if task == "summarization" else MagicMock())
        listener.summarization_pipeline = mock_summarizer_pipeline # Ensure it's set

        text_content = "This is a long text to summarize."
        summary = await listener._call_llm_summarize(text_content)

        self.assertEqual(summary, "Test summary")
        mock_summarizer_pipeline.assert_called_once_with(
            text_content,
            min_length=self.listener_config.default_summarization_min_length,
            max_length=self.listener_config.default_summarization_max_length,
            truncation=True
        )

    async def test_call_llm_summarize_pipeline_success_with_instructions(self):
        mock_summarizer_pipeline = MagicMock(return_value=[{'summary_text': 'Test summary'}])
        listener = self.create_listener_with_patched_pipeline(lambda task, model, tokenizer: mock_summarizer_pipeline if task == "summarization" else MagicMock())
        listener.summarization_pipeline = mock_summarizer_pipeline

        text_content = "This is a long text to summarize."
        instructions = {'min_length': 5, 'max_length': 50}
        summary = await listener._call_llm_summarize(text_content, instructions=instructions)

        self.assertEqual(summary, "Test summary")
        mock_summarizer_pipeline.assert_called_once_with(
            text_content,
            min_length=5,
            max_length=50,
            truncation=True
        )

    async def test_call_llm_summarize_pipeline_unavailable_llm_client_fallback_success(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.summarization_pipeline = None # Simulate pipeline failure
        self.mock_llm_client.summarize = AsyncMock(return_value="Summary from client") # AsyncMock for await

        summary = await listener._call_llm_summarize("Text for client")

        self.assertEqual(summary, "Summary from client")
        self.mock_llm_client.summarize.assert_called_once_with(
            "Text for client",
            max_length=self.listener_config.default_summarization_max_length
        )

    async def test_call_llm_summarize_pipeline_unavailable_llm_client_fallback_fails(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.summarization_pipeline = None
        self.mock_llm_client.summarize = AsyncMock(side_effect=Exception("Client error"))

        summary = await listener._call_llm_summarize("Text for client")
        self.assertIsNone(summary)
        self.mock_llm_client.summarize.assert_called_once()

    async def test_call_llm_summarize_pipeline_and_client_unavailable(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.summarization_pipeline = None
        listener.llm_client = None # No client either

        summary = await listener._call_llm_summarize("Some text")
        self.assertIsNone(summary)

    async def test_call_llm_summarize_pipeline_exception(self):
        mock_summarizer_pipeline = MagicMock(side_effect=Exception("Pipeline failure"))
        listener = self.create_listener_with_patched_pipeline(lambda task, model, tokenizer: mock_summarizer_pipeline if task == "summarization" else MagicMock())
        listener.summarization_pipeline = mock_summarizer_pipeline

        summary = await listener._call_llm_summarize("Some text")
        self.assertIsNone(summary)

    # 4.2 _call_llm_extract_entities
    async def test_call_llm_extract_entities_pipeline_success(self):
        mock_ner_output = [
            {'entity_group': 'PER', 'score': 0.99, 'word': 'John Doe', 'start': 12, 'end': 20},
            {'entity_group': 'LOC', 'score': 0.98, 'word': 'New York', 'start': 30, 'end': 38}
        ]
        mock_ner_pipeline = MagicMock(return_value=mock_ner_output)
        listener = self.create_listener_with_patched_pipeline(lambda task, model, tokenizer: mock_ner_pipeline if task == "ner" else MagicMock())
        listener.ner_pipeline = mock_ner_pipeline

        text_content = "Text mentioning John Doe who lives in New York City."
        entities = await listener._call_llm_extract_entities(text_content)

        self.assertIsNotNone(entities)
        self.assertEqual(len(entities), 2)

        self.assertEqual(entities[0]['text'], 'John Doe')
        self.assertEqual(entities[0]['type'], 'PER')
        self.assertIn("[John Doe]", entities[0]['context'])
        self.assertEqual(entities[0]['score'], 0.99)

        self.assertEqual(entities[1]['text'], 'New York')
        self.assertEqual(entities[1]['type'], 'LOC')
        self.assertIn("[New York]", entities[1]['context'])

        mock_ner_pipeline.assert_called_once_with(text_content)

    async def test_call_llm_extract_entities_pipeline_unavailable(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.ner_pipeline = None # Simulate NER pipeline init failure

        entities = await listener._call_llm_extract_entities("Some text")
        self.assertEqual(entities, [])

    async def test_call_llm_extract_entities_pipeline_exception(self):
        mock_ner_pipeline = MagicMock(side_effect=Exception("NER pipeline error"))
        listener = self.create_listener_with_patched_pipeline(lambda task, model, tokenizer: mock_ner_pipeline if task == "ner" else MagicMock())
        listener.ner_pipeline = mock_ner_pipeline

        entities = await listener._call_llm_extract_entities("Some text")
        self.assertEqual(entities, [])

    # 5. Insight Generation Tests (_generate_insights)
    @patch.object(LLMListener, '_call_llm_summarize', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_entities', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_relations', new_callable=AsyncMock)
    async def test_generate_insights_all_operations(
        self, mock_extract_relations, mock_extract_entities, mock_summarize
    ):
        listener = self.create_listener_with_patched_pipeline(MagicMock())

        mock_summarize.return_value = "Generated Summary"
        mock_extract_entities.return_value = [{"text": "Entity", "type": "TYPE"}]
        mock_extract_relations.return_value = [{"subject": "Entity", "verb": "is", "object": "Related"}]

        data = "Some input data"
        raw_data_id = "raw_doc_123"
        context_instructions = {"summarize": True, "extract_entities": True, "extract_relations": True}

        insights = await listener._generate_insights(data, context_instructions, raw_data_doc_id=raw_data_id)

        mock_summarize.assert_called_once_with(data, instructions=None)
        mock_extract_entities.assert_called_once_with(data, instructions=None)
        mock_extract_relations.assert_called_once_with(data, mock_extract_entities.return_value, instructions=None)

        self.assertEqual(insights["summary"], "Generated Summary")
        self.assertEqual(insights["entities"], mock_extract_entities.return_value)
        self.assertEqual(insights["relations"], mock_extract_relations.return_value)
        self.assertEqual(insights["original_data"], data)
        self.assertEqual(insights["raw_data_doc_id"], raw_data_id)

    @patch.object(LLMListener, '_call_llm_summarize', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_entities', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_relations', new_callable=AsyncMock)
    async def test_generate_insights_summarize_only(
        self, mock_extract_relations, mock_extract_entities, mock_summarize
    ):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        mock_summarize.return_value = "Summary"

        context_instructions = {"summarize": True, "extract_entities": False, "extract_relations": False}
        insights = await listener._generate_insights("data", context_instructions)

        mock_summarize.assert_called_once()
        mock_extract_entities.assert_not_called()
        mock_extract_relations.assert_not_called()
        self.assertEqual(insights["summary"], "Summary")
        self.assertIsNone(insights["entities"])
        self.assertIsNone(insights["relations"])

    @patch.object(LLMListener, '_call_llm_summarize', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_entities', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_relations', new_callable=AsyncMock)
    async def test_generate_insights_no_context_instructions(
        self, mock_extract_relations, mock_extract_entities, mock_summarize
    ):
        # Default behavior: all True if context_instructions is None
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        mock_summarize.return_value = "S"
        mock_extract_entities.return_value = [{"text": "E", "type": "T"}]
        mock_extract_relations.return_value = [{"s": "E"}]

        insights = await listener._generate_insights("data", None) # No instructions

        mock_summarize.assert_called_once()
        mock_extract_entities.assert_called_once()
        mock_extract_relations.assert_called_once() # Called because entities were returned
        self.assertIsNotNone(insights["summary"])
        self.assertIsNotNone(insights["entities"])
        self.assertIsNotNone(insights["relations"])

    @patch.object(LLMListener, '_call_llm_summarize', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_entities', new_callable=AsyncMock)
    @patch.object(LLMListener, '_call_llm_extract_relations', new_callable=AsyncMock)
    async def test_generate_insights_skip_relations_if_no_entities(
        self, mock_extract_relations, mock_extract_entities, mock_summarize
    ):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        mock_summarize.return_value = "S"
        mock_extract_entities.return_value = [] # No entities found

        context_instructions = {"summarize": True, "extract_entities": True, "extract_relations": True}
        insights = await listener._generate_insights("data", context_instructions)

        mock_summarize.assert_called_once()
        mock_extract_entities.assert_called_once()
        mock_extract_relations.assert_not_called() # Should not be called if entities is empty
        self.assertEqual(insights["summary"], "S")
        self.assertEqual(insights["entities"], [])
        self.assertIsNone(insights["relations"])

    # 6. Data Structuring Tests (_structure_data)
    async def test_structure_data_full_insights(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        insights = {
            "summary": "Test summary text.",
            "entities": [
                {"text": "Entity1", "type": "PERSON", "context": "Ctx1", "score": 0.9, "start_char": 0, "end_char": 7},
                {"text": "Entity2", "type": "LOC", "context": "Ctx2", "score": 0.8, "start_char": 10, "end_char": 18},
            ],
            "relations": [
                {"subject": "Entity1", "verb": "knows", "object": "Entity2", "context": "CtxRel1"}
            ],
            "original_data": "Original full text here.",
            "raw_data_doc_id": "raw_doc_id_789"
        }

        structured_insight = await listener._structure_data(insights)

        self.assertIsInstance(structured_insight, StructuredInsight)
        self.assertIsNotNone(structured_insight.created_at)
        self.assertEqual(structured_insight.original_data_type, "str")
        self.assertTrue(structured_insight.source_data_preview.startswith("Original full text"))
        self.assertEqual(structured_insight.raw_data_id, "raw_doc_id_789")

        self.assertIsInstance(structured_insight.summary, Summary)
        self.assertEqual(structured_insight.summary.text, "Test summary text.")

        self.assertIsInstance(structured_insight.entities, list)
        self.assertEqual(len(structured_insight.entities), 2)
        self.assertIsInstance(structured_insight.entities[0], Entity)
        self.assertEqual(structured_insight.entities[0].text, "Entity1")
        self.assertEqual(structured_insight.entities[0].type, "PERSON")

        self.assertIsInstance(structured_insight.relations, list)
        self.assertEqual(len(structured_insight.relations), 1)
        self.assertIsInstance(structured_insight.relations[0], Relation)
        self.assertEqual(structured_insight.relations[0].subject, "Entity1")
        self.assertEqual(structured_insight.relations[0].verb, "knows")

    async def test_structure_data_partial_insights_no_relations(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        insights = {
            "summary": "Another summary.",
            "entities": [{"text": "OnlyEntity", "type": "ORG", "context": "CtxOrg"}],
            "relations": None, # No relations found/extracted
            "original_data": "Some other text.",
            "raw_data_doc_id": None # No raw cache id
        }

        structured_insight = await listener._structure_data(insights)

        self.assertIsInstance(structured_insight, StructuredInsight)
        self.assertIsInstance(structured_insight.summary, Summary)
        self.assertEqual(structured_insight.summary.text, "Another summary.")
        self.assertIsInstance(structured_insight.entities, list)
        self.assertEqual(len(structured_insight.entities), 1)
        self.assertEqual(structured_insight.entities[0].text, "OnlyEntity")
        self.assertIsNone(structured_insight.relations) # Should be None or empty list based on Pydantic default
        self.assertIsNone(structured_insight.raw_data_id)

    async def test_structure_data_minimal_insights(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        insights = {
            "summary": None,
            "entities": None,
            "relations": None,
            "original_data": None, # Original data might be None
            "raw_data_doc_id": "raw_only_123"
        }

        structured_insight = await listener._structure_data(insights)

        self.assertIsInstance(structured_insight, StructuredInsight)
        self.assertIsNone(structured_insight.summary)
        self.assertIsNone(structured_insight.entities)
        self.assertIsNone(structured_insight.relations)
        self.assertIsNone(structured_insight.original_data_type) # type of None is NoneType
        self.assertIsNone(structured_insight.source_data_preview)
        self.assertEqual(structured_insight.raw_data_id, "raw_only_123")

    async def test_structure_data_entity_creation_error(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        # Malformed entity data (e.g., missing 'type')
        insights = {
            "summary": "Summary here",
            "entities": [{"text": "GoodEntity", "type": "TYPE"}, {"text": "BadEntity"}], # Second entity is malformed
            "relations": None,
            "original_data": "Test data",
            "raw_data_doc_id": "doc1"
        }
        # Pydantic will raise a ValidationError if a required field is missing.
        # The current _structure_data catches general Exception and logs, then sets entities to [].

        with self.assertLogs(logger=listener.logger, level='ERROR') as cm:
            structured_insight = await listener._structure_data(insights)

        self.assertTrue(any("Error creating Entity objects" in log_msg for log_msg in cm.output))
        self.assertIsInstance(structured_insight.entities, list)
        self.assertEqual(len(structured_insight.entities), 0) # Entities list becomes empty on error

    # 7. Memory Writing Tests (_write_to_memory)
    async def test_write_to_memory_all_systems_present_and_successful(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        # Ensure all memory clients are fresh mocks for this test
        listener.memory_systems["stm"] = AsyncMock(spec=BaseMemorySystem)
        listener.memory_systems["ltm"] = AsyncMock(spec=BaseMemorySystem)
        listener.memory_systems["graph_db"] = AsyncMock(spec=BaseMemorySystem)

        summary_data = Summary(text="Test Summary")
        entity_data = [Entity(text="E1", type="T1")]
        relation_data = [Relation(subject="E1", verb="is", object="E2")]

        structured = StructuredInsight(
            summary=summary_data,
            entities=entity_data,
            relations=relation_data,
            raw_data_id="raw_123"
        )
        doc_id_base = "raw_123" # As it's prioritized

        await listener._write_to_memory(structured)

        listener.memory_systems["stm"].save_summary.assert_called_once_with(
            summary_id=f"{doc_id_base}_summary",
            summary_obj=summary_data,
            metadata=ANY
        )
        listener.memory_systems["ltm"].save_document.assert_called_once_with(
            doc_id=f"{doc_id_base}_ltm_doc",
            document_content=structured,
            metadata=ANY
        )
        listener.memory_systems["graph_db"].add_entities.assert_called_once_with(
            entities=entity_data,
            document_id=doc_id_base,
            metadata=ANY
        )
        listener.memory_systems["graph_db"].add_relations.assert_called_once_with(
            relations=relation_data,
            document_id=doc_id_base,
            metadata=ANY
        )

    async def test_write_to_memory_missing_stm_and_graph_db(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.memory_systems["stm"] = None
        listener.memory_systems["graph_db"] = None
        listener.memory_systems["ltm"] = AsyncMock(spec=BaseMemorySystem) # LTM is present

        structured = StructuredInsight(
            summary=Summary(text="Test Summary"),
            entities=[Entity(text="E1", type="T1")],
            raw_data_id="raw_456"
        )
        await listener._write_to_memory(structured)

        self.assertFalse(hasattr(listener.memory_systems["stm"], 'save_summary') or listener.memory_systems["stm"] is None) # No stm.save_summary
        listener.memory_systems["ltm"].save_document.assert_called_once() # LTM should be called
        self.assertFalse(hasattr(listener.memory_systems["graph_db"], 'add_entities') or listener.memory_systems["graph_db"] is None) # No graph_db.add_entities

    async def test_write_to_memory_ltm_raises_exception(self):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.memory_systems["ltm"] = AsyncMock(spec=BaseMemorySystem)
        listener.memory_systems["ltm"].save_document.side_effect = Exception("LTM DB Connection Error")
        # Keep other clients mocked to ensure they are still called or skipped appropriately
        listener.memory_systems["stm"] = AsyncMock(spec=BaseMemorySystem)
        listener.memory_systems["graph_db"] = AsyncMock(spec=BaseMemorySystem)


        structured = StructuredInsight(summary=Summary(text="S"), entities=[Entity(text="E", type="T")], raw_data_id="raw_789")

        with self.assertLogs(logger=listener.logger, level='ERROR') as cm:
            await listener._write_to_memory(structured)

        self.assertTrue(any("Failed to write document to LTM" in log_msg for log_msg in cm.output))
        # Check that other operations were still attempted
        listener.memory_systems["stm"].save_summary.assert_called_once()
        listener.memory_systems["graph_db"].add_entities.assert_called_once()

    # 8. Overall Processing Tests (process_data)
    @patch.object(LLMListener, '_write_to_memory', new_callable=AsyncMock)
    @patch.object(LLMListener, '_structure_data', new_callable=AsyncMock)
    @patch.object(LLMListener, '_generate_insights', new_callable=AsyncMock)
    @patch.object(LLMListener, '_preprocess_data', new_callable=AsyncMock)
    async def test_process_data_successful_flow_with_raw_cache(
        self, mock_preprocess, mock_generate_insights, mock_structure_data, mock_write_to_memory
    ):
        # Setup listener and mocks for memory systems if needed beyond self.setUp
        # For this test, self.mock_raw_cache from setUp will be used.
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.memory_systems['raw_cache'] = self.mock_raw_cache # Ensure it's the one we can assert on

        raw_data_input = "Initial raw data"
        preprocessed_data_output = "Processed data"
        raw_cache_doc_id_output = "raw_doc_id_test_1"
        insights_dict_output = {"summary": "S", "entities": [], "relations": [], "original_data": preprocessed_data_output, "raw_data_doc_id": raw_cache_doc_id_output}
        structured_insight_output = MagicMock(spec=StructuredInsight) # A mock StructuredInsight object
        structured_insight_output.raw_data_id = raw_cache_doc_id_output # Ensure it has the ID

        # Configure mock return values
        mock_preprocess.return_value = preprocessed_data_output
        self.mock_raw_cache.store = AsyncMock(return_value=raw_cache_doc_id_output) # Mock store on self.mock_raw_cache
        mock_generate_insights.return_value = insights_dict_output
        mock_structure_data.return_value = structured_insight_output
        mock_write_to_memory.return_value = None # _write_to_memory doesn't return anything

        context_instr = {"summarize": True}
        await listener.process_data(raw_data_input, context_instructions=context_instr)

        # Assertions
        mock_preprocess.assert_called_once_with(raw_data_input)
        self.mock_raw_cache.store.assert_called_once_with(doc_id=ANY, data=preprocessed_data_output) # doc_id is generated internally
        mock_generate_insights.assert_called_once_with(preprocessed_data_output, context_instr, raw_data_doc_id=raw_cache_doc_id_output)
        mock_structure_data.assert_called_once_with(insights_dict_output)
        mock_write_to_memory.assert_called_once_with(structured_insight_output)

    @patch.object(LLMListener, '_write_to_memory', new_callable=AsyncMock)
    @patch.object(LLMListener, '_structure_data', new_callable=AsyncMock)
    @patch.object(LLMListener, '_generate_insights', new_callable=AsyncMock)
    @patch.object(LLMListener, '_preprocess_data', new_callable=AsyncMock)
    async def test_process_data_successful_flow_no_raw_cache(
        self, mock_preprocess, mock_generate_insights, mock_structure_data, mock_write_to_memory
    ):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.memory_systems['raw_cache'] = None # No raw_cache client

        raw_data_input = "Initial raw data"
        preprocessed_data_output = "Processed data"
        # raw_cache_doc_id should be None
        insights_dict_output = {"summary": "S", "entities": [], "relations": [], "original_data": preprocessed_data_output, "raw_data_doc_id": None}
        structured_insight_output = MagicMock(spec=StructuredInsight)
        structured_insight_output.raw_data_id = None

        mock_preprocess.return_value = preprocessed_data_output
        mock_generate_insights.return_value = insights_dict_output
        mock_structure_data.return_value = structured_insight_output

        await listener.process_data(raw_data_input, context_instructions=None)

        mock_preprocess.assert_called_once_with(raw_data_input)
        mock_generate_insights.assert_called_once_with(preprocessed_data_output, None, raw_data_doc_id=None)
        mock_structure_data.assert_called_once_with(insights_dict_output)
        mock_write_to_memory.assert_called_once_with(structured_insight_output)

    @patch.object(LLMListener, '_write_to_memory', new_callable=AsyncMock)
    @patch.object(LLMListener, '_structure_data', new_callable=AsyncMock)
    @patch.object(LLMListener, '_generate_insights', new_callable=AsyncMock)
    @patch.object(LLMListener, '_preprocess_data', new_callable=AsyncMock)
    async def test_process_data_generate_insights_fails(
        self, mock_preprocess, mock_generate_insights, mock_structure_data, mock_write_to_memory
    ):
        listener = self.create_listener_with_patched_pipeline(MagicMock())
        listener.memory_systems['raw_cache'] = self.mock_raw_cache

        raw_data_input = "Initial raw data"
        preprocessed_data_output = "Processed data"
        raw_cache_doc_id_output = "raw_doc_id_test_err"

        mock_preprocess.return_value = preprocessed_data_output
        self.mock_raw_cache.store = AsyncMock(return_value=raw_cache_doc_id_output)
        mock_generate_insights.side_effect = Exception("Error during insight generation")

        with self.assertLogs(logger=listener.logger, level='ERROR') as cm:
            await listener.process_data(raw_data_input, context_instructions=None)

        self.assertTrue(any("Error during data processing pipeline" in log_msg for log_msg in cm.output))
        self.assertTrue(any("Error during insight generation" in log_msg for log_msg in cm.output))

        mock_preprocess.assert_called_once()
        self.mock_raw_cache.store.assert_called_once() # Raw cache store happens before insights
        mock_generate_insights.assert_called_once()
        mock_structure_data.assert_not_called() # Should not be called after failure
        mock_write_to_memory.assert_not_called() # Should not be called after failure


# Need to add AsyncMock to unittest.mock for older Python versions if not present
# For Python 3.8+ AsyncMock is part of unittest.mock
if not hasattr(unittest.mock, 'AsyncMock'):
    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super(AsyncMock, self).__call__(*args, **kwargs)
    unittest.mock.AsyncMock = AsyncMock


if __name__ == '__main__':
    unittest.main()
