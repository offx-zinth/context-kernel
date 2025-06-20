import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncio
import datetime

from contextkernel.core_logic.llm_listener import LLMListener, LLMListenerConfig, StructuredInsight, Summary, Entity, Relation
from contextkernel.core_logic.summarizer import Summarizer, SummarizerConfig # For mocking Summarizer
from contextkernel.core_logic.llm_retriever import HuggingFaceEmbeddingModel # For mocking
from contextkernel.core_logic.exceptions import MemoryAccessError


class TestLLMListenerGraphEnrichment(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.listener_config = LLMListenerConfig(
            embedding_model_name="test-embedding-model" # Needs a name for embedding model init
        )

        # Mock memory systems
        self.mock_raw_cache = AsyncMock()
        self.mock_stm = AsyncMock()
        self.mock_ltm = AsyncMock()
        self.mock_graph_db = AsyncMock()
        # Specific graph methods that will be called
        self.mock_graph_db.ensure_source_document_node = AsyncMock(return_value=True)
        self.mock_graph_db.add_memory_fragment_link = AsyncMock(return_value=True)
        self.mock_graph_db.add_entities_to_document = AsyncMock(return_value=True)
        self.mock_graph_db.add_relations_to_document = AsyncMock(return_value=True)

        self.memory_systems = {
            "raw_cache": self.mock_raw_cache,
            "stm": self.mock_stm,
            "ltm": self.mock_ltm,
            "graph_db": self.mock_graph_db
        }

        # Mock Summarizer
        self.mock_summarizer_instance = AsyncMock(spec=Summarizer)
        self.mock_summarizer_instance.summarize = AsyncMock(return_value="Test summary")
        self.patcher_summarizer = patch('contextkernel.core_logic.llm_listener.Summarizer', return_value=self.mock_summarizer_instance)
        self.mock_summarizer_cls = self.patcher_summarizer.start()

        # Mock Embedding Model
        self.mock_embedding_model_instance = AsyncMock(spec=HuggingFaceEmbeddingModel)
        self.mock_embedding_model_instance.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        self.patcher_embedding_model = patch('contextkernel.core_logic.llm_listener.HuggingFaceEmbeddingModel', return_value=self.mock_embedding_model_instance)
        self.mock_hf_embedding_model_cls = self.patcher_embedding_model.start()

        # Mock Pipelines (NER/RE)
        self.mock_ner_pipeline = AsyncMock(return_value=[
            {"word": "TestCo", "entity_group": "ORG", "score": 0.99},
            {"word": "Alice", "entity_group": "PER", "score": 0.98}
        ])
        self.mock_re_pipeline = AsyncMock(return_value=[
            {"subject": "Alice", "verb": "works_at", "object": "TestCo"}
        ])
        # Patch only if model names are configured (which they are by default for NER)
        if self.listener_config.entity_extraction_model_name or \
           self.listener_config.relation_extraction_model_name or \
           self.listener_config.general_llm_for_re_model_name:
            self.patcher_pipeline = patch('contextkernel.core_logic.llm_listener.pipeline')
            self.mock_pipeline_constructor = self.patcher_pipeline.start()
            # Configure side_effect to return specific mocks based on task
            def pipeline_side_effect(task, model, tokenizer=None):
                if task == "ner": return self.mock_ner_pipeline
                if task == "text2text-generation" or task == "text-generation": return self.mock_re_pipeline
                return MagicMock() # Default mock for other pipelines
            self.mock_pipeline_constructor.side_effect = pipeline_side_effect
        else:
            self.patcher_pipeline = None


        self.listener = LLMListener(self.listener_config, self.memory_systems)

        # Common raw_data and instructions
        self.raw_data = "Alice works at TestCo. It is a great company."
        self.context_instructions = {
            "summarize": True, "extract_entities": True, "extract_relations": True
        }
        self.raw_id = f"raw_{datetime.datetime.now(datetime.timezone.utc).isoformat()}"
        self.mock_raw_cache.store = AsyncMock(return_value=self.raw_id)


    def tearDown(self):
        self.patcher_summarizer.stop()
        self.patcher_embedding_model.stop()
        if self.patcher_pipeline:
            self.patcher_pipeline.stop()

    async def test_process_data_full_enrichment(self):
        # Configure mocks for full processing
        # NER pipeline is already mocked in setUp to return entities
        # RE pipeline is also mocked in setUp

        structured_insight = await self.listener.process_data(self.raw_data, self.context_instructions)
        self.assertIsNotNone(structured_insight)
        doc_id_base = structured_insight.raw_data_id or f"doc_{structured_insight.created_at.isoformat()}"

        # 1. Assert Source Document Node Creation
        self.mock_graph_db.ensure_source_document_node.assert_called_once()
        args_ensure_doc, _ = self.mock_graph_db.ensure_source_document_node.call_args
        self.assertEqual(args_ensure_doc[0], doc_id_base) # document_id
        self.assertIn("preview", args_ensure_doc[1])
        self.assertEqual(args_ensure_doc[1]["preview"], self.raw_data[:100] + "...")

        # Check that other operations were queued (we'll check calls after gather)
        # Need to inspect calls to memory_ops.append(...) which is tricky.
        # Instead, we check the final calls on the graph_db mock.

        # 2. Assert Memory Fragment Linking
        # STM
        self.mock_stm.save_summary.assert_called_once()
        stm_fragment_id = f"{doc_id_base}_summary"
        expected_stm_props = {
            "id": stm_fragment_id, "text": "Test summary",
            "source_document_id": doc_id_base, "type": "summary",
            "created_at": structured_insight.summary.created_at.isoformat()
        }

        # LTM
        self.mock_ltm.save_document.assert_called_once()
        ltm_fragment_id = f"{doc_id_base}_ltm_doc"
        expected_ltm_text = "Test summary" # Since summary was generated
        expected_ltm_props = {
            "id": ltm_fragment_id, "text_preview": expected_ltm_text[:255],
            "has_embedding": True, "source_document_id": doc_id_base,
            "type": "ltm_document_content", "created_at": structured_insight.created_at.isoformat()
        }

        # RawCache
        raw_cache_fragment_id = self.raw_id
        expected_raw_props = {
            "id": raw_cache_fragment_id, "type": "raw_data_log",
            "source_document_id": doc_id_base, "created_at": structured_insight.created_at.isoformat()
        }

        # Check add_memory_fragment_link calls
        # Need to iterate through calls as order within asyncio.gather is not guaranteed for these specific ones
        add_mem_frag_calls = self.mock_graph_db.add_memory_fragment_link.call_args_list
        self.assertTrue(any(
            call(document_id=doc_id_base, fragment_id=stm_fragment_id, fragment_main_label="STMEntry",
                 relationship_type="HAS_STM_REPRESENTATION", fragment_properties=unittest.mock.ANY) # ANY for props due to datetime object
            in add_mem_frag_calls
        ))
        # Actual check for properties for STM
        found_stm_call = False
        for call_args in add_mem_frag_calls:
            if call_args[1]['fragment_id'] == stm_fragment_id:
                self.assertDictEqual(call_args[1]['fragment_properties'], expected_stm_props)
                found_stm_call = True; break
        self.assertTrue(found_stm_call, "STM fragment properties not as expected or call not found.")

        found_ltm_call = False
        for call_args in add_mem_frag_calls:
            if call_args[1]['fragment_id'] == ltm_fragment_id:
                 self.assertDictEqual(call_args[1]['fragment_properties'], expected_ltm_props)
                 found_ltm_call = True; break
        self.assertTrue(found_ltm_call, "LTM fragment properties not as expected or call not found.")

        found_raw_call = False
        for call_args in add_mem_frag_calls:
            if call_args[1]['fragment_id'] == raw_cache_fragment_id:
                 self.assertDictEqual(call_args[1]['fragment_properties'], expected_raw_props)
                 found_raw_call = True; break
        self.assertTrue(found_raw_call, "RawCache fragment properties not as expected or call not found.")


        # 3. Assert Entities and Relations Linking
        self.mock_graph_db.add_entities_to_document.assert_called_once()
        args_entities, _ = self.mock_graph_db.add_entities_to_document.call_args
        self.assertEqual(args_entities[0], doc_id_base)
        self.assertEqual(len(args_entities[1]), 2) # Alice, TestCo
        self.assertEqual(args_entities[1][0]['text'], "TestCo") # NER output might not be ordered as input
        self.assertEqual(args_entities[1][1]['text'], "Alice")

        self.mock_graph_db.add_relations_to_document.assert_called_once()
        args_relations, _ = self.mock_graph_db.add_relations_to_document.call_args
        self.assertEqual(args_relations[0], doc_id_base)
        self.assertEqual(len(args_relations[1]), 1)
        self.assertEqual(args_relations[1][0]['subject'], "Alice")
        self.assertEqual(args_relations[1][0]['verb'], "works_at")


    async def test_process_data_only_summary(self):
        self.context_instructions = {"summarize": True, "extract_entities": False, "extract_relations": False}
        # NER and RE pipelines will not be called effectively
        self.listener.ner_pipeline = None # Explicitly disable for this test's clarity
        self.listener.re_pipeline = None
        self.listener.re_llm_pipeline = None


        structured_insight = await self.listener.process_data(self.raw_data, self.context_instructions)
        doc_id_base = structured_insight.raw_data_id or f"doc_{structured_insight.created_at.isoformat()}"

        self.mock_graph_db.ensure_source_document_node.assert_called_once_with(
            document_id=doc_id_base,
            properties=unittest.mock.ANY
        )
        # Check add_memory_fragment_link was called for STM
        # Other fragment links (LTM, RawCache) would also be called
        self.assertTrue(any(
            call[1]['fragment_main_label'] == "STMEntry" for call in self.mock_graph_db.add_memory_fragment_link.call_args_list
        ))

        self.mock_graph_db.add_entities_to_document.assert_called_once_with(document_id=doc_id_base, entities=[])
        self.mock_graph_db.add_relations_to_document.assert_called_once_with(document_id=doc_id_base, relations=[])

    async def test_process_data_no_graph_db(self):
        self.listener.memory_systems["graph_db"] = None # Remove graph_db

        structured_insight = await self.listener.process_data(self.raw_data, self.context_instructions)
        self.assertIsNotNone(structured_insight) # Should still process other parts

        self.mock_graph_db.ensure_source_document_node.assert_not_called()
        self.mock_graph_db.add_memory_fragment_link.assert_not_called()
        self.mock_graph_db.add_entities_to_document.assert_not_called()
        self.mock_graph_db.add_relations_to_document.assert_not_called()

        # Other memory systems should still be called
        self.mock_stm.save_summary.assert_called_once()
        self.mock_ltm.save_document.assert_called_once()
        self.mock_raw_cache.store.assert_called_once()

    async def test_entity_relation_transformation_to_dicts(self):
        # This tests the _write_to_memory part implicitly via process_data
        # We need to ensure Pydantic models are converted to dicts

        # Create a listener where graph_db methods are simple mocks that allow inspection of args
        graph_db_spy = AsyncMock()
        graph_db_spy.ensure_source_document_node = AsyncMock(return_value=True)
        graph_db_spy.add_memory_fragment_link = AsyncMock(return_value=True)
        graph_db_spy.add_entities_to_document = AsyncMock(return_value=True)
        graph_db_spy.add_relations_to_document = AsyncMock(return_value=True)

        memory_systems_with_spy = self.memory_systems.copy()
        memory_systems_with_spy["graph_db"] = graph_db_spy

        listener_with_spy = LLMListener(self.listener_config, memory_systems_with_spy)
        # Need to re-mock internal LLM calls for this new listener instance
        listener_with_spy._call_llm_summarize = AsyncMock(return_value="Test summary")
        listener_with_spy._call_llm_extract_entities = AsyncMock(return_value=[
            {"text": "Entity1", "type": "TYPEA", "metadata": {"k": "v"}}
        ])
        listener_with_spy._call_llm_extract_relations = AsyncMock(return_value=[
            {"subject": "Entity1", "verb": "links_to", "object": "Entity2", "context": "ctx"}
        ])
        listener_with_spy.embedding_model.generate_embedding = AsyncMock(return_value=[0.5,0.5]) # Mock for this instance
        listener_with_spy.raw_cache = self.mock_raw_cache # ensure raw_cache is part of this instance

        await listener_with_spy.process_data("Some data", {"summarize": True, "extract_entities": True, "extract_relations": True})

        # Check add_entities_to_document call
        graph_db_spy.add_entities_to_document.assert_called_once()
        args_entities, _ = graph_db_spy.add_entities_to_document.call_args
        self.assertIsInstance(args_entities[1], list)
        self.assertIsInstance(args_entities[1][0], dict) # Ensure it's a dict, not Pydantic model
        self.assertEqual(args_entities[1][0]["text"], "Entity1")
        self.assertIn("created_at", args_entities[1][0]) # model_dump includes these

        # Check add_relations_to_document call
        graph_db_spy.add_relations_to_document.assert_called_once()
        args_relations, _ = graph_db_spy.add_relations_to_document.call_args
        self.assertIsInstance(args_relations[1], list)
        self.assertIsInstance(args_relations[1][0], dict)
        self.assertEqual(args_relations[1][0]["subject"], "Entity1")
        self.assertIn("created_at", args_relations[1][0])


if __name__ == '__main__':
    unittest.main()
