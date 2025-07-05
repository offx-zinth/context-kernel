import logging
import asyncio
import datetime
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict # Ensure SettingsConfigDict is imported

try:
    from transformers import pipeline, Pipeline
except ImportError:
    pipeline = None 
    Pipeline = None # type: ignore 

from contextkernel.core_logic.summarizer import Summarizer, SummarizerConfig
from contextkernel.core_logic.llm_retriever import HuggingFaceEmbeddingModel
from .exceptions import (
    ConfigurationError, ExternalServiceError, EmbeddingError, 
    MemoryAccessError, SummarizationError, PipelineError, CoreLogicError
)

logger = logging.getLogger(__name__)

class LLMListenerConfig(BaseSettings):
    summarizer_config: SummarizerConfig = Field(default_factory=SummarizerConfig)
    entity_extraction_model_name: Optional[str] = "dbmdz/bert-large-cased-finetuned-conll03-english"
    relation_extraction_model_name: Optional[str] = None
    general_llm_for_re_model_name: Optional[str] = "distilgpt2"
    embedding_model_name: Optional[str] = "all-MiniLM-L6-v2"
    default_summarization_min_length: int = 30
    default_summarization_max_length: int = 150

    model_config = SettingsConfigDict(env_prefix='LLM_LISTENER_')


class BaseMemorySystem:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"{self.__class__.__name__} initialized.")

class RawCacheInterface(BaseMemorySystem):
    async def store(self, doc_id: str, data: Any) -> Optional[str]: raise NotImplementedError
    async def load(self, doc_id: str) -> Optional[Any]: raise NotImplementedError

class STMInterface(BaseMemorySystem):
    async def save_summary(self, summary_id: str, summary_obj: Any, metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError
    async def load_summary(self, summary_id: str) -> Optional[Any]: raise NotImplementedError

class LTMInterface(BaseMemorySystem):
    async def save_document(self, doc_id: str, text_content: str, embedding: List[float], metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError

class GraphDBInterface(BaseMemorySystem):
    async def add_entities(self, entities: List[Any], document_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError
    async def add_relations(self, relations: List[Any], document_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> None: raise NotImplementedError

class StubRawCache(RawCacheInterface):
    def __init__(self): super().__init__(); self.cache: Dict[str, Any] = {}
    async def store(self, doc_id: str, data: Any) -> Optional[str]: self.cache[doc_id] = data; return doc_id
    async def load(self, doc_id: str) -> Optional[Any]: return self.cache.get(doc_id)

class StubSTM(STMInterface):
    def __init__(self): super().__init__(); self.cache: Dict[str, Any] = {}
    async def save_summary(self, summary_id: str, summary_obj: Any, metadata: Optional[Dict[str, Any]]=None) -> None: self.cache[summary_id] = {"summary": summary_obj, "metadata": metadata or {}}
    async def load_summary(self, summary_id: str) -> Optional[Any]: return self.cache.get(summary_id)

class TimestampedModel(BaseModel):
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class Entity(TimestampedModel): text: str; type: str; metadata: Optional[Dict[str, Any]] = None
class Relation(TimestampedModel): subject: str; verb: str; object: str; context: Optional[str] = None; metadata: Optional[Dict[str, Any]] = None
class Summary(TimestampedModel): text: str; source_data_hash: Optional[str] = None
class StructuredInsight(TimestampedModel):
    original_data_type: Optional[str] = None; source_data_preview: Optional[str] = None
    summary: Optional[Summary] = None; entities: Optional[List[Entity]] = None
    relations: Optional[List[Relation]] = None; raw_data_id: Optional[str] = None
    content_embedding: Optional[List[float]] = None

class ContextAgent: # Renamed from LLMListener
    def __init__(self, listener_config: LLMListenerConfig, # Parameter name kept for compatibility during transition if settings are mapped
                 # memory_systems: Dict[str, BaseMemorySystem], # Removed
                 data_processing_config: Optional[Dict[str, Any]] = None,
                 llm_client: Optional[Any] = None,
                 chunker: Optional[Any] = None): # Added chunker dependency
        self.logger = logger 
        self.config = listener_config # Renamed attribute for clarity internal to this class
        # self.memory_systems = memory_systems # Removed
        self.data_processing_config = data_processing_config or {} # Kept for potential future use or more granular control
        self.llm_client = llm_client # Kept, as LLM client might be used by summarizer/NER/RE
        self.chunker = chunker # Store chunker instance

        if not self.chunker:
            # Fallback or default chunker initialization if desired, or raise error
            # For now, let's assume chunker is a required dependency.
            raise ConfigurationError("Chunker instance is required for ContextAgent.")

        try:
            # Summarizer might still be part of this agent's direct responsibilities for insight generation
            self.summarizer = Summarizer(self.config.summarizer_config)
        except ConfigurationError as e:
            self.logger.error(f"CRITICAL: Summarizer config failed: {e}", exc_info=True); raise
        
        if not self.config.embedding_model_name:
            # Embedding might be needed for the StructuredInsight if not handled by chunker/labeler
            raise ConfigurationError("embedding_model_name is required for ContextAgent.")
        try:
            self.embedding_model = HuggingFaceEmbeddingModel(model_name=self.config.embedding_model_name)
        except (EmbeddingError, ConfigurationError) as e:
            self.logger.error(f"CRITICAL: Embedding model init failed: {e}", exc_info=True); raise

        self.ner_pipeline, self.re_pipeline, self.re_llm_pipeline = None, None, None
        if pipeline is None:
            if any(getattr(self.config, m, None) for m in
                   ['entity_extraction_model_name', 'relation_extraction_model_name', 'general_llm_for_re_model_name']):
                raise ConfigurationError("Transformers 'pipeline' unavailable but NER/RE models configured.")
            self.logger.warning("Transformers 'pipeline' unavailable; NER/RE disabled.")
        else: self._initialize_pipelines()
        self.logger.info("ContextAgent initialized.")

    def _initialize_pipelines(self):
        if self.config.entity_extraction_model_name:
            try:
                self.ner_pipeline = pipeline("ner", model=self.config.entity_extraction_model_name, tokenizer=self.config.entity_extraction_model_name)
                self.logger.info(f"NER pipeline initialized: {self.config.entity_extraction_model_name}")
            except Exception as e: self.logger.error(f"Failed to init NER: {e}", exc_info=True)
        else: self.logger.info("NER disabled (no model name).")

        if self.config.relation_extraction_model_name:
            try:
                self.re_pipeline = pipeline("text2text-generation", model=self.config.relation_extraction_model_name, tokenizer=self.config.relation_extraction_model_name)
                self.logger.info(f"Dedicated RE pipeline initialized: {self.config.relation_extraction_model_name}")
            except Exception as e: self.logger.error(f"Failed to init dedicated RE: {e}", exc_info=True)
        elif self.config.general_llm_for_re_model_name:
            try:
                self.re_llm_pipeline = pipeline("text-generation", model=self.config.general_llm_for_re_model_name, tokenizer=self.config.general_llm_for_re_model_name)
                self.logger.info(f"General LLM for RE initialized: {self.config.general_llm_for_re_model_name}")
            except Exception as e: self.logger.error(f"Failed to init general LLM for RE: {e}", exc_info=True)
        else: self.logger.info("RE disabled (no model name).")

    # _preprocess_data might be simplified or removed if chunker handles initial processing
    # async def _preprocess_data(self, raw_data: Any) -> str:
    #     self.logger.debug(f"Preprocessing data (type: {type(raw_data)}).")
    #     return str(raw_data)

    async def _call_llm_summarize(self, text_content: str, instructions: Optional[Dict[str, Any]]=None) -> Optional[str]:
        if not self.summarizer: raise ConfigurationError("Summarizer unavailable.")
        custom_conf = None
        if instructions:
            try: custom_conf = SummarizerConfig(**{**self.summarizer.default_config.model_dump(), **{k:v for k,v in instructions.items() if k in self.summarizer.default_config.model_fields}})
            except Exception as e: self.logger.warning(f"Failed to make custom SummarizerConfig: {e}", exc_info=True)
        try: return await self.summarizer.summarize(text_content, config=custom_conf)
        except (SummarizationError, ExternalServiceError) as e: self.logger.error(f"Summarization failed: {e}", exc_info=True); raise
        except Exception as e: raise SummarizationError(f"Unexpected summarizer error: {e}") from e

    async def _call_llm_extract_entities(self, text_content: str, instructions: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        if not self.ner_pipeline: self.logger.warning("NER pipeline unavailable."); return []
        try:
            results = self.ner_pipeline(text_content)
            return [{"text": e.get('word'), "type": e.get('entity_group', e.get('label', 'UNKNOWN')), **e} for e in results] if results else []
        except Exception as e: raise ExternalServiceError(f"NER pipeline failed: {e}") from e

    async def _call_llm_extract_relations(self, text_content: str, entities: Optional[List[Dict[str, Any]]], instructions: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        found = []
        try:
            if self.re_pipeline:
                outputs = self.re_pipeline(text_content)
                if isinstance(outputs, list): found.extend(r for r in outputs if isinstance(r, dict) and all(k in r for k in ['subject','relation','object']))
            elif self.re_llm_pipeline:
                if not self.re_llm_pipeline.tokenizer: raise ConfigurationError("RE LLM tokenizer missing.")
                prompt = f"Entities: {entities}. Extract relations (Subj; Pred; Obj) from: \"{text_content}\""
                outputs = self.re_llm_pipeline(prompt, max_length=len(self.re_llm_pipeline.tokenizer.encode(prompt)) + 150)
                if outputs and outputs[0]['generated_text']:
                    for line in outputs[0]['generated_text'][len(prompt):].strip().split('\n'):
                        match = re.match(r'\(\s*(.*?)\s*;\s*(.*?)\s*;\s*(.*?)\s*\)', line.strip())
                        if match: found.append({"subject": match.group(1), "verb": match.group(2), "object": match.group(3)})
        except Exception as e: raise ExternalServiceError(f"RE pipeline failed: {e}") from e
        return found

    async def _generate_insights(self, chunk_text: str, chunk_label: Optional[Dict[str, Any]], instructions: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Generates insights for a single chunk of text.
        The 'raw_id' parameter is removed as raw data storage is handled before chunking,
        or the MemoryManager will handle it based on the StructuredInsight.
        """
        instructions = instructions or {}
        summary, entities, relations, embedding = None, None, None, None

        # Use chunk_label if available to guide insight generation (e.g. skip summarization for "code_block")
        # This is a placeholder for more sophisticated logic based on labels.
        is_question = chunk_label.get("intent") == "question" if chunk_label else False

        if instructions.get("summarize", True) and not is_question: # Example: don't summarize questions
             summary = await self._call_llm_summarize(chunk_text, instructions.get('summarization_params'))

        if instructions.get("extract_entities", True):
            entities = await self._call_llm_extract_entities(chunk_text, instructions.get('entity_params'))
        
        if instructions.get("extract_relations", True) and entities: # Relations often depend on entities
            relations = await self._call_llm_extract_relations(chunk_text, entities, instructions.get('relation_params'))

        if chunk_text.strip() and self.embedding_model:
            try:
                embedding = await self.embedding_model.generate_embedding(chunk_text)
            except EmbeddingError as e:
                self.logger.error(f"Content embedding for chunk failed: {e}", exc_info=True) # Non-critical for this chunk

        self.logger.info("Generated insights for chunk.")
        return {
            "summary_text": summary,
            "entities_data": entities,
            "relations_data": relations,
            "original_chunk_text": chunk_text, # Keep the original chunk text
            "chunk_label": chunk_label, # Store the label associated with this chunk
            "content_embedding_data": embedding
        }

    async def _structure_data(self, insights_for_chunk: Dict[str, Any]) -> StructuredInsight:
        """
        Packages insights from a single chunk into a StructuredInsight object.
        The raw_data_id is no longer set here directly; MemoryManager might associate it later.
        """
        self.logger.info("Packaging chunk insights into StructuredInsight.")
        return StructuredInsight(
            original_data_type="text_chunk", # Indicate this insight is from a chunk
            source_data_preview=insights_for_chunk.get("original_chunk_text", "")[:100] + "...", # Preview of the chunk
            summary=Summary(text=insights_for_chunk["summary_text"]) if insights_for_chunk.get("summary_text") else None,
            entities=[Entity(**e) for e in insights_for_chunk.get("entities_data", []) if e], # Ensure 'e' is not None
            relations=[Relation(**r) for r in insights_for_chunk.get("relations_data", []) if r], # Ensure 'r' is not None
            # raw_data_id is removed here. It will be managed by MemoryManager or the calling loop.
            content_embedding=insights_for_chunk.get("content_embedding_data"),
            # Potentially add chunk_label or other chunk-specific metadata to StructuredInsight if needed
            # metadata={"chunk_label": insights_for_chunk.get("chunk_label")} # Example
        )

    # _write_to_memory method is entirely REMOVED from this class.
    # Its logic now belongs in memory_system/memory_manager.py and is called
    # by the write_and_verify_loop in main.py.


    async def process_data(self, raw_data_content: str, context_instructions: Optional[Dict[str, Any]]=None) -> List[StructuredInsight]:
        """
        Processes raw data content:
        1. Chunks the data using the provided chunker.
        2. For each chunk, labels it.
        3. For each labeled chunk, generates insights.
        4. Structures the insights into a StructuredInsight object.
        Returns a list of StructuredInsight objects, one for each processed chunk.
        """
        self.logger.info(f"Processing raw data content. Instructions: {context_instructions}")
        all_structured_insights: List[StructuredInsight] = []

        try:
            if not raw_data_content.strip():
                self.logger.warning("Empty raw_data_content provided.")
                return []

            # 1. Chunk the data
            # Assuming chunker.split_text returns a list of text chunks
            # And chunker.label_chunk returns a dictionary of labels for a chunk
            # These methods might need to be async if they involve I/O or heavy computation

            # For simplicity, assuming chunker has a combined method or we call them sequentially.
            # Let's assume chunker.chunk_and_label(content) -> List[Tuple[str, Dict]]
            # If not, we'd do:
            # text_chunks = await self.chunker.split_text(raw_data_content)
            # for chunk_text in text_chunks:
            #     chunk_label = await self.chunker.label_chunk(chunk_text)
            #     ...

            # Using a placeholder for chunker interaction as its final API is defined in a later step.
            # This is a conceptual representation.
            # Example: self.chunker might yield (chunk_text, chunk_label_dict)

            # Placeholder: Simple split by newline for now, assuming chunker is not fully integrated.
            # Replace this with actual chunker calls once chunker.py is updated.
            text_chunks = raw_data_content.split('\n\n') # Simple placeholder
            if not self.chunker: # Should have been caught in __init__
                 raise ConfigurationError("Chunker not available for processing data.")

            for chunk_text in text_chunks:
                if not chunk_text.strip():
                    continue

                # 2. Label each chunk (assuming chunker.label_chunk is async)
                chunk_label = await self.chunker.label_chunk(chunk_text) # Actual call to chunker
                self.logger.info(f"Chunk labeled: {chunk_label}")

                # 3. Generate insights for the current chunk
                # raw_id is removed from _generate_insights call
                insights = await self._generate_insights(chunk_text, chunk_label, context_instructions)

                # 4. Structure the insights for the current chunk
                structured_insight_for_chunk = await self._structure_data(insights)
                all_structured_insights.append(structured_insight_for_chunk)

            self.logger.info(f"Data processing complete. Generated {len(all_structured_insights)} structured insights.")
            return all_structured_insights

        except (ConfigurationError, SummarizationError, EmbeddingError, ExternalServiceError, PipelineError, CoreLogicError) as e:
            self.logger.error(f"Core logic error in ContextAgent.process_data: {type(e).__name__} - {e}", exc_info=True); raise
        except Exception as e:
            self.logger.error(f"Unexpected error in ContextAgent.process_data: {e}", exc_info=True)
            raise CoreLogicError(f"Unexpected error in ContextAgent.process_data: {e}") from e
