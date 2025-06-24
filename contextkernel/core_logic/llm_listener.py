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

class LLMListener:
    def __init__(self, listener_config: LLMListenerConfig, memory_systems: Dict[str, BaseMemorySystem],
                 data_processing_config: Optional[Dict[str, Any]] = None, llm_client: Optional[Any] = None):
        self.logger = logger 
        self.listener_config = listener_config
        self.memory_systems = memory_systems
        self.data_processing_config = data_processing_config or {}
        self.llm_client = llm_client

        try:
            self.summarizer = Summarizer(self.listener_config.summarizer_config)
        except ConfigurationError as e:
            self.logger.error(f"CRITICAL: Summarizer config failed: {e}", exc_info=True); raise
        
        if not self.listener_config.embedding_model_name:
            raise ConfigurationError("embedding_model_name is required for LLMListener.")
        try:
            self.embedding_model = HuggingFaceEmbeddingModel(model_name=self.listener_config.embedding_model_name)
        except (EmbeddingError, ConfigurationError) as e:
            self.logger.error(f"CRITICAL: Embedding model init failed: {e}", exc_info=True); raise

        self.ner_pipeline, self.re_pipeline, self.re_llm_pipeline = None, None, None
        if pipeline is None:
            if any(getattr(self.listener_config, m, None) for m in 
                   ['entity_extraction_model_name', 'relation_extraction_model_name', 'general_llm_for_re_model_name']):
                raise ConfigurationError("Transformers 'pipeline' unavailable but NER/RE models configured.")
            self.logger.warning("Transformers 'pipeline' unavailable; NER/RE disabled.")
        else: self._initialize_pipelines()
        self.logger.info("LLMListener initialized.")

    def _initialize_pipelines(self):
        if self.listener_config.entity_extraction_model_name:
            try:
                self.ner_pipeline = pipeline("ner", model=self.listener_config.entity_extraction_model_name, tokenizer=self.listener_config.entity_extraction_model_name)
                self.logger.info(f"NER pipeline initialized: {self.listener_config.entity_extraction_model_name}")
            except Exception as e: self.logger.error(f"Failed to init NER: {e}", exc_info=True)
        else: self.logger.info("NER disabled (no model name).")

        if self.listener_config.relation_extraction_model_name:
            try:
                self.re_pipeline = pipeline("text2text-generation", model=self.listener_config.relation_extraction_model_name, tokenizer=self.listener_config.relation_extraction_model_name)
                self.logger.info(f"Dedicated RE pipeline initialized: {self.listener_config.relation_extraction_model_name}")
            except Exception as e: self.logger.error(f"Failed to init dedicated RE: {e}", exc_info=True)
        elif self.listener_config.general_llm_for_re_model_name:
            try:
                self.re_llm_pipeline = pipeline("text-generation", model=self.listener_config.general_llm_for_re_model_name, tokenizer=self.listener_config.general_llm_for_re_model_name)
                self.logger.info(f"General LLM for RE initialized: {self.listener_config.general_llm_for_re_model_name}")
            except Exception as e: self.logger.error(f"Failed to init general LLM for RE: {e}", exc_info=True)
        else: self.logger.info("RE disabled (no model name).")

    async def _preprocess_data(self, raw_data: Any) -> str:
        self.logger.debug(f"Preprocessing data (type: {type(raw_data)}).")
        return str(raw_data)

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

    async def _generate_insights(self, data: str, instructions: Optional[Dict[str, Any]], raw_id: Optional[str]) -> Dict[str, Any]:
        summary, entities, relations, embedding = None, None, None, None
        if instructions.get("summarize", True): summary = await self._call_llm_summarize(data, instructions.get('summarization_params'))
        if instructions.get("extract_entities", True): entities = await self._call_llm_extract_entities(data, instructions.get('entity_params'))
        if instructions.get("extract_relations", True): relations = await self._call_llm_extract_relations(data, entities, instructions.get('relation_params'))
        if data.strip() and self.embedding_model:
            try: embedding = await self.embedding_model.generate_embedding(data)
            except EmbeddingError as e: self.logger.error(f"Content embedding failed: {e}", exc_info=True) # Non-critical
        self.logger.info("Generating insights (summary, entities, relations, embedding) to structure the data.")
        return {"summary": summary, "entities": entities, "relations": relations, "original_data": data, "raw_data_doc_id": raw_id, "content_embedding": embedding}

    async def _structure_data(self, insights: Dict[str, Any]) -> StructuredInsight:
        self.logger.info("Packaging insights into StructuredInsight.")
        return StructuredInsight(
            original_data_type=type(insights.get("original_data")).__name__,
            source_data_preview=insights.get("original_data", "")[:100] + "...",
            summary=Summary(text=insights["summary"]) if insights.get("summary") else None,
            entities=[Entity(**e) for e in insights.get("entities", [])],
            relations=[Relation(**r) for r in insights.get("relations", [])],
            raw_data_id=insights.get("raw_data_doc_id"),
            content_embedding=insights.get("content_embedding"))

    async def _write_to_memory(self, structured_data: StructuredInsight) -> None:
        self.logger.info(f"Persisting structured insights (Raw ID: {structured_data.raw_data_id or 'N/A'}) and enriching graph using new GraphDB methods.")
        doc_id_base = structured_data.raw_data_id or f"doc_{structured_data.created_at.isoformat()}"
        
        memory_ops = []

        # Standard memory operations (STM, LTM) - these remain largely the same
        stm_system = self.memory_systems.get("stm")
        if structured_data.summary and stm_system:
            self.logger.debug(f"Queueing STM save for summary of {doc_id_base}")
            stm_summary_id = f"{doc_id_base}_summary"
            memory_ops.append(stm_system.save_summary(
                summary_id=stm_summary_id,
                summary_obj=structured_data.summary,
                metadata={
                    "doc_id_base": doc_id_base,
                    "raw_data_id": structured_data.raw_data_id,
                    "source_preview": structured_data.source_data_preview,
                    "type": "summary" # Added type for clarity
                }
            ))

        ltm_content = structured_data.source_data_preview
        if structured_data.summary and structured_data.summary.text:
            ltm_content = structured_data.summary.text

        ltm_system = self.memory_systems.get("ltm")
        if ltm_system and ltm_content:
            self.logger.debug(f"Queueing LTM save for document {doc_id_base}")
            ltm_doc_id = f"{doc_id_base}_ltm_doc"
            memory_ops.append(ltm_system.save_document(
                doc_id=ltm_doc_id,
                text_content=ltm_content,
                embedding=structured_data.content_embedding or [],
                metadata={
                    "doc_id_base": doc_id_base,
                    "raw_data_id": structured_data.raw_data_id,
                    "original_data_type": structured_data.original_data_type,
                    "type": "document_content" # Added type for clarity
                }
            ))

        # Graph enrichment operations using new GraphDB methods
        graph_db = self.memory_systems.get("graph_db")
        if graph_db:
            self.logger.info(f"Starting graph enrichment for SourceDocument ID: {doc_id_base} using new methods.")

            source_doc_props = {
                "raw_data_id": structured_data.raw_data_id,
                "preview": structured_data.source_data_preview,
                "original_data_type": structured_data.original_data_type,
                "created_at": structured_data.created_at.isoformat(),
                "updated_at": structured_data.updated_at.isoformat() # Assuming GraphDB methods handle internal updated_at
            }
            # Ensure SourceDocument node is created first as other operations depend on it.
            # This call is awaitable and should complete before subsequent graph ops are queued if they depend on it.
            try:
                await graph_db.ensure_source_document_node(document_id=doc_id_base, properties=source_doc_props)
                self.logger.info(f"SourceDocument node ensured for ID: {doc_id_base}")

                # Link STM entry if summary exists
                if structured_data.summary and stm_system: # Check stm_system as well, though save_summary is already conditional
                    stm_fragment_id = f"{doc_id_base}_summary" # Consistent with save_summary id
                    stm_node_props = {
                        "id": stm_fragment_id, # Store its own ID
                        "text": structured_data.summary.text,
                        "source_document_id": doc_id_base, # Link back to source
                        "type": "summary",
                        "created_at": structured_data.summary.created_at.isoformat()
                    }
                    memory_ops.append(graph_db.add_memory_fragment_link(
                        document_id=doc_id_base,
                        fragment_id=stm_fragment_id,
                        fragment_main_label="STMEntry",
                        relationship_type="HAS_STM_REPRESENTATION",
                        fragment_properties=stm_node_props
                    ))
                    self.logger.debug(f"Queueing STMEntry link for {stm_fragment_id} to {doc_id_base}")

                # Link LTM entry
                if ltm_system and ltm_content: # Check ltm_system as well
                    ltm_fragment_id = f"{doc_id_base}_ltm_doc" # Consistent with save_document id
                    ltm_node_props = {
                        "id": ltm_fragment_id,
                        "text_preview": ltm_content[:255], # Store a preview in graph, full content in LTM system
                        "has_embedding": bool(structured_data.content_embedding),
                        "source_document_id": doc_id_base,
                        "type": "ltm_document_content",
                        "created_at": structured_data.created_at.isoformat() # Use main insight's timestamp
                    }
                    memory_ops.append(graph_db.add_memory_fragment_link(
                        document_id=doc_id_base,
                        fragment_id=ltm_fragment_id,
                        fragment_main_label="LTMLogEntry",
                        relationship_type="HAS_LTM_REPRESENTATION",
                        fragment_properties=ltm_node_props
                    ))
                    self.logger.debug(f"Queueing LTMLogEntry link for {ltm_fragment_id} to {doc_id_base}")

                # Link Raw Cache entry if raw_data_id exists
                if structured_data.raw_data_id: # raw_data_id is the ID for RawCacheEntry
                    raw_cache_node_props = {
                        "id": structured_data.raw_data_id,
                        "type": "raw_data_log",
                        "source_document_id": doc_id_base,
                         "created_at": structured_data.created_at.isoformat() # Assuming same creation time
                    }
                    memory_ops.append(graph_db.add_memory_fragment_link(
                        document_id=doc_id_base,
                        fragment_id=structured_data.raw_data_id,
                        fragment_main_label="RawCacheEntry",
                        relationship_type="REFERENCES_RAW_CACHE", # Or HAS_RAW_CACHE_REPRESENTATION
                        fragment_properties=raw_cache_node_props
                    ))
                    self.logger.debug(f"Queueing RawCacheEntry link for {structured_data.raw_data_id} to {doc_id_base}")

                # Add entities, linking them to doc_id_base
                if structured_data.entities:
                    entities_as_dicts = [entity.model_dump(exclude_none=True) for entity in structured_data.entities]
                    memory_ops.append(graph_db.add_entities_to_document(document_id=doc_id_base, entities=entities_as_dicts))
                    self.logger.debug(f"Queueing {len(entities_as_dicts)} entities for document {doc_id_base}")

                # Add relations, linking them to doc_id_base (implicitly via entities)
                if structured_data.relations:
                    relations_as_dicts = [relation.model_dump(exclude_none=True) for relation in structured_data.relations]
                    memory_ops.append(graph_db.add_relations_to_document(document_id=doc_id_base, relations=relations_as_dicts))
                    self.logger.debug(f"Queueing {len(relations_as_dicts)} relations for document {doc_id_base}")

            except Exception as e_graph_init: # Catch error from ensure_source_document_node
                self.logger.error(f"GraphDB: Failed to ensure source document node for {doc_id_base}: {e_graph_init}", exc_info=True)
                # Potentially raise MemoryAccessError here if this is critical

        # Execute all queued operations (STM, LTM, and new GraphDB operations)
        if memory_ops:
            results = await asyncio.gather(*memory_ops, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    self.logger.error(f"Memory op {i} failed: {res}", exc_info=True)
                    raise MemoryAccessError(f"At least one memory operation failed: {res}") from res
            self.logger.info(f"All memory operations for {doc_id_base} (STM, LTM, GraphDB links) completed.")
        else:
            self.logger.info(f"No memory operations were queued for {doc_id_base}.")


    async def process_data(self, raw_data: Any, context_instructions: Optional[Dict[str, Any]]=None) -> Optional[StructuredInsight]:
        self.logger.info(f"Optimizing and structuring raw data. Instructions: {context_instructions}")
        raw_id: Optional[str] = None
        try:
            data_str = await self._preprocess_data(raw_data)
            if not data_str.strip(): self.logger.warning("Empty preprocessed data."); return None
            if rc := self.memory_systems.get("raw_cache"):
                try: raw_id = await rc.store(doc_id=f"raw_{datetime.datetime.now(datetime.timezone.utc).isoformat()}", data=data_str) # type: ignore
                except Exception as e: self.logger.error(f"RawCache store failed: {e}", exc_info=True) # Non-critical for flow

            insights = await self._generate_insights(data_str, context_instructions or {}, raw_id)
            structured = await self._structure_data(insights)
            await self._write_to_memory(structured)
            self.logger.info("Data processing complete.")
            return structured
        except (ConfigurationError, SummarizationError, EmbeddingError, ExternalServiceError, PipelineError, MemoryAccessError, CoreLogicError) as e:
            self.logger.error(f"Core logic error: {type(e).__name__} - {e}", exc_info=True); raise
        except Exception as e:
            self.logger.error(f"Unexpected error in process_data: {e}", exc_info=True)
            raise CoreLogicError(f"Unexpected error in data processing: {e}") from e
