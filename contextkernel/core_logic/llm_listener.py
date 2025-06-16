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
        return {"summary": summary, "entities": entities, "relations": relations, "original_data": data, "raw_data_doc_id": raw_id, "content_embedding": embedding}

    async def _structure_data(self, insights: Dict[str, Any]) -> StructuredInsight:
        return StructuredInsight(
            original_data_type=type(insights.get("original_data")).__name__,
            source_data_preview=insights.get("original_data", "")[:100] + "...",
            summary=Summary(text=insights["summary"]) if insights.get("summary") else None,
            entities=[Entity(**e) for e in insights.get("entities", [])],
            relations=[Relation(**r) for r in insights.get("relations", [])],
            raw_data_id=insights.get("raw_data_doc_id"),
            content_embedding=insights.get("content_embedding"))

    async def _write_to_memory(self, structured_data: StructuredInsight) -> None:
        doc_id_base = structured_data.raw_data_id or structured_data.created_at.isoformat()
        ops = []
        if structured_data.summary and "stm" in self.memory_systems:
            ops.append(self.memory_systems["stm"].save_summary(summary_id=f"{doc_id_base}_summary", summary_obj=structured_data.summary, metadata={"doc_id_base": doc_id_base, "raw_data_id": structured_data.raw_data_id}))
        if "ltm" in self.memory_systems and (s_prev := structured_data.source_data_preview or (structured_data.summary and structured_data.summary.text)):
            ops.append(self.memory_systems["ltm"].save_document(doc_id=f"{doc_id_base}_ltm", text_content=s_prev, embedding=structured_data.content_embedding or [], metadata={"doc_id_base": doc_id_base, "raw_data_id": structured_data.raw_data_id}))
        if "graph_db" in self.memory_systems:
            if structured_data.entities: ops.append(self.memory_systems["graph_db"].add_entities(entities=structured_data.entities, document_id=doc_id_base, metadata={"raw_data_id": structured_data.raw_data_id}))
            if structured_data.relations: ops.append(self.memory_systems["graph_db"].add_relations(relations=structured_data.relations, document_id=doc_id_base, metadata={"raw_data_id": structured_data.raw_data_id}))
        
        results = await asyncio.gather(*ops, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception): self.logger.error(f"Memory op {i} failed: {res}", exc_info=res); raise MemoryAccessError(f"Memory operation failed: {res}") from res

    async def process_data(self, raw_data: Any, context_instructions: Optional[Dict[str, Any]]=None) -> Optional[StructuredInsight]:
        self.logger.info(f"Processing data. Instructions: {context_instructions}")
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
