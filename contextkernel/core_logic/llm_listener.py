import logging
import asyncio
import datetime
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ContextKernel imports
from contextkernel.core_logic.summarizer import Summarizer, SummarizerConfig
from contextkernel.core_logic.llm_retriever import HuggingFaceEmbeddingModel # For embedding generation
from contextkernel.core_logic.chunker import SemanticChunker # New dependency
from contextkernel.core_logic.hallucination_detector import HallucinationDetector # New dependency
from contextkernel.memory_system.memory_manager import MemoryManager # New dependency
from contextkernel.interfaces.api import IngestData, IngestResponse # For ingest_data method
from contextkernel.utils.config import AppSettings # For ingest_data method

from .exceptions import (
    ConfigurationError, ExternalServiceError, EmbeddingError,
    SummarizationError, PipelineError, CoreLogicError, MemoryAccessError # Added MemoryAccessError
)

try:
    from transformers import pipeline, Pipeline
except ImportError:
    pipeline = None
    Pipeline = None # type: ignore

logger = logging.getLogger(__name__)

class ContextAgentConfig(BaseSettings): # Renamed from LLMListenerConfig
    summarizer_config: SummarizerConfig = Field(default_factory=SummarizerConfig)
    entity_extraction_model_name: Optional[str] = "dbmdz/bert-large-cased-finetuned-conll03-english"
    relation_extraction_model_name: Optional[str] = None
    general_llm_for_re_model_name: Optional[str] = "distilgpt2"
    embedding_model_name: Optional[str] = "all-MiniLM-L6-v2" # Used for insights if not by chunker
    default_summarization_min_length: int = 30
    default_summarization_max_length: int = 150

    # Prefix for environment variable loading, e.g., CONTEXT_AGENT_EMBEDDING_MODEL_NAME
    model_config = SettingsConfigDict(env_prefix='CONTEXT_AGENT_')


# Data models (Entity, Relation, Summary, StructuredInsight) remain crucial.
# These could potentially be moved to a shared types module later if used by many components.
class TimestampedModel(BaseModel):
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class Entity(TimestampedModel):
    text: str
    type: str
    metadata: Optional[Dict[str, Any]] = None

class Relation(TimestampedModel):
    subject: str
    verb: str
    object: str
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Summary(TimestampedModel):
    text: str
    source_data_hash: Optional[str] = None # Could be hash of chunk text

class StructuredInsight(TimestampedModel):
    original_data_type: Optional[str] = None # e.g., "text_chunk", "document_section"
    source_data_preview: Optional[str] = None # Preview of the original chunk/data
    summary: Optional[Summary] = None
    entities: Optional[List[Entity]] = Field(default_factory=list)
    relations: Optional[List[Relation]] = Field(default_factory=list)
    raw_data_id: Optional[str] = None # ID linking to raw data in RawCache, set by MemoryManager or ingest flow
    content_embedding: Optional[List[float]] = None # Embedding of the chunk's content or summary
    chunk_label: Optional[Dict[str, Any]] = None # Label from SemanticChunker


class ContextAgent: # Renamed from LLMListener
    def __init__(self,
                 config: ContextAgentConfig,
                 chunker: SemanticChunker,
                 hallucination_detector: HallucinationDetector, # Added
                 memory_manager: MemoryManager, # Added
                 llm_client: Optional[Any] = None, # For summarizer, NER, RE
                 data_processing_config: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.config = config
        self.chunker = chunker
        self.hallucination_detector = hallucination_detector # Stored for ingest_data
        self.memory_manager = memory_manager # Stored for ingest_data
        self.llm_client = llm_client # Used by summarizer, NER, RE pipelines
        self.data_processing_config = data_processing_config or {}

        if not self.chunker:
            raise ConfigurationError("SemanticChunker instance is required for ContextAgent.")
        if not self.hallucination_detector:
            raise ConfigurationError("HallucinationDetector instance is required for ContextAgent.")
        if not self.memory_manager:
            raise ConfigurationError("MemoryManager instance is required for ContextAgent.")

        try:
            self.summarizer = Summarizer(self.config.summarizer_config, llm_client=self.llm_client)
        except ConfigurationError as e:
            self.logger.error(f"CRITICAL: Summarizer config failed: {e}", exc_info=True); raise

        if not self.config.embedding_model_name:
            self.logger.warning("ContextAgent: embedding_model_name not configured. Content embedding for insights will be skipped.")
            self.embedding_model = None
        else:
            try:
                self.embedding_model = HuggingFaceEmbeddingModel(model_name=self.config.embedding_model_name)
            except (EmbeddingError, ConfigurationError) as e:
                self.logger.error(f"CRITICAL: Embedding model init failed: {e}", exc_info=True); raise

        self.ner_pipeline, self.re_pipeline, self.re_llm_pipeline = None, None, None
        if pipeline is None: # Check if transformers.pipeline is available
            if any(getattr(self.config, m, None) for m in
                   ['entity_extraction_model_name', 'relation_extraction_model_name', 'general_llm_for_re_model_name']):
                # Only raise if models are configured but pipeline factory is missing
                raise ConfigurationError("Transformers 'pipeline' factory is unavailable but NER/RE models are configured.")
            self.logger.warning("Transformers 'pipeline' factory unavailable; NER/RE features disabled.")
        else:
            self._initialize_pipelines() # Initialize NER/RE pipelines if factory is present

        self.logger.info("ContextAgent initialized successfully.")

    def _initialize_pipelines(self):
        # This method remains largely the same, sets up NER/RE pipelines if models are configured
        if self.config.entity_extraction_model_name:
            try:
                # Pass llm_client if pipeline expects it, though HF pipelines usually manage their own models
                self.ner_pipeline = pipeline("ner", model=self.config.entity_extraction_model_name, tokenizer=self.config.entity_extraction_model_name)
                self.logger.info(f"NER pipeline initialized with model: {self.config.entity_extraction_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize NER pipeline: {e}", exc_info=True)
                self.ner_pipeline = None # Ensure it's None if init fails
        else:
            self.logger.info("NER pipeline disabled (no model name configured).")

        if self.config.relation_extraction_model_name:
            try:
                self.re_pipeline = pipeline("text2text-generation", model=self.config.relation_extraction_model_name, tokenizer=self.config.relation_extraction_model_name)
                self.logger.info(f"Dedicated RE pipeline initialized with model: {self.config.relation_extraction_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize dedicated RE pipeline: {e}", exc_info=True)
                self.re_pipeline = None
        elif self.config.general_llm_for_re_model_name: # Fallback to general LLM for RE
            try:
                self.re_llm_pipeline = pipeline("text-generation", model=self.config.general_llm_for_re_model_name, tokenizer=self.config.general_llm_for_re_model_name)
                self.logger.info(f"General LLM for RE initialized with model: {self.config.general_llm_for_re_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize general LLM for RE pipeline: {e}", exc_info=True)
                self.re_llm_pipeline = None
        else:
            self.logger.info("Relation Extraction pipeline disabled (no model names configured).")


    async def _call_llm_summarize(self, text_content: str, instructions: Optional[Dict[str, Any]]=None) -> Optional[str]:
        # This method remains largely the same
        if not self.summarizer: raise ConfigurationError("Summarizer unavailable.")
        custom_conf = None
        if instructions:
            try: custom_conf = SummarizerConfig(**{**self.summarizer.default_config.model_dump(), **{k:v for k,v in instructions.items() if k in self.summarizer.default_config.model_fields}})
            except Exception as e: self.logger.warning(f"Failed to create custom SummarizerConfig from instructions: {e}", exc_info=True)
        try:
            return await self.summarizer.summarize(text_content, config=custom_conf)
        except (SummarizationError, ExternalServiceError) as e:
            self.logger.error(f"Summarization task failed: {e}", exc_info=True); raise
        except Exception as e: # Catch any other unexpected error from summarizer
            raise SummarizationError(f"An unexpected error occurred during summarization: {e}") from e


    async def _call_llm_extract_entities(self, text_content: str, instructions: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        # This method remains largely the same
        if not self.ner_pipeline:
            self.logger.warning("NER pipeline unavailable. Skipping entity extraction.")
            return []
        try:
            # Assuming ner_pipeline call is synchronous, run in executor if called from async
            loop = asyncio.get_running_loop()
            raw_results = await loop.run_in_executor(None, self.ner_pipeline, text_content)
            # Process raw_results into the desired format
            processed_entities = [{"text": e.get('word'), "type": e.get('entity_group', e.get('label', 'UNKNOWN')), "score": e.get('score'), **e} for e in raw_results if e.get('word')]
            return processed_entities
        except Exception as e:
            self.logger.error(f"NER pipeline execution failed: {e}", exc_info=True)
            raise ExternalServiceError(f"NER pipeline failed during execution: {e}") from e


    async def _call_llm_extract_relations(self, text_content: str, entities: Optional[List[Dict[str, Any]]], instructions: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        # This method remains largely the same, ensuring async execution for pipeline calls
        found_relations = []
        loop = asyncio.get_running_loop()
        try:
            if self.re_pipeline:
                # Assuming re_pipeline is sync, run in executor
                outputs = await loop.run_in_executor(None, self.re_pipeline, text_content)
                if isinstance(outputs, list) and outputs: # Ensure outputs is a list and not empty
                    for item in outputs: # Iterate through potentially multiple generated texts
                        # Parse item['generated_text'] if needed, or assume item is already a dict
                        if isinstance(item, dict) and all(k in item for k in ['subject','relation','object']): # Basic check
                            found_relations.append(item)
                        # Add more sophisticated parsing if needed based on pipeline output
            elif self.re_llm_pipeline:
                if not self.re_llm_pipeline.tokenizer:
                    raise ConfigurationError("RE LLM tokenizer is missing, cannot extract relations.")
                prompt = f"Given the text: \"{text_content}\"\nAnd entities: {entities}\nExtract relations as (Subject; Predicate; Object)."
                # Assuming re_llm_pipeline is sync, run in executor
                outputs = await loop.run_in_executor(None, self.re_llm_pipeline, prompt, max_length=len(self.re_llm_pipeline.tokenizer.encode(prompt)) + 150) # type: ignore
                if outputs and outputs[0]['generated_text']:
                    generated_text_after_prompt = outputs[0]['generated_text'][len(prompt):].strip()
                    for line in generated_text_after_prompt.split('\n'):
                        match = re.match(r'\(\s*(.*?)\s*;\s*(.*?)\s*;\s*(.*?)\s*\)', line.strip())
                        if match:
                            found_relations.append({"subject": match.group(1), "verb": match.group(2), "object": match.group(3)})
        except Exception as e:
            self.logger.error(f"Relation extraction pipeline execution failed: {e}", exc_info=True)
            raise ExternalServiceError(f"Relation extraction pipeline failed: {e}") from e
        return found_relations

    async def _generate_insights(self, chunk_text: str, chunk_label: Optional[Dict[str, Any]], instructions: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        # This method remains largely the same; raw_data_id is not handled here.
        instructions = instructions or {}
        summary_text, entities_data, relations_data, embedding_data = None, None, None, None

        is_question = chunk_label.get("intent") == "question" if chunk_label else False

        if instructions.get("summarize", True) and not is_question:
            summary_text = await self._call_llm_summarize(chunk_text, instructions.get('summarization_params'))

        if instructions.get("extract_entities", True):
            entities_data = await self._call_llm_extract_entities(chunk_text, instructions.get('entity_params'))

        if instructions.get("extract_relations", True) and entities_data: # Relations often depend on entities
            relations_data = await self._call_llm_extract_relations(chunk_text, entities_data, instructions.get('relation_params'))

        if chunk_text.strip() and self.embedding_model:
            try:
                embedding_data = await self.embedding_model.generate_embedding(chunk_text)
            except EmbeddingError as e:
                self.logger.error(f"Content embedding generation for chunk failed: {e}", exc_info=True) # Non-critical for this chunk

        self.logger.debug(f"Generated insights for chunk: Label='{chunk_label.get('intent') if chunk_label else 'N/A'}', Summary='{bool(summary_text)}', Entities={len(entities_data or [])}, Relations={len(relations_data or [])}")
        return {
            "summary_text": summary_text,
            "entities_data": entities_data,
            "relations_data": relations_data,
            "original_chunk_text": chunk_text,
            "chunk_label": chunk_label,
            "content_embedding_data": embedding_data
        }

    async def _structure_data(self, insights_for_chunk: Dict[str, Any]) -> StructuredInsight:
        # This method remains largely the same; raw_data_id is not set here.
        self.logger.debug("Packaging chunk insights into StructuredInsight object.")
        return StructuredInsight(
            original_data_type="text_chunk",
            source_data_preview=(insights_for_chunk.get("original_chunk_text", "")[:250] + "...") if insights_for_chunk.get("original_chunk_text") else None,
            summary=Summary(text=insights_for_chunk["summary_text"]) if insights_for_chunk.get("summary_text") else None,
            entities=[Entity(**e) for e in insights_for_chunk.get("entities_data", []) if isinstance(e, dict)],
            relations=[Relation(**r) for r in insights_for_chunk.get("relations_data", []) if isinstance(r, dict)],
            content_embedding=insights_for_chunk.get("content_embedding_data"),
            chunk_label=insights_for_chunk.get("chunk_label")
            # raw_data_id will be set by MemoryManager or the specific flow (e.g. ingest_data)
        )

    # _write_to_memory method is REMOVED as per plan. Logic moves to MemoryManager.store().

    async def process_data(self, raw_data_content: str, context_instructions: Optional[Dict[str, Any]]=None) -> List[StructuredInsight]:
        """
        Processes raw data content:
        1. Chunks the data using the configured SemanticChunker.
        2. For each chunk, labels it using the chunker.
        3. For each labeled chunk, generates insights (summary, entities, relations, embedding).
        4. Structures the insights into a StructuredInsight object.
        Returns a list of StructuredInsight objects, one for each processed chunk.
        """
        self.logger.info(f"Processing raw data content. Length: {len(raw_data_content)}. Instructions: {context_instructions}")
        all_structured_insights: List[StructuredInsight] = []

        if not raw_data_content or not raw_data_content.strip():
            self.logger.warning("Empty raw_data_content provided to process_data.")
            return []

        try:
            # 1. Chunk the data using SemanticChunker.split_text
            # Assuming default chunking method if not specified, or it's configured in SemanticChunker.
            text_chunks = self.chunker.split_text(raw_data_content) # Max_tokens etc. are part of chunker's config or defaults
            if not text_chunks:
                self.logger.info("No chunks produced from raw_data_content.")
                return []
            self.logger.info(f"Split content into {len(text_chunks)} chunks.")

            for i, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    self.logger.debug(f"Skipping empty chunk {i+1}.")
                    continue
                self.logger.debug(f"Processing chunk {i+1}/{len(text_chunks)}: '{chunk_text[:70]}...'")

                # 2. Label each chunk using SemanticChunker.label_chunk
                chunk_label = await self.chunker.label_chunk(chunk_text)
                self.logger.debug(f"Chunk {i+1} labeled: {chunk_label}")

                # 3. Generate insights for the current chunk
                insights = await self._generate_insights(chunk_text, chunk_label, context_instructions)

                # 4. Structure the insights for the current chunk
                structured_insight_for_chunk = await self._structure_data(insights)
                all_structured_insights.append(structured_insight_for_chunk)

            self.logger.info(f"Data processing complete. Generated {len(all_structured_insights)} structured insights.")
            return all_structured_insights

        except (ConfigurationError, SummarizationError, EmbeddingError, ExternalServiceError, PipelineError, CoreLogicError) as e:
            self.logger.error(f"Core logic error in ContextAgent.process_data: {type(e).__name__} - {e}", exc_info=True)
            raise # Re-raise specific, known errors
        except Exception as e:
            self.logger.error(f"Unexpected error in ContextAgent.process_data: {e}", exc_info=True)
            # Wrap unexpected errors in a generic CoreLogicError
            raise CoreLogicError(f"An unexpected error occurred in ContextAgent.process_data: {e}") from e

    async def ingest_data(self, data: IngestData, settings: AppSettings) -> IngestResponse:
        """
        Handles synchronous data ingestion.
        Processes data, validates insights, and stores them using MemoryManager.
        This method is called directly by the /ingest API endpoint.
        """
        self.logger.info(f"Starting ingestion for document_id='{data.document_id}' from source_uri='{data.source_uri}'")

        if not data.content and not data.source_uri: # Should be caught by API, but double check
            raise ValueError("Either 'content' or 'source_uri' must be provided for ingestion.")

        # For now, assume data.content is populated. Handling data.source_uri to fetch content
        # would be an additional step (e.g., using a helper to read from URL/file).
        # Let's proceed with data.content for this implementation.
        raw_content = data.content
        if not raw_content:
            # Placeholder: If only source_uri is provided, fetch content.
            # This would require a utility function. For now, error if no direct content.
            # In a real scenario: raw_content = await fetch_content_from_uri(data.source_uri)
            self.logger.error(f"Ingestion error: No direct content provided and source_uri fetching not implemented. Doc ID: {data.document_id}")
            return IngestResponse(document_id=data.document_id or "unknown", status="failed", message="No content to process.")

        processed_insights_count = 0
        stored_insights_count = 0
        # Use document_id from IngestData if provided, otherwise generate one or let MemoryManager handle it.
        # For now, we'll pass it down via StructuredInsight.raw_data_id.
        base_document_id = data.document_id or f"ingested_{datetime.datetime.utcnow().timestamp()}"

        try:
            # 1. Process data to get structured insights
            # Pass context_instructions if any are derivable from IngestData.metadata or settings
            context_instructions = data.metadata.get("processing_instructions") if data.metadata else None
            structured_insights = await self.process_data(raw_content, context_instructions=context_instructions)
            processed_insights_count = len(structured_insights)
            self.logger.info(f"Ingestion: Processed content into {processed_insights_count} insights for doc: {base_document_id}")

            if not structured_insights:
                return IngestResponse(document_id=base_document_id, status="success", message="Content processed, but no insights generated.")

            # 2. Validate and Store each insight
            for insight in structured_insights:
                # Assign the base_document_id to each insight if it's meant to group them.
                # This assumes raw_data_id in StructuredInsight can serve as this grouping key.
                insight.raw_data_id = f"{base_document_id}_insight_{processed_insights_count - len(structured_insights) + 1}" # Example unique ID

                # Determine content for hallucination check (e.g., summary or preview)
                content_to_validate = insight.summary.text if insight.summary else insight.source_data_preview
                if not content_to_validate:
                    self.logger.warning(f"Skipping validation for insight (no content): {insight.raw_data_id}")
                    continue

                validation_result = await self.hallucination_detector.detect(content_to_validate)
                self.logger.info(f"Ingestion: Validation for insight {insight.raw_data_id}: {validation_result.is_valid}")

                if validation_result.is_valid:
                    await self.memory_manager.store(insight)
                    stored_insights_count += 1
                    self.logger.info(f"Ingestion: Stored insight {insight.raw_data_id} for doc: {base_document_id}")
                else:
                    self.logger.warning(f"Ingestion: Hallucination detected for insight {insight.raw_data_id}. Details: {validation_result.explanation}. Not storing.")
                    # Optionally, log this failed validation event to memory

            return IngestResponse(
                document_id=base_document_id,
                status="success",
                message=f"Ingestion processed. {processed_insights_count} insights generated, {stored_insights_count} stored."
            )

        except (ConfigurationError, MemoryAccessError, CoreLogicError) as e:
            self.logger.error(f"Ingestion failed for doc {base_document_id} due to core error: {e}", exc_info=True)
            return IngestResponse(document_id=base_document_id, status="failed", message=f"Core error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during ingestion for doc {base_document_id}: {e}", exc_info=True)
            return IngestResponse(document_id=base_document_id, status="failed", message=f"Unexpected error: {e}")
