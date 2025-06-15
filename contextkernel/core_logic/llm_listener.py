# LLM Listener Module (llm_listener.py) - Context Optimizer & Memory Writer

# 1. Purpose of the file/module:
# This module acts as the "Context Optimizer" and "Memory Writer" for the
# ContextKernel. Its primary responsibilities are:
#   - Processing new information: This information might come from user interactions,
#     external data feeds, or be identified by the Context Agent as a "context gap."
#   - Summarization & Insight Extraction: Utilizing Language Models (LLMs) to condense,
#     summarize, or extract key insights, entities, and relationships from the
#     processed information.
#   - Context Refinement: Enriching the information by linking it to existing knowledge,
#     disambiguating entities, or adding relevant metadata.
#   - Structuring Data: Transforming the refined insights into a structured format
#     suitable for storage in the kernel's memory systems.
#   - Writing to Memory: Interacting with the various memory components (STM, LTM,
#     GraphDB, RawCache) to save these structured insights, effectively building and
#     updating the kernel's knowledge base.

# 2. Core Logic:
# The LLM Listener's workflow typically involves:
#   - Receiving Data: Input data can be raw text, documents, conversation snippets,
#     or pointers to data that needs processing. This might be triggered by the
#     Context Agent or an external event.
#   - Pre-processing (Optional): Cleaning or preparing the data for LLM interaction.
#   - LLM Interaction for Insight Generation:
#     - Summarization: Using an LLM to create concise summaries (extractive or abstractive).
#     - Entity/Keyword Extraction: Identifying key people, places, topics, concepts.
#     - Relation Extraction: Discovering relationships between identified entities.
#     - Question Answering: Potentially asking questions about the text to extract specific facts.
#   - Data Structuring: Organizing the extracted insights. This might involve:
#     - Defining schemas or models (e.g., using Pydantic) for different types of information.
#     - Formatting data for graph databases (nodes, edges, properties).
#     - Preparing data for vector embedding (if not already embedded).
#   - Memory System Interaction:
#     - Deciding which memory tier(s) are appropriate for the structured insights
#       (e.g., raw data to RawCache, embeddings and text to LTM, entities and
#       relationships to GraphDB, frequently accessed summaries to STM).
#     - Calling the respective memory module interfaces to save the data.
#   - (Optional) Feedback Loop: Potentially providing feedback to the Context Agent
#     or other components about the outcome of the processing.

# 3. Key Inputs/Outputs:
#   - Inputs:
#     - Raw or Semi-processed Data: Text, documents, user conversation logs, API responses.
#     - Instructions/Context from Context Agent: Guidance on what to process, what kind
#       of insights to look for, or how to prioritize.
#     - Configuration: Parameters for LLM interaction (e.g., summarization length,
#       model choice), data structuring rules.
#   - Outputs:
#     - Structured Data written to Memory: No direct return value to the caller typically,
#       but side effects in the form of data written to STM, LTM, GraphDB, RawCache.
#     - Updates to GraphDB: New nodes and relationships representing the learned knowledge.
#     - Log Messages: Information about the processing and storage operations.

# 4. Dependencies/Needs:
#   - LLM Models: Access to LLMs, particularly those proficient in summarization
#     (e.g., BART, T5, Pegasus), question answering, and general text understanding.
#   - Memory System Interfaces: Python APIs to interact with `GraphDB.py`, `LTM.py`,
#     `STM.py`, and `RawCache.py`.
#   - Configuration: Settings for LLM API keys, model names, summarization parameters
#     (length, style, abstractive/extractive), data storage policies.
#   - Data Processing Libraries: For text manipulation, NER, keyword extraction if
#     not solely relying on a single LLM call.

# 5. Real-world solutions/enhancements:

#   Summarization Libraries/Models:
#   - Hugging Face Transformers:
#     - Abstractive Models: `facebook/bart-large-cnn`, `google/pegasus-xsum`, `t5-base`.
#     - Extractive Summarization: Can be done by selecting important sentences using
#       sentence embeddings and clustering, or using models fine-tuned for extractive tasks.
#     (https://huggingface.co/models?pipeline_tag=summarization)
#   - Gensim: `summarize` function (TextRank algorithm for extractive summarization).
#     (https://radimrehurek.com/gensim/summarization/summariser.html)
#   - Sumy: Library with various extractive summarization methods (LexRank, Luhn, LSA).
#     (https://pypi.org/project/sumy/)

#   Context Refinement Techniques:
#   - Named Entity Recognition (NER):
#     - spaCy: `doc.ents` provides quick and efficient NER. (https://spacy.io/)
#     - Hugging Face Transformers: Use pre-trained NER models for higher accuracy or more entity types.
#   - Keyword/Keyphrase Extraction:
#     - YAKE!: Unsupervised keyword extraction. (https://pypi.org/project/yake/)
#     - KeyBERT: Uses BERT embeddings to find keywords similar to the document. (https://pypi.org/project/keybert/)
#     - TF-IDF based methods (e.g., from `scikit-learn`).
#   - Relation Extraction:
#     - Can be complex; may involve training custom models, using OpenNRE-style libraries,
#       or carefully crafting prompts for LLMs to output structured relationship data.
#   - Data Disambiguation: Linking extracted entities to canonical entries in a knowledge base.

#   Knowledge Base Integration:
#   - RDF Triples: Format extracted information (subject-predicate-object) for storage
#     in triple stores (e.g., Neo4j with RDF support, Apache Jena, GraphDB).
#     Example: `(Entity: "Paris", Property: "isCapitalOf", Value: "France")`.
#   - Ontology Updating: If a formal ontology exists, new instances or relationships
#     can be added. Tools like `rdflib` can be used to manipulate RDF data.
#   - Populating Graph Databases: Directly create nodes and edges in property graphs
#     like Neo4j using their Python drivers.

#   Data Structuring:
#   - Pydantic: Define Python classes that represent the schema for different types
#     of insights. This ensures data consistency before writing to memory.
#     (https://docs.pydantic.dev/)
#   - Standard Dataclasses: Python's built-in dataclasses can also be used for simpler structures.

#   Conditional Processing Logic:
#   - Develop rules or a model to decide:
#     - Which memory tier is most appropriate for a piece of information (e.g., raw logs
#       to RawCache, processed summaries to LTM, entities/relations to GraphDB).
#     - Whether information is novel enough to be stored or if it's redundant.
#     - The level of summarization or detail required based on source or type of info.
#   - Example: A brief user chat might only update STM and LTM, while ingesting a large
#     technical document might populate RawCache, LTM (chunked embeddings), and GraphDB
#     (extracted entities and concepts).

import logging
import asyncio
import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, BaseSettings

try:
    from transformers import pipeline, Pipeline
except ImportError:
    pipeline = None # Placeholder if transformers is not installed
    Pipeline = None # type: ignore # Placeholder for type hinting


# Configuration Model for LLMListener
class LLMListenerConfig(BaseSettings):
    summarization_model_name: str = "t5-small"
    entity_extraction_model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    enable_stub_relation_extraction: bool = False
    default_summarization_min_length: int = 30
    default_summarization_max_length: int = 150

    class Config:
        env_prefix = 'LLM_LISTENER_' # e.g., LLM_LISTENER_SUMMARIZATION_MODEL_NAME


# Placeholder Memory System Interfaces (would typically be imported)
class BaseMemorySystem: # Base class for type hinting
    def __init__(self): # Moved __init__ to be before other methods as per common convention
        self.logger = logging.getLogger(self.__class__.__name__)

    async def store(self, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(f"BaseMemorySystem.store called with args: {args}, kwargs: {kwargs}")
        await asyncio.sleep(0) # Simulate async
        pass
    async def save_summary(self, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(f"BaseMemorySystem.save_summary called with args: {args}, kwargs: {kwargs}")
        await asyncio.sleep(0)
        pass
    async def save_document(self, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(f"BaseMemorySystem.save_document called with args: {args}, kwargs: {kwargs}")
        await asyncio.sleep(0)
        pass
    async def add_entities(self, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(f"BaseMemorySystem.add_entities called with args: {args}, kwargs: {kwargs}")
        await asyncio.sleep(0)
        pass
    async def add_relations(self, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(f"BaseMemorySystem.add_relations called with args: {args}, kwargs: {kwargs}")
        await asyncio.sleep(0)
        pass

    # Add a logger for debugging purposes within the base class itself if needed,
    # or rely on the logger from the class using these interfaces.
    # For simplicity here, we'll assume concrete implementations would have their own logging.
class STMInterface(BaseMemorySystem): pass
class LTMInterface(BaseMemorySystem): pass
class GraphDBInterface(BaseMemorySystem): pass
class RawCacheInterface(BaseMemorySystem): pass

# Pydantic Models for Structured Data
class TimestampedModel(BaseModel):
    # Note: Pydantic v2 uses model_fields_set for default_factory context,
    # but for v1 style, this is standard.
    # For Pydantic v1, default_factory should not take arguments.
    # For Pydantic v2, you might use `default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)`
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    def model_post_init(self, __context: Any) -> None:
        # A common pattern to update 'updated_at' on modification,
        # though not automatically done by Pydantic on every change.
        # This hook is more for initial post-validation setup.
        # Actual update of 'updated_at' would need to be handled by application logic
        # when an instance is modified.
        pass

class Entity(TimestampedModel):
    text: str
    type: str
    # Example of using HttpUrl if an entity had a canonical link
    # url: Optional[HttpUrl] = None
    metadata: Optional[Dict[str, Any]] = None

class Relation(TimestampedModel):
    subject: str # Could be an ID linking to an Entity model instance
    verb: str
    object: str  # Could be an ID linking to an Entity model instance
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Summary(TimestampedModel):
    text: str
    source_data_hash: Optional[str] = None # e.g., a hash of the input text used for summarization

class StructuredInsight(TimestampedModel):
    original_data_type: Optional[str] = None
    source_data_preview: Optional[str] = None # A small preview or hash of the raw data
    summary: Optional[Summary] = None
    entities: Optional[List[Entity]] = None
    relations: Optional[List[Relation]] = None
    raw_data_id: Optional[str] = None # ID if raw data was stored in RawCache (e.g., from RawCacheInterface.store)


class LLMListener:
    def __init__(self,
                 listener_config: LLMListenerConfig, # Updated parameter name and type
                 memory_systems: Dict[str, BaseMemorySystem],
                 data_processing_config: Optional[Dict[str, Any]] = None,
                 llm_client: Optional[Any] = None):
        self.logger = logging.getLogger(__name__)
        self.listener_config = listener_config # Store the new config object
        self.memory_systems = memory_systems
        self.data_processing_config = data_processing_config if data_processing_config is not None else {}
        self.llm_client = llm_client # External client, kept for now
        self.summarization_pipeline: Optional[Pipeline] = None
        self.ner_pipeline: Optional[Pipeline] = None

        if pipeline is None:
            self.logger.warning(
                "transformers library not installed. Summarization and NER features will be largely unavailable. "
                "Please install with: pip install transformers torch sentencepiece"
            )
        else:
            # Initialize Summarization Pipeline
            self.logger.info(f"Attempting to initialize summarization pipeline with model: {self.listener_config.summarization_model_name}")
            try:
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model=self.listener_config.summarization_model_name,
                    tokenizer=self.listener_config.summarization_model_name,
                )
                self.logger.info(f"Summarization pipeline initialized successfully with model: {self.listener_config.summarization_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize summarization pipeline with model {self.listener_config.summarization_model_name}: {e}", exc_info=True)
                self.summarization_pipeline = None

            # Initialize NER Pipeline
            self.logger.info(f"Attempting to initialize NER pipeline with model: {self.listener_config.entity_extraction_model_name}")
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.listener_config.entity_extraction_model_name,
                    tokenizer=self.listener_config.entity_extraction_model_name,
                )
                self.logger.info(f"NER pipeline initialized successfully with model: {self.listener_config.entity_extraction_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize NER pipeline with model {self.listener_config.entity_extraction_model_name}: {e}", exc_info=True)
                self.ner_pipeline = None

        self.logger.info(
            f"LLMListener initialized. "
            f"Summarization pipeline ready: {self.summarization_pipeline is not None}. "
            f"NER pipeline ready: {self.ner_pipeline is not None}. "
            f"Listener Config: {self.listener_config.model_dump_json(indent=2)}, " # Log new config
            f"Memory Systems: {list(self.memory_systems.keys())}, "
            f"Data Processing Config: {self.data_processing_config}, "
            f"External LLM Client provided: {self.llm_client is not None}"
        )

    async def _preprocess_data(self, raw_data: Any) -> Any:
        """Placeholder for data preprocessing logic."""
        self.logger.debug(f"Preprocessing data: {raw_data}")
        # In a real implementation, this would involve cleaning, transforming, etc.
        await asyncio.sleep(0) # Simulate async operation
        return raw_data # Passthrough for now

    async def _call_llm_summarize(self, text_content: Any, instructions: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Stub for calling LLM for summarization."""
        self.logger.info(f"Attempting to summarize content with instructions: {instructions}")
        self.logger.debug(f"Text content for summarization (first 100 chars): {str(text_content)[:100]}")

        if not self.summarization_pipeline:
            self.logger.warning("Summarization pipeline not available. Returning None.")
            if self.llm_client and hasattr(self.llm_client, 'summarize'):
                self.logger.info("Attempting summarization with external llm_client as fallback.")
                try:
                    # Use configured defaults if not in instructions for external client too
                    default_max_length_ext = self.listener_config.default_summarization_max_length
                    max_length_ext = instructions.get('max_length', default_max_length_ext) if instructions else default_max_length_ext

                    summary_text = await self.llm_client.summarize(str(text_content), max_length=max_length_ext)
                    self.logger.info("Summarization via external llm_client successful.")
                    return summary_text
                except Exception as e:
                    self.logger.error(f"Error during summarization with external llm_client: {e}", exc_info=True)
                    return None
            self.logger.warning("No summarization method available (pipeline or client).")
            return None

        try:
            # Default summarization parameters from listener_config
            min_length_default = self.listener_config.default_summarization_min_length
            max_length_default = self.listener_config.default_summarization_max_length

            min_length = instructions.get('min_length', min_length_default) if instructions else min_length_default
            max_length = instructions.get('max_length', max_length_default) if instructions else max_length_default

            # Log that the call is synchronous
            self.logger.info(
                "Calling Hugging Face summarization pipeline (synchronous call within async method). "
                f"Params: min_length={min_length}, max_length={max_length}"
            )

            # The pipeline expects a list of texts. We are summarizing one chunk at a time.
            # For very long texts that exceed model limits, chunking might be needed before this call.
            # However, this method processes a `text_chunk` which is assumed to be manageable.
            summary_results = self.summarization_pipeline(
                str(text_content),
                min_length=min_length,
                max_length=max_length,
                truncation=True # Ensure text is truncated if too long for the model
            )

            if summary_results and isinstance(summary_results, list) and summary_results[0].get('summary_text'):
                summary_text = summary_results[0]['summary_text']
                self.logger.info("Hugging Face summarization successful.")
                self.logger.debug(f"Generated summary: {summary_text}")
                return summary_text
            else:
                self.logger.error(f"Summarization did not return expected output format. Result: {summary_results}")
                return None
        except Exception as e:
            self.logger.error(f"Error during Hugging Face summarization pipeline execution: {e}", exc_info=True)
            return None

    async def _call_llm_extract_entities(self, text_content: Any, instructions: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Extracts entities using Hugging Face NER pipeline."""
        self.logger.info(f"Attempting to extract entities. Instructions: {instructions}")
        self.logger.debug(f"Text content for NER (first 100 chars): {str(text_content)[:100]}")

        if not self.ner_pipeline:
            self.logger.warning("NER pipeline not available. Returning empty list.")
            # Optionally, could add a fallback to self.llm_client if it has an NER method.
            return []

        try:
            self.logger.info("Calling Hugging Face NER pipeline (synchronous call within async method).")
            ner_results = self.ner_pipeline(str(text_content))

            extracted_entities = []
            if ner_results and isinstance(ner_results, list):
                for entity in ner_results:
                    # Default NER pipeline output: {'entity_group': 'PER', 'score': 0.99, 'word': 'Wolfgang', 'start': 0, 'end': 8}
                    # We need to transform this.
                    entity_text = entity.get('word')
                    entity_type = entity.get('entity_group', 'UNKNOWN') # Some models use 'entity_group', others 'label'

                    # Create a small context snippet
                    start_offset = entity.get('start', 0)
                    end_offset = entity.get('end', 0)
                    context_window = 30 # Characters before and after
                    context_start = max(0, start_offset - context_window)
                    context_end = min(len(str(text_content)), end_offset + context_window)
                    context_snippet = f"...{str(text_content)[context_start:start_offset]}[{entity_text}]{str(text_content)[end_offset:context_end]}..."

                    extracted_entities.append({
                        "text": entity_text,
                        "type": entity_type,
                        "context": context_snippet,
                        "score": entity.get('score'),
                        "start_char": start_offset,
                        "end_char": end_offset
                    })
                self.logger.info(f"Hugging Face NER successful. Extracted {len(extracted_entities)} entities.")
                self.logger.debug(f"Extracted entities: {extracted_entities}")
                return extracted_entities
            else:
                self.logger.warning(f"NER pipeline did not return expected list format. Result: {ner_results}")
                return []
        except Exception as e:
            self.logger.error(f"Error during Hugging Face NER pipeline execution: {e}", exc_info=True)
            return [] # Return empty list on error, or None if preferred

    async def _call_llm_extract_relations(self, text_content: Any, entities: Optional[List[Dict[str, Any]]], instructions: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Placeholder for relation extraction. Currently returns a stubbed example or empty list."""
        self.logger.info(f"Attempting to extract relations (placeholder implementation). Entities: {entities}, Instructions: {instructions}")

        # This is a placeholder. True LLM-based relation extraction is complex and would require
        # a dedicated pipeline or sophisticated prompting with a text-generation/QA model.
        # For now, we return a stub or an empty list, based on config.
        if self.listener_config.enable_stub_relation_extraction and entities and len(entities) >= 2:
            # Only return a stub if explicitly enabled and enough entities exist
            self.logger.debug(f"Using stubbed relation extraction (enabled by config). Text content (first 100 chars): {str(text_content)[:100]}")
            example_relation = {
                "subject": entities[0]['text'],
                "verb": "is_related_to_stub",
                "object": entities[1]['text'],
                "context": f"Stub context from: {str(text_content)[:50]}..."
            }
            self.logger.info(f"Returning stubbed relation: {example_relation}")
            return [example_relation]

        self.logger.info("Relation extraction placeholder: No relations extracted or stub disabled.")
        return []

    async def _generate_insights(self, data: Any, context_instructions: Optional[Dict[str, Any]], raw_data_doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates insights from data using LLM calls.
        Orchestrates calls for summarization, entity extraction, and relation extraction.
        Includes raw_data_doc_id in the returned insights if provided.
        """
        self.logger.info(f"Starting insight generation for data. Instructions: {context_instructions}")
        self.logger.debug(f"Data for insight generation: {data}")

        summary: Optional[str] = None
        entities: Optional[List[Dict[str, Any]]] = None
        relations: Optional[List[Dict[str, Any]]] = None

        # Default to True if not specified, or if context_instructions is None
        should_summarize = context_instructions.get("summarize", True) if context_instructions else True
        should_extract_entities = context_instructions.get("extract_entities", True) if context_instructions else True
        should_extract_relations = context_instructions.get("extract_relations", True) if context_instructions else True


        if should_summarize:
            self.logger.info("Attempting summarization...")
            summarization_params = context_instructions.get('summarization_params') if context_instructions else None
            summary = await self._call_llm_summarize(data, instructions=summarization_params)
            self.logger.info(f"Summarization result (stubbed): {summary}")
        else:
            self.logger.info("Skipping summarization based on context_instructions.")

        if should_extract_entities:
            self.logger.info("Attempting entity extraction...")
            entity_params = context_instructions.get('entity_params') if context_instructions else None
            entities = await self._call_llm_extract_entities(data, instructions=entity_params)
            self.logger.info(f"Entity extraction result (stubbed): {entities}")
        else:
            self.logger.info("Skipping entity extraction based on context_instructions.")

        if should_extract_relations:
            if entities: # Relation extraction might depend on entities
                self.logger.info("Attempting relation extraction...")
                relation_params = context_instructions.get('relation_params') if context_instructions else None
                relations = await self._call_llm_extract_relations(data, entities, instructions=relation_params)
                self.logger.info(f"Relation extraction result (stubbed): {relations}")
            else:
                self.logger.info("Skipping relation extraction because no entities were extracted or extraction was skipped.")
        else:
            self.logger.info("Skipping relation extraction based on context_instructions.")

        self.logger.info("Insight generation process completed.")
        # Include raw_data_doc_id in the returned dictionary
        return {"summary": summary, "entities": entities, "relations": relations, "original_data": data, "raw_data_doc_id": raw_data_doc_id}

    async def _structure_data(self, insights: Dict[str, Any]) -> StructuredInsight:
        """
        Structures the generated insights into Pydantic models.
        The 'insights' dictionary is expected to contain 'raw_data_doc_id' if available.
        """
        self.logger.info("Starting data structuring with Pydantic models...")
        self.logger.debug(f"Received insights for structuring: {insights}")

        summary_text = insights.get("summary")
        entity_list = insights.get("entities") # Expected to be List[Dict]
        relation_list = insights.get("relations") # Expected to be List[Dict]
        original_data = insights.get("original_data")
        # Get raw_data_id from insights, which should have been passed down if generated
        raw_data_id_val = insights.get("raw_data_doc_id")

        summary_obj: Optional[Summary] = None
        if summary_text:
            summary_obj = Summary(text=summary_text, source_data_hash=None) # Hash could be added

        entity_objects: Optional[List[Entity]] = None
        if entity_list:
            try:
                entity_objects = [Entity(**e_data) for e_data in entity_list]
            except Exception as e: # Catch Pydantic validation errors or other issues
                self.logger.error(f"Error creating Entity objects: {e}", exc_info=True)
                # Decide how to handle partial failure: skip entities, or raise error
                entity_objects = [] # Or None, depending on desired error handling

        relation_objects: Optional[List[Relation]] = None
        if relation_list:
            try:
                relation_objects = [Relation(**r_data) for r_data in relation_list]
            except Exception as e:
                self.logger.error(f"Error creating Relation objects: {e}", exc_info=True)
                relation_objects = []


        source_preview = str(original_data)[:100] + "..." if original_data else None

        structured_insight = StructuredInsight(
            original_data_type=type(original_data).__name__ if original_data else None,
            source_data_preview=source_preview,
            summary=summary_obj,
            entities=entity_objects,
            relations=relation_objects,
            raw_data_id=raw_data_id_val # This now comes from the insights dict
        )

        # No async work typically needed here, Pydantic model creation is synchronous.
        # await asyncio.sleep(0)

        self.logger.info("Data structuring with Pydantic models completed.")
        self.logger.debug(f"StructuredInsight object created: {structured_data.model_dump_json(indent=2)}") # Pydantic v2 preferred

        return structured_insight

    async def _write_to_memory(self, structured_data: StructuredInsight) -> None:
        """
        Writes StructuredInsight data to appropriate memory systems.
        Assumes structured_data.raw_data_id is populated if raw data was cached.
        Uses conditional logic based on available data and memory system clients.
        """
        self.logger.info("Starting memory writing operations with StructuredInsight...")
        self.logger.debug(f"StructuredInsight for memory: {structured_data.model_dump_json(indent=2)}")

        # Use created_at from the StructuredInsight model as a consistent ID base.
        # structured_data.raw_data_id can also serve as a base or part of it if available.
        doc_id_base = structured_data.raw_data_id if structured_data.raw_data_id else structured_data.created_at.isoformat()

        # Retrieve memory system clients
        stm: Optional[STMInterface] = self.memory_systems.get("stm") # type: ignore
        ltm: Optional[LTMInterface] = self.memory_systems.get("ltm") # type: ignore
        graph_db: Optional[GraphDBInterface] = self.memory_systems.get("graph_db") # type: ignore

        # Write summary to STM
        if structured_data.summary and stm:
            self.logger.info(f"Attempting to write summary to STM (ID base: {doc_id_base})...")
            try:
                summary_id = f"{doc_id_base}_summary"
                await stm.save_summary(summary_id=summary_id,
                                       summary_obj=structured_data.summary, # Pass the Summary Pydantic model
                                       metadata={"doc_id_base": doc_id_base, "raw_data_id": structured_data.raw_data_id})
                self.logger.info(f"Successfully wrote summary to STM with id: {summary_id}")
            except Exception as e:
                self.logger.error(f"Failed to write summary to STM: {e}", exc_info=True)
        elif structured_data.summary and not stm:
            self.logger.warning("STM client not available, skipping summary storage.")

        # Write insights (the whole StructuredInsight object) to LTM
        if ltm: # LTM might always be written to, or conditionally like others
            self.logger.info(f"Attempting to write document to LTM (ID base: {doc_id_base})...")
            try:
                ltm_doc_id = f"{doc_id_base}_ltm_doc"
                # Pass the full StructuredInsight Pydantic model
                await ltm.save_document(doc_id=ltm_doc_id,
                                        document_content=structured_data,
                                        metadata={"doc_id_base": doc_id_base, "raw_data_id": structured_data.raw_data_id})
                self.logger.info(f"Successfully wrote document to LTM with id: {ltm_doc_id}")
            except Exception as e:
                self.logger.error(f"Failed to write document to LTM: {e}", exc_info=True)
        else:
            self.logger.warning("LTM client not available, skipping document storage.")

        # Write entities and relations to GraphDB
        if graph_db:
            if structured_data.entities:
                self.logger.info(f"Attempting to write {len(structured_data.entities)} entities to GraphDB (Doc ID: {doc_id_base})...")
                try:
                    # Pass list of Entity Pydantic models
                    await graph_db.add_entities(entities=structured_data.entities,
                                                document_id=doc_id_base, # Link entities to the source document/ID
                                                metadata={"raw_data_id": structured_data.raw_data_id})
                    self.logger.info(f"Successfully wrote entities to GraphDB for document_id: {doc_id_base}")
                except Exception as e:
                    self.logger.error(f"Failed to write entities to GraphDB: {e}", exc_info=True)

            if structured_data.relations:
                self.logger.info(f"Attempting to write {len(structured_data.relations)} relations to GraphDB (Doc ID: {doc_id_base})...")
                try:
                    # Pass list of Relation Pydantic models
                    await graph_db.add_relations(relations=structured_data.relations,
                                                 document_id=doc_id_base, # Link relations to the source document/ID
                                                 metadata={"raw_data_id": structured_data.raw_data_id})
                    self.logger.info(f"Successfully wrote relations to GraphDB for document_id: {doc_id_base}")
                except Exception as e:
                    self.logger.error(f"Failed to write relations to GraphDB: {e}", exc_info=True)
        elif structured_data.entities or structured_data.relations: # Only warn if there was something to write
            self.logger.warning("GraphDB client not available, skipping graph data storage.")

        self.logger.info("Memory writing operations completed.")


    async def process_data(self, raw_data: Any, context_instructions: Optional[Dict[str, Any]] = None):
        """
        Processes raw data by preprocessing, generating insights using LLMs,
        structuring the insights, and writing them to memory systems.
        """
        self.logger.info(f"Received data for processing. Instructions: {context_instructions}")
        self.logger.debug(f"Raw data (type: {type(raw_data)}): {str(raw_data)[:200]}...") # Log snippet

        raw_data_doc_id: Optional[str] = None
        try:
            preprocessed_data = await self._preprocess_data(raw_data)
            self.logger.debug(f"Preprocessed data (type: {type(preprocessed_data)}): {str(preprocessed_data)[:200]}...")

            # Store raw_data in RawCache if available
            raw_cache_client: Optional[RawCacheInterface] = self.memory_systems.get("raw_cache") # type: ignore
            if raw_cache_client:
                try:
                    # Generate a unique ID for the raw data document
                    # Using timestamp and a part of the data hash could be an option for uniqueness
                    # For simplicity, using a timestamp-based ID here.
                    raw_doc_id_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    # A more robust ID might involve hashing preprocessed_data if it's consistently hashable
                    # raw_data_hash = hashlib.sha256(str(preprocessed_data).encode()).hexdigest()[:16]
                    # temp_raw_id = f"raw_{raw_doc_id_timestamp}_{raw_data_hash}"
                    temp_raw_id = f"raw_{raw_doc_id_timestamp}" # Simplified ID

                    self.logger.info(f"Attempting to store preprocessed data in RawCache with tentative ID: {temp_raw_id}")
                    # The store method is expected to return the actual ID used (could be same or different)
                    raw_data_doc_id = await raw_cache_client.store(doc_id=temp_raw_id, data=preprocessed_data)
                    self.logger.info(f"Successfully stored data in RawCache. Document ID: {raw_data_doc_id}")
                except Exception as e:
                    self.logger.error(f"Failed to store data in RawCache: {e}", exc_info=True)
                    # Decide if processing should continue without raw_cache_id or halt.
                    # For now, we'll continue, and raw_data_doc_id will remain None.

            # Pass raw_data_doc_id to insight generation
            insights = await self._generate_insights(preprocessed_data, context_instructions, raw_data_doc_id=raw_data_doc_id)
            self.logger.debug(f"Generated insights: {insights}")

            # raw_data_doc_id is already in 'insights' if generated, _structure_data will pick it up.
            structured_data = await self._structure_data(insights)
            self.logger.debug(f"Structured data: {structured_data.model_dump_json(indent=2)}")

            # _write_to_memory no longer needs raw_data_source, it uses structured_data.raw_data_id
            await self._write_to_memory(structured_data)

            self.logger.info("Data processing completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during data processing pipeline: {e}", exc_info=True)
            # Potentially re-raise or handle specific exceptions as needed
