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
from typing import Any, Dict, List, Optional, Union # Added Union for Pydantic Field default_factory
from pydantic import BaseModel, Field, HttpUrl # HttpUrl might be useful later

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
                 llm_config: Dict[str, Any],
                 memory_systems: Dict[str, BaseMemorySystem],
                 data_processing_config: Optional[Dict[str, Any]] = None,
                 llm_client: Optional[Any] = None): # Added llm_client
        self.logger = logging.getLogger(__name__)
        self.llm_config = llm_config
        self.memory_systems = memory_systems
        self.data_processing_config = data_processing_config if data_processing_config is not None else {}
        self.llm_client = llm_client # Store the llm_client

        self.logger.info(f"LLMListener initialized with llm_config: {self.llm_config}, "
                         f"memory_systems: {list(self.memory_systems.keys())}, " # Log keys to avoid large object logging
                         f"data_processing_config: {self.data_processing_config}, "
                         f"llm_client provided: {self.llm_client is not None}")

    async def _preprocess_data(self, raw_data: Any) -> Any:
        """Placeholder for data preprocessing logic."""
        self.logger.debug(f"Preprocessing data: {raw_data}")
        # In a real implementation, this would involve cleaning, transforming, etc.
        await asyncio.sleep(0) # Simulate async operation
        return raw_data # Passthrough for now

    async def _call_llm_summarize(self, text_content: Any, instructions: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Stub for calling LLM for summarization."""
        self.logger.info(f"Attempting to summarize content with instructions: {instructions}")
        self.logger.debug(f"LLM Summarize - Text content: {text_content}, Instructions: {instructions}, LLM Config: {self.llm_config.get('summarization_model')}")

        if self.llm_client and hasattr(self.llm_client, 'summarize'):
            try:
                # Assuming instructions might contain 'max_length' or similar parameters
                max_length = instructions.get('max_length', 100) if instructions else 100
                summary_text = await self.llm_client.summarize(str(text_content), max_length=max_length)
                self.logger.info(f"Summarization via llm_client successful.")
                return summary_text
            except Exception as e:
                self.logger.error(f"Error during summarization with llm_client: {e}", exc_info=True)
                # Fallback to old stub or handle error appropriately

        # Fallback to old stub if no llm_client or if client call fails
        await asyncio.sleep(0) # Simulate async LLM call
        self.logger.info("Using fallback stub summarization in _call_llm_summarize.")
        return f"This is a stubbed summary of: {str(text_content)[:50]}..."

    async def _call_llm_extract_entities(self, text_content: Any, instructions: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Stub for calling LLM for entity extraction."""
        self.logger.info(f"Attempting to extract entities from content with instructions: {instructions}")
        self.logger.debug(f"LLM Extract Entities - Text content: {text_content}, Instructions: {instructions}, LLM Config: {self.llm_config.get('entity_extraction_model')}")
        await asyncio.sleep(0) # Simulate async LLM call
        return [
            {"text": "StubEntity1", "type": "PERSON", "context": str(text_content)[:30]},
            {"text": "StubEntity2", "type": "LOCATION", "context": str(text_content)[:30]}
        ]

    async def _call_llm_extract_relations(self, text_content: Any, entities: Optional[List[Dict[str, Any]]], instructions: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Stub for calling LLM for relation extraction."""
        self.logger.info(f"Attempting to extract relations from content with entities: {entities} and instructions: {instructions}")
        self.logger.debug(f"LLM Extract Relations - Text content: {text_content}, Entities: {entities}, Instructions: {instructions}, LLM Config: {self.llm_config.get('relation_extraction_model')}")
        await asyncio.sleep(0) # Simulate async LLM call
        if entities and len(entities) >= 2:
            return [{
                "subject": entities[0]['text'],
                "verb": "is_related_to_stub",
                "object": entities[1]['text'],
                "context": str(text_content)[:30]
            }]
        return [{"subject": "UnknownSubject", "verb": "has_stub_relation", "object": "UnknownObject"}]

    async def _generate_insights(self, data: Any, context_instructions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates insights from data using LLM calls (stubs).
        Orchestrates calls for summarization, entity extraction, and relation extraction.
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
        return {"summary": summary, "entities": entities, "relations": relations, "original_data": data}

    async def _structure_data(self, insights: Dict[str, Any], raw_data_id_val: Optional[str] = None) -> StructuredInsight:
        """
        Structures the generated insights into Pydantic models.
        """
        self.logger.info("Starting data structuring with Pydantic models...")
        self.logger.debug(f"Received insights for structuring: {insights}")

        summary_text = insights.get("summary")
        entity_list = insights.get("entities") # Expected to be List[Dict]
        relation_list = insights.get("relations") # Expected to be List[Dict]
        original_data = insights.get("original_data")

        summary_obj: Optional[Summary] = None
        if summary_text:
            # In a real scenario, source_data_hash might be generated from original_data
            summary_obj = Summary(text=summary_text, source_data_hash=None)

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
            raw_data_id=raw_data_id_val # Pass the ID from raw cache storage if available
        )

        await asyncio.sleep(0) # Simulate async work if any complex transformation were done

        self.logger.info("Data structuring with Pydantic models completed.")
        self.logger.debug(f"StructuredInsight object created: {structured_insight.model_dump_json(indent=2)}") # Pydantic v2

        return structured_insight

    async def _write_to_memory(self, structured_data: StructuredInsight, raw_data_doc_id: Optional[str] = None) -> None:
        """
        Writes StructuredInsight data to appropriate memory systems (stubs).
        Uses conditional logic based on available data and memory system clients.
        """
        self.logger.info("Starting memory writing operations with StructuredInsight...")
        self.logger.debug(f"StructuredInsight for memory: {structured_data.model_dump_json(indent=2)}")

        # Use created_at from the StructuredInsight model as a consistent ID if available
        # otherwise, fallback to a new timestamp. This assumes raw_data_doc_id is for the raw data itself.
        doc_id_base = structured_data.created_at.isoformat()

        # Retrieve memory system clients
        stm: Optional[STMInterface] = self.memory_systems.get("stm") # type: ignore
        ltm: Optional[LTMInterface] = self.memory_systems.get("ltm") # type: ignore
        graph_db: Optional[GraphDBInterface] = self.memory_systems.get("graph_db") # type: ignore
        # raw_cache client is used before this method now, to get raw_data_doc_id
        # So, it's not directly used here for storing raw_data_source anymore.
        # raw_data_source is also not passed here anymore.

        try:
            # Raw data is assumed to be stored before this method, and raw_data_doc_id is passed.
            # The StructuredInsight model now contains raw_data_id.

            # Write summary to STM
            if structured_data.summary and stm is not None:
                self.logger.info("Attempting to write summary to STM...")
                try:
                    # Pass Pydantic model directly or specific fields
                    await stm.save_summary(summary_id=doc_id_base + "_summary",
                                           summary_obj=structured_data.summary, # Pass the Summary object
                                           summary_text=structured_data.summary.text, # Or just the text
                                           metadata={"doc_id_base": doc_id_base})
                    self.logger.info(f"Successfully wrote summary to STM with id: {doc_id_base}_summary")
                except Exception as e:
                    self.logger.error(f"Failed to write summary to STM: {e}", exc_info=True)
            elif processed_summary and stm is None:
                self.logger.warning("STM client not available, skipping summary storage.")

            # Write insights (summary, entities) to LTM
            if (structured_data.summary or structured_data.entities) and ltm is not None:
                self.logger.info("Attempting to write insights to LTM...")
                try:
                    # LTM might store the whole StructuredInsight model (converted to dict if necessary by client)
                    # or specific parts.
                    await ltm.save_document(doc_id=doc_id_base + "_ltm_doc",
                                            document_content=structured_data, # Pass the StructuredInsight object
                                            metadata={"doc_id_base": doc_id_base})
                    self.logger.info(f"Successfully wrote document to LTM with id: {doc_id_base}_ltm_doc")
                except Exception as e:
                    self.logger.error(f"Failed to write document to LTM: {e}", exc_info=True)
            elif (processed_summary or structured_data.get("extracted_entities")) and ltm is None:
                 self.logger.warning("LTM client not available, skipping document storage.")


            # Write entities and relations to GraphDB
            if (structured_data.entities or structured_data.relations) and graph_db is not None:
                self.logger.info("Attempting to write to GraphDB...")
                try:
                    if structured_data.entities:
                        # Pass list of Entity Pydantic models
                        await graph_db.add_entities(entities=structured_data.entities,
                                                    document_id=doc_id_base,
                                                    metadata={"doc_id_base": doc_id_base})
                        self.logger.info(f"Successfully wrote {len(structured_data.entities)} entities to GraphDB for doc_id: {doc_id_base}")
                    if structured_data.relations:
                        # Pass list of Relation Pydantic models
                        await graph_db.add_relations(relations=structured_data.relations,
                                                     document_id=doc_id_base,
                                                     metadata={"doc_id_base": doc_id_base})
                        self.logger.info(f"Successfully wrote {len(structured_data.relations)} relations to GraphDB for doc_id: {doc_id_base}")
                except Exception as e:
                    self.logger.error(f"Failed to write to GraphDB: {e}", exc_info=True)
            elif (entities or relations) and graph_db is None:
                self.logger.warning("GraphDB client not available, skipping graph data storage.")

            self.logger.info("Memory writing operations completed.")

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during memory writing operations: {e}", exc_info=True)


    async def process_data(self, raw_data: Any, context_instructions: Optional[Dict[str, Any]] = None):
        """
        Processes raw data by preprocessing, generating insights using LLMs,
        structuring the insights, and writing them to memory systems.
        """
        self.logger.info(f"Received data for processing. Instructions: {context_instructions}")
        self.logger.debug(f"Raw data: {raw_data}") # Be cautious logging raw_data if it can be very large

        try:
            preprocessed_data = await self._preprocess_data(raw_data)
            self.logger.debug(f"Preprocessed data: {preprocessed_data}")

            insights = await self._generate_insights(preprocessed_data, context_instructions)
            self.logger.debug(f"Generated insights: {insights}")

            structured_data = await self._structure_data(insights)
            self.logger.debug(f"Structured data: {structured_data}")

            await self._write_to_memory(structured_data, raw_data_source=raw_data)

            self.logger.info("Data processing completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during data processing: {e}", exc_info=True)
            # Potentially re-raise or handle specific exceptions as needed
