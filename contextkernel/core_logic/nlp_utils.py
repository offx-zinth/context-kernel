import logging
from typing import List
from pydantic import Field # BaseModel removed as classes using it are removed
from pydantic_settings import BaseSettings, SettingsConfigDict # SettingsConfigDict needs to be imported for Pydantic V2+
import spacy
from spacy.matcher import Matcher

# Exceptions specific to old ContextAgent/detect_intent are removed if not generally applicable.
# ConfigurationError might still be useful if NLPConfig loading can fail.
# ExternalServiceError might be relevant if initialize_matcher or future NLP utils call external services.
# For now, keeping them to avoid breaking imports if other modules use them, will prune later if unused.
from .exceptions import (
    ConfigurationError,
    # IntentDetectionError, # Specific to removed detect_intent
    ExternalServiceError,
    # CoreLogicError # General, might be useful elsewhere
)

# Transformers pipeline import is removed as detect_intent (which used it) is removed.
# If SemanticChunker needs it directly, it will handle its own imports.


# Configuration Model - Kept as it's used by SemanticChunker setup in main.py
class NLPConfig(BaseSettings):
    spacy_model_name: str = "en_core_web_sm"
    # Fields related to old detect_intent's confidence logic might be removable if
    # SemanticChunker has its own confidence handling or these are not used by its setup.
    # For now, keeping them as they are part of NLPConfig used in main.py.
    low_confidence_threshold: float = 0.6 # Potentially unused by new chunker directly from here
    default_intent_confidence: float = 0.5 # Used by SemanticChunker
    high_confidence_threshold: float = 0.8 # Used by SemanticChunker
    intent_candidate_labels: List[str] = Field(default_factory=lambda: ["search information", "save information", "summarize text", "general question"]) # Used by SemanticChunker
    intent_classifier_model: str = "facebook/bart-large-mnli" # Potentially used by SemanticChunker if it sets up its own classifier
    use_spacy_matcher_first: bool = True # Used by SemanticChunker

    model_config = SettingsConfigDict(env_prefix='NLP_')


# Pydantic Models: IntentExtractionResult, RoutingDecision, TaskResult are removed
# as they were specific to the old ContextAgent's direct intent handling and routing.
# SemanticChunker.label_chunk returns a Dict, not IntentExtractionResult.

logger = logging.getLogger(__name__)

# initialize_matcher is kept as it's used in main.py for SemanticChunker setup.
def initialize_matcher(nlp: spacy.Language) -> Matcher:
    matcher = Matcher(nlp.vocab)
    # Define patterns for intents. These might be specific to the old ContextAgent's needs.
    # SemanticChunker might use a different set or configuration for its Matcher.
    # For now, keeping these patterns as they are part of the existing initialize_matcher.
    # If SemanticChunker evolves to need different patterns, this function might need adjustment
    # or SemanticChunker might manage its own pattern definition.
    patterns = {
        "search_info": [
            [{"LOWER": "search"}, {"IS_ASCII": True, "OP": "+"}],
            [{"LOWER": "find"}, {"IS_ASCII": True, "OP": "+"}],
            [{"LOWER": "look"}, {"LOWER": "up"}, {"IS_ASCII": True, "OP": "+"}]
        ],
        "save_info": [
            [{"LOWER": "save"}, {"IS_ASCII": True, "OP": "+"}],
            [{"LOWER": "remember"}, {"IS_ASCII": True, "OP": "+"}],
            [{"LOWER": "store"}, {"IS_ASCII": True, "OP": "+"}]
        ],
        "summarization_intent": [ # This intent might be less relevant if summarization is a standard part of insight generation
            [{"LOWER": "summarize"}, {"IS_ASCII": True, "OP": "+"}],
            [{"LOWER": "tl;dr"}, {"IS_ASCII": True, "OP": "+"}],
            [{"LOWER": "give"}, {"LOWER": "me"}, {"LOWER": "a"}, {"LOWER": "summary"}, {"LOWER": "of"}, {"IS_ASCII": True, "OP": "+"}]
        ]
        # Future: Add more patterns or make them configurable via NLPConfig
    }
    for intent, p_list in patterns.items():
        matcher.add(intent, p_list)
    logger.info(f"SpaCy Matcher initialized with {len(patterns)} intent patterns: {list(patterns.keys())}")
    return matcher

# process_input function is removed as its logic (lower, strip) is simple and
# can be handled directly by consumers like SemanticChunker.

# detect_intent async function is removed as its core logic is now part of SemanticChunker.label_chunk.

# keyword_end_token_index helper function is removed as it was specific to detect_intent.
