import logging
from typing import Dict, Any, List, Optional as TypingOptional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings # For Config model
import spacy
import datetime
from spacy.matcher import Matcher

from ..utils.state_manager import InMemoryStateManager, RedisStateManager, AbstractStateManager
from .exceptions import (
    ConfigurationError, IntentDetectionError, ExternalServiceError,
    CoreLogicError, MemoryAccessError, SummarizationError, EmbeddingError, PipelineError
)

try:
    from transformers import pipeline, Pipeline
except ImportError:
    pipeline = None
    Pipeline = None # type: ignore


# Configuration Model
class NLPConfig(BaseSettings):
    spacy_model_name: str = "en_core_web_sm"
    low_confidence_threshold: float = 0.6
    default_intent_confidence: float = 0.5
    high_confidence_threshold: float = 0.8
    intent_candidate_labels: List[str] = Field(default_factory=lambda: ["search information", "save information", "summarize text", "general question"])
    intent_classifier_model: str = "facebook/bart-large-mnli"
    use_spacy_matcher_first: bool = True
    # model_config = SettingsConfigDict(env_prefix='CONTEXT_AGENT_') # Consider if a new prefix is needed, e.g., NLP_
    model_config = SettingsConfigDict(env_prefix='NLP_')


# Pydantic Models
class IntentExtractionResult(BaseModel):
    intent: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: TypingOptional[float] = None
    original_input: TypingOptional[str] = None
    spacy_doc: Any = None
    matched_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    class Config: arbitrary_types_allowed = True

class RoutingDecision(BaseModel):
    target_module: str
    task_parameters: Dict[str, Any] = Field(default_factory=dict)
    original_intent: TypingOptional[IntentExtractionResult] = None

class TaskResult(BaseModel):
    status: str
    data: TypingOptional[Any] = None
    message: TypingOptional[str] = None
    error_details: TypingOptional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)

def initialize_matcher(nlp: spacy.Language) -> Matcher:
    matcher = Matcher(nlp.vocab)
    patterns = {
        "search_info": [[{"LOWER": "search"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "find"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "look"}, {"LOWER": "up"}, {"IS_ASCII": True, "OP": "+"}]],
        "save_info": [[{"LOWER": "save"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "remember"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "store"}, {"IS_ASCII": True, "OP": "+"}]],
        "summarization_intent": [[{"LOWER": "summarize"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "tl;dr"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "give"}, {"LOWER": "me"}, {"LOWER": "a"}, {"LOWER": "summary"}, {"LOWER": "of"}, {"IS_ASCII": True, "OP": "+"}]]
    }
    for intent, p_list in patterns.items(): matcher.add(intent, p_list)
    return matcher

def process_input(raw_input: any) -> str:
    try:
        return str(raw_input).lower().strip()
    except Exception as e:
        logger.error(f"Input processing error for '{raw_input}': {e}", exc_info=True); return ""

async def detect_intent(
    processed_input: str,
    nlp: spacy.Language,
    matcher: Matcher,
    intent_classifier: TypingOptional[Pipeline],
    config: NLPConfig
) -> IntentExtractionResult:
    intent, entities, confidence, spacy_doc, patterns = "unknown_intent", {}, config.default_intent_confidence, None, []
    used_spacy = False

    if not nlp:
        logger.error("spaCy NLP model not initialized. Intent detection severely limited.")
        raise ConfigurationError("spaCy model essential for detect_intent is not available.")

    try:
        spacy_doc = nlp(processed_input)
        if config.use_spacy_matcher_first and matcher:
            matches = matcher(spacy_doc)
            if matches:
                match_id, start, end = matches[0]
                intent, confidence, used_spacy = nlp.vocab.strings[match_id], config.high_confidence_threshold, True
                for m_id, s, e in matches: patterns.append({"pattern_name": nlp.vocab.strings[m_id], "matched_text": spacy_doc[s:e].text})
                # Basic entity from match
                entity_text = spacy_doc[keyword_end_token_index(spacy_doc, matches[0]):end].text.strip()
                if entity_text:
                    if intent == "search_info": entities["query"] = entity_text
                    elif intent == "save_info": entities["data"] = entity_text
                    elif intent == "summarization_intent": entities["text_to_summarize"] = entity_text

        if not used_spacy or confidence < config.high_confidence_threshold:
            if intent_classifier:
                try:
                    res = intent_classifier(processed_input, config.intent_candidate_labels, multi_label=False)
                    if res and res['labels']: intent, confidence = res['labels'][0], res['scores'][0]
                    else: logger.warning("Zero-shot classifier returned no labels."); intent, confidence = "general_question", config.default_intent_confidence
                except Exception as e_clf:
                    logger.error(f"Zero-shot classification error: {e_clf}", exc_info=True)
                    if not used_spacy: intent, confidence = "general_question", config.default_intent_confidence
                    raise ExternalServiceError(f"Zero-shot classifier failed: {e_clf}") from e_clf
            elif not used_spacy: intent, confidence = "general_question", config.default_intent_confidence

        # spaCy NER
        for ent in spacy_doc.ents:
            label = ent.label_.lower()
            entities.setdefault(label, []).append(ent.text)
        for k, v_list in entities.items(): # Consolidate list values
            if isinstance(v_list, list): entities[k] = ", ".join(v_list)

        return IntentExtractionResult(intent=intent, entities=entities, confidence=confidence, original_input=processed_input, spacy_doc=spacy_doc, matched_patterns=patterns)

    except AttributeError as ae:
        logger.error(f"AttributeError in detect_intent (spaCy/Matcher issue?): {ae}", exc_info=True)
        raise IntentDetectionError(f"Component (spaCy/Matcher) error: {ae}") from ae
    except Exception as e:
        logger.error(f"Unexpected error in detect_intent for '{processed_input}': {e}", exc_info=True)
        raise IntentDetectionError(f"Unexpected error in intent detection: {e}") from e

# Helper
def keyword_end_token_index(doc: spacy.tokens.Doc, match: tuple) -> int:
    start_token_idx = match[1]; first_token_text = doc[start_token_idx].lower_
    if first_token_text in ["search", "find", "save", "remember", "store", "summarize", "tl;dr"]: return start_token_idx + 1
    if first_token_text == "look" and doc[start_token_idx+1].lower_ == "up": return start_token_idx + 2
    if first_token_text == "give" and doc[start_token_idx+4].lower_ == "of": return start_token_idx + 5
    return match[2]
