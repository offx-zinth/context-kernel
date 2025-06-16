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
class ContextAgentConfig(BaseSettings):
    spacy_model_name: str = "en_core_web_sm"
    low_confidence_threshold: float = 0.6
    default_intent_confidence: float = 0.5
    high_confidence_threshold: float = 0.8
    intent_candidate_labels: List[str] = Field(default_factory=lambda: ["search information", "save information", "summarize text", "general question"])
    intent_classifier_model: str = "facebook/bart-large-mnli"
    use_spacy_matcher_first: bool = True
    state_manager_type: str = "in_memory"
    redis_host: str = "localhost"
    redis_port: int = 6379

    model_config = SettingsConfigDict(env_prefix='CONTEXT_AGENT_')

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


class ContextAgent:
    def __init__(self, llm_service: Any, memory_system: Any, agent_config: ContextAgentConfig, state_manager: TypingOptional[AbstractStateManager] = None):
        self.logger = logging.getLogger(__name__)
        self.agent_config = agent_config
        self.llm_service = llm_service
        self.memory_system = memory_system
        self.intent_classifier: Optional[Pipeline] = None
        self.nlp: Optional[spacy.Language] = None
        self.matcher: Optional[Matcher] = None
        self.state_manager: Optional[AbstractStateManager] = state_manager

        if not self.agent_config.spacy_model_name:
            self.logger.warning("No spaCy model name configured; spaCy features disabled.")
        else:
            try:
                self.nlp = spacy.load(self.agent_config.spacy_model_name)
                self.matcher = Matcher(self.nlp.vocab)
                self._initialize_matchers()
                self.logger.info(f"spaCy model '{self.agent_config.spacy_model_name}' initialized.")
            except OSError as e:
                self.logger.error(f"spaCy model '{self.agent_config.spacy_model_name}' not found or invalid: {e}", exc_info=True)
                if self.agent_config.use_spacy_matcher_first:
                    raise ConfigurationError(f"Required spaCy model '{self.agent_config.spacy_model_name}' failed to load.") from e
            except Exception as e:
                self.logger.error(f"Unexpected error loading spaCy model '{self.agent_config.spacy_model_name}': {e}", exc_info=True)

        if pipeline is None:
            self.logger.warning("Transformers 'pipeline' unavailable. Intent classifier (zero-shot) disabled.")
            if self.agent_config.intent_classifier_model:
                 raise ConfigurationError("Transformers 'pipeline' is unavailable, but intent_classifier_model is configured.")
        elif not self.agent_config.intent_classifier_model:
            self.logger.info("No intent_classifier_model configured; zero-shot intent classification skipped.")
        else:
            try:
                self.intent_classifier = pipeline("zero-shot-classification", model=self.agent_config.intent_classifier_model)
                self.logger.info(f"Intent classifier '{self.agent_config.intent_classifier_model}' loaded.")
            except Exception as e:
                self.logger.error(f"Failed to load intent classifier '{self.agent_config.intent_classifier_model}': {e}", exc_info=True)

        if self.state_manager is None:
            self.logger.info(f"Initializing StateManager (type: {self.agent_config.state_manager_type})")
            if self.agent_config.state_manager_type == "redis":
                try:
                    self.state_manager = RedisStateManager(host=self.agent_config.redis_host, port=self.agent_config.redis_port)
                except ImportError as ie:
                    self.logger.error(f"aioredis import failed for RedisStateManager: {ie}. Falling back to InMemoryStateManager.", exc_info=True)
                    self.state_manager = InMemoryStateManager()
                except Exception as e:
                    self.logger.error(f"RedisStateManager init failed (host: {self.agent_config.redis_host}): {e}. Falling back to InMemoryStateManager.", exc_info=True)
                    self.state_manager = InMemoryStateManager()
            elif self.agent_config.state_manager_type == "in_memory":
                self.state_manager = InMemoryStateManager()
            else:
                self.logger.warning(f"Unknown state_manager_type '{self.agent_config.state_manager_type}'. Defaulting to 'in_memory'.")
                self.state_manager = InMemoryStateManager()
        else:
             self.logger.info(f"Using pre-injected StateManager: {type(self.state_manager).__name__}.")
        self.logger.info(f"ContextAgent fully initialized. spaCy: {self.nlp is not None}. Intent Classifier: {self.intent_classifier is not None}. StateManager: {self.state_manager is not None}.")

    def _initialize_matchers(self):
        if not self.matcher: return
        patterns = {
            "search_info": [[{"LOWER": "search"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "find"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "look"}, {"LOWER": "up"}, {"IS_ASCII": True, "OP": "+"}]],
            "save_info": [[{"LOWER": "save"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "remember"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "store"}, {"IS_ASCII": True, "OP": "+"}]],
            "summarization_intent": [[{"LOWER": "summarize"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "tl;dr"}, {"IS_ASCII": True, "OP": "+"}], [{"LOWER": "give"}, {"LOWER": "me"}, {"LOWER": "a"}, {"LOWER": "summary"}, {"LOWER": "of"}, {"IS_ASCII": True, "OP": "+"}]]
        }
        for intent, p_list in patterns.items(): self.matcher.add(intent, p_list)

    def process_input(self, raw_input: any) -> str:
        try:
            return str(raw_input).lower().strip()
        except Exception as e:
            self.logger.error(f"Input processing error for '{raw_input}': {e}", exc_info=True); return ""

    async def detect_intent(self, processed_input: str) -> IntentExtractionResult:
        intent, entities, confidence, spacy_doc, patterns = "unknown_intent", {}, self.agent_config.default_intent_confidence, None, []
        used_spacy = False
        if not self.nlp:
            self.logger.error("spaCy NLP model not initialized. Intent detection severely limited.")
            raise ConfigurationError("spaCy model essential for detect_intent is not available.")
        try:
            spacy_doc = self.nlp(processed_input)
            if self.agent_config.use_spacy_matcher_first and self.matcher:
                matches = self.matcher(spacy_doc)
                if matches:
                    match_id, start, end = matches[0]
                    intent, confidence, used_spacy = self.nlp.vocab.strings[match_id], self.agent_config.high_confidence_threshold, True
                    for m_id, s, e in matches: patterns.append({"pattern_name": self.nlp.vocab.strings[m_id], "matched_text": spacy_doc[s:e].text})
                    # Basic entity from match
                    entity_text = spacy_doc[keyword_end_token_index(spacy_doc, matches[0]):end].text.strip()
                    if entity_text:
                        if intent == "search_info": entities["query"] = entity_text
                        elif intent == "save_info": entities["data"] = entity_text
                        elif intent == "summarization_intent": entities["text_to_summarize"] = entity_text

            if not used_spacy or confidence < self.agent_config.high_confidence_threshold:
                if self.intent_classifier:
                    try:
                        res = self.intent_classifier(processed_input, self.agent_config.intent_candidate_labels, multi_label=False)
                        if res and res['labels']: intent, confidence = res['labels'][0], res['scores'][0]
                        else: self.logger.warning("Zero-shot classifier returned no labels."); intent, confidence = "general_question", self.agent_config.default_intent_confidence
                    except Exception as e_clf:
                        self.logger.error(f"Zero-shot classification error: {e_clf}", exc_info=True)
                        if not used_spacy: intent, confidence = "general_question", self.agent_config.default_intent_confidence
                        raise ExternalServiceError(f"Zero-shot classifier failed: {e_clf}") from e_clf
                elif not used_spacy: intent, confidence = "general_question", self.agent_config.default_intent_confidence

            # spaCy NER
            for ent in spacy_doc.ents:
                label = ent.label_.lower()
                entities.setdefault(label, []).append(ent.text)
            for k, v_list in entities.items(): # Consolidate list values
                if isinstance(v_list, list): entities[k] = ", ".join(v_list)
            return IntentExtractionResult(intent=intent, entities=entities, confidence=confidence, original_input=processed_input, spacy_doc=spacy_doc, matched_patterns=patterns)
        except AttributeError as ae:
            self.logger.error(f"AttributeError in detect_intent (spaCy/Matcher issue?): {ae}", exc_info=True)
            raise IntentDetectionError(f"Component (spaCy/Matcher) error: {ae}") from ae
        except Exception as e:
            self.logger.error(f"Unexpected error in detect_intent for '{processed_input}': {e}", exc_info=True)
            raise IntentDetectionError(f"Unexpected error in intent detection: {e}") from e

    def decide_route(self, intent_result: IntentExtractionResult) -> RoutingDecision:
        target, params = "ErrorHandler", {"original_intent_info": intent_result.model_dump(exclude_none=True, exclude={'spacy_doc'})}
        try:
            intent, entities, conf = intent_result.intent, intent_result.entities or {}, intent_result.confidence
            if intent == "intent_detection_error": params["error_message"], params["details"] = "Intent detection failed.", entities.get("error_message", "N/A")
            elif conf is None or conf < self.agent_config.low_confidence_threshold:
                params.update({"error_message": "Intent unclear (low confidence).", "confidence_score": conf, "detected_intent": intent})
            elif intent == "search_info": target, params = "LLMRetriever", entities
            elif intent == "save_info": target, params = "LLMListener", entities
            elif intent == "summarize_text": target, params = "LLMListener", {"raw_data": entities.get("text_to_summarize", intent_result.original_input), "context_instructions": {"summarize": True}}
            elif intent == "general_question": target, params = "LLMRetriever", {"query": intent_result.original_input}
            else: params["error_message"] = f"Unknown or unhandled intent: {intent}"
            self.logger.info(f"Route: Target='{target}', Params='{params}'.")
            return RoutingDecision(target_module=target, task_parameters=params, original_intent=intent_result)
        except Exception as e:
            self.logger.error(f"Critical error in decide_route for intent '{intent_result.intent}': {e}", exc_info=True)
            return RoutingDecision(target_module="ErrorHandler", task_parameters={"error_message": "Critical routing error.", "details": str(e), "original_intent_info": intent_result.model_dump(exclude_none=True, exclude={'spacy_doc'})}, original_intent=intent_result)

    async def dispatch_task(self, route: RoutingDecision) -> TaskResult:
        target, params = route.target_module, route.task_parameters
        self.logger.info(f"Dispatching to module='{target}'. Params: {params}")
        try:
            if target == "LLMRetriever":
                if not (self.llm_service and hasattr(self.llm_service, 'retriever') and hasattr(self.llm_service.retriever, 'retrieve')):
                    raise ConfigurationError("LLMRetriever service/method not configured.")
                query = params.get('query')
                if query is None: raise CoreLogicError("Missing 'query' for LLMRetriever.")
                res = await self.llm_service.retriever.retrieve(query=query, **{k:v for k,v in params.items() if k!='query'})
                return TaskResult(status="success", data=res, message="LLMRetriever processed.")
            elif target == "LLMListener":
                if not (self.llm_service and hasattr(self.llm_service, 'listener') and hasattr(self.llm_service.listener, 'process_data')):
                    raise ConfigurationError("LLMListener service/method not configured.")
                res = await self.llm_service.listener.process_data(raw_data=params.get('raw_data'), context_instructions=params.get('context_instructions'))
                return TaskResult(status="success", data=res, message="LLMListener processed.")
            elif target == "ErrorHandler":
                return TaskResult(status="error", message=params.get("error_message", "Error handled."), error_details=params)
            else:
                self.logger.warning(f"Unknown module for dispatch: '{target}'.")
                raise CoreLogicError(f"Unknown module for dispatch: {target}")
        except (ConfigurationError, CoreLogicError) as e: self.logger.error(f"Dispatch error: {e}", exc_info=True); raise
        except (EmbeddingError, MemoryAccessError, SummarizationError, PipelineError, ExternalServiceError) as e_service:
            self.logger.error(f"Service error in '{target}': {e_service}", exc_info=True)
            raise ExternalServiceError(f"Error in {target}: {e_service}") from e_service
        except Exception as e:
            self.logger.error(f"Unexpected dispatch error to '{target}': {e}", exc_info=True)
            raise CoreLogicError(f"Unexpected dispatch error to {target}: {e}") from e

    async def handle_request(self, raw_input: any, conversation_id: TypingOptional[str] = None, current_context: TypingOptional[Dict[str,Any]] = None) -> TaskResult:
        self.logger.info(f"--- Handling request for Conversation ID: {conversation_id or 'N/A'} ---")
        try:
            current_context = current_context or {}
            if self.state_manager and conversation_id:
                try:
                    retrieved_state = await self.state_manager.get_state(conversation_id)
                    if retrieved_state: current_context = {**retrieved_state, **current_context}
                except MemoryAccessError as e: self.logger.error(f"State retrieval failed for {conversation_id}: {e}", exc_info=True) # Proceed without state

            processed_input = self.process_input(raw_input)
            if not processed_input and raw_input not in [None, ""]:
                 return TaskResult(status="error", message="Input processing failed.", error_details={"original_input": str(raw_input)[:200]})

            intent_res = await self.detect_intent(processed_input)
            route_decision = self.decide_route(intent_res)
            task_result = await self.dispatch_task(route_decision)

            if self.state_manager and conversation_id:
                try:
                    state_to_save = {"last_user_input": processed_input, "last_intent": intent_res.intent, "last_entities": intent_res.entities, "last_task_module": route_decision.target_module, "last_task_status": task_result.status, "timestamp": datetime.datetime.utcnow().isoformat()}
                    await self.state_manager.save_state(conversation_id, state_to_save)
                except MemoryAccessError as e: self.logger.error(f"State saving failed for {conversation_id}: {e}", exc_info=True) # Non-critical for current result

            self.logger.info(f"Request handled. Final status: {task_result.status}")
            return task_result

        except (IntentDetectionError, ConfigurationError, ExternalServiceError, CoreLogicError, MemoryAccessError) as e:
            self.logger.error(f"{type(e).__name__} in handle_request: {e}", exc_info=True)
            return TaskResult(status="error", message=f"{type(e).__name__}: {str(e)[:100]}", error_details={"type": type(e).__name__, "details": str(e)})
        except Exception as e:
            self.logger.error(f"Unexpected critical error in handle_request for {conversation_id or 'N/A'}: {e}", exc_info=True)
            return TaskResult(status="error", message=f"Unexpected critical error: {str(e)[:100]}", error_details={"type": type(e).__name__, "details": str(e)})

# Helper
def keyword_end_token_index(doc: spacy.tokens.Doc, match: tuple) -> int:
    start_token_idx = match[1]; first_token_text = doc[start_token_idx].lower_
    if first_token_text in ["search", "find", "save", "remember", "store", "summarize", "tl;dr"]: return start_token_idx + 1
    if first_token_text == "look" and doc[start_token_idx+1].lower_ == "up": return start_token_idx + 2
    if first_token_text == "give" and doc[start_token_idx+4].lower_ == "of": return start_token_idx + 5
    return match[2]
