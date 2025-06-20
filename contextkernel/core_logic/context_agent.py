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
    # Assuming LLMRetriever and LLMListener are type names defined elsewhere or will be
    def __init__(self,
                 agent_config: ContextAgentConfig,
                 memory_system: Any,
                 retriever: Any, # Replace Any with actual LLMRetriever type if available
                 listener: Any,  # Replace Any with actual LLMListener type if available
                 state_manager: TypingOptional[AbstractStateManager] = None):
        self.logger = logging.getLogger(__name__)
        self.agent_config = agent_config 
        self.memory_system = memory_system
        self.retriever = retriever
        self.listener = listener
        self.intent_classifier: Optional[Pipeline] = None
        self.nlp: Optional[spacy.Language] = None
        self.matcher: Optional[Matcher] = None
        self.state_manager: Optional[AbstractStateManager] = state_manager

        self.intent_handlers = {
            "search_info": self._handle_search_info,
            "save_info": self._handle_save_info,
            "summarize_text": self._handle_summarize_text,
            "general_question": self._handle_general_question,
            # "intent_detection_error" and low confidence are handled separately
        }

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

    def _handle_search_info(self, intent_result: IntentExtractionResult) -> tuple[str, Dict[str, Any]]:
        return "LLMRetriever", intent_result.entities or {}

    def _handle_save_info(self, intent_result: IntentExtractionResult) -> tuple[str, Dict[str, Any]]:
        return "LLMListener", intent_result.entities or {}

    def _handle_summarize_text(self, intent_result: IntentExtractionResult) -> tuple[str, Dict[str, Any]]:
        entities = intent_result.entities or {}
        return "LLMListener", {"raw_data": entities.get("text_to_summarize", intent_result.original_input), "context_instructions": {"summarize": True}}

    def _handle_general_question(self, intent_result: IntentExtractionResult) -> tuple[str, Dict[str, Any]]:
        return "LLMRetriever", {"query": intent_result.original_input}

    def _handle_default_error(self, intent_result: IntentExtractionResult, error_message: str, details: TypingOptional[Any] = None) -> tuple[str, Dict[str, Any]]:
        params = {
            "original_intent_info": intent_result.model_dump(exclude_none=True, exclude={'spacy_doc'}),
            "error_message": error_message
        }
        if details:
            params["details"] = details
        return "ErrorHandler", params

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
        try:
            intent_str = intent_result.intent
            confidence = intent_result.confidence
            entities = intent_result.entities or {}

            if intent_str == "intent_detection_error":
                target, params = self._handle_default_error(intent_result, "Intent detection failed.", entities.get("error_message", "N/A"))
            elif confidence is None or confidence < self.agent_config.low_confidence_threshold:
                details = {"confidence_score": confidence, "detected_intent": intent_str}
                target, params = self._handle_default_error(intent_result, "Intent unclear (low confidence).", details)
            else:
                handler = self.intent_handlers.get(intent_str)
                if handler:
                    target, params = handler(intent_result)
                else:
                    target, params = self._handle_default_error(intent_result, f"Unknown or unhandled intent: {intent_str}")

            self.logger.info(f"Route: Target='{target}', Params='{params}'.")
            return RoutingDecision(target_module=target, task_parameters=params, original_intent=intent_result)
        except Exception as e: 
            self.logger.error(f"Critical error in decide_route for intent '{intent_result.intent}': {e}", exc_info=True)
            # Use _handle_default_error for consistency, though it might be overkill if intent_result is problematic
            target, params = self._handle_default_error(intent_result, "Critical routing error.", str(e))
            return RoutingDecision(target_module=target, task_parameters=params, original_intent=intent_result)

    async def dispatch_task(self, route: RoutingDecision) -> TaskResult:
        target, params = route.target_module, route.task_parameters
        self.logger.info(f"Dispatching to module='{target}'. Params: {params}")
        try:
            if target == "LLMRetriever":
                if not self.retriever: # Check if retriever component is available
                    raise ConfigurationError("LLMRetriever component not configured.")
                query = params.get('query')
                if query is None: raise CoreLogicError("Missing 'query' for LLMRetriever.")
                # Assuming self.retriever has a 'retrieve' method
                res = await self.retriever.retrieve(query=query, **{k:v for k,v in params.items() if k!='query'})
                return TaskResult(status="success", data=res, message="LLMRetriever processed.")
            elif target == "LLMListener":
                if not self.listener: # Check if listener component is available
                    raise ConfigurationError("LLMListener component not configured.")
                # Assuming self.listener has a 'process_data' method
                res = await self.listener.process_data(raw_data=params.get('raw_data'), context_instructions=params.get('context_instructions'))
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
    """
    Identifies the end token index of an argument to a keyword within a matched span.
    Uses dependency parsing to find direct objects (dobj), open clausal complements (xcomp),
    or clausal complements (ccomp) of keywords.
    """
    match_id, match_start_idx, match_end_idx = match

    # Keywords that typically take direct arguments we want to capture.
    # This list can be expanded.
    KEYWORDS = {"search", "find", "save", "remember", "store", "summarize", "get", "retrieve", "show", "tell", "explain"}
    # Multi-word keywords might need special handling if not caught as a single token by the matcher.
    # e.g., "look up", "figure out". The current spaCy Matcher patterns might handle some of these.
    # For "give me a summary of", "of" is a preposition, and its object (PPOBJ) would be the argument.

    keyword_token = None

    # Iterate through tokens in the match to find the keyword.
    # The assumption is that the Matcher pattern correctly identifies a span
    # containing the keyword and potentially its arguments.
    # We prioritize the *first* keyword found in the match span.
    for i in range(match_start_idx, match_end_idx):
        token = doc[i]
        if token.lower_ in KEYWORDS:
            keyword_token = token
            break
        # Handling for "look up" type phrases if "look" is the verb and "up" is prt (particle)
        if token.lower_ == "look" and i + 1 < match_end_idx and doc[i+1].lower_ == "up":
            keyword_token = token # "look" is the main verb
            break
        # Handling for "give me a summary of" - "summarize" is the implicit keyword
        # but "of" is a good anchor if "summarize" (or similar) is matched.
        # This part is tricky and might be better handled by specific matcher patterns
        # that directly target the PPOBJ of "of" if "summary of" is matched.
        # For now, let's assume the main keywords will be verbs.
        if token.lower_ == "give" and token.doc.text.lower().startswith("give me a summary of", token.idx):
             # find "of" and its object
            for t in doc[token.i : match_end_idx]:
                if t.lower_ == "of":
                    for child in t.children:
                        if child.dep_ == "pobj": # Prepositional object
                             # The argument is the subtree of this pobj
                            return child.subtree.last.i + 1
                    break # Found "of", processed it
            # Fallback if "of" or its pobj isn't found as expected after "give me a summary of"
            return match_end_idx


    if keyword_token is None:
        # No identifiable keyword from our list in the matched span.
        # This might mean the pattern matched something else, or the keyword is not in our list.
        # Fallback to the end of the match.
        return match_end_idx

    # Found a keyword_token. Now look for its arguments using dependency parsing.
    # Relevant dependency labels for arguments.
    ARGUMENT_DEPS = {"dobj", "xcomp", "ccomp"}

    argument_tokens = []

    for child in keyword_token.children:
        if child.dep_ in ARGUMENT_DEPS:
            # The argument is the entire subtree of this child.
            # We want to find the rightmost token of this subtree.
            argument_tokens.extend(list(child.subtree))
        elif child.dep_ == "prep": # Prepositional phrase modifying the verb
            # e.g., "search for news", "news" is pobj of "for"
            # We could also look for pobj of prepositions that are children of the keyword
            for p_child in child.children:
                if p_child.dep_ == "pobj":
                    argument_tokens.extend(list(p_child.subtree))

    if argument_tokens:
        # Find the rightmost token among all identified argument components.
        # The token indices are absolute within the doc.
        last_arg_token_idx = -1
        for token in argument_tokens:
            if token.i > last_arg_token_idx:
                last_arg_token_idx = token.i

        # Ensure the argument's end is not before the keyword itself
        # and not beyond the original match end, unless the subtree naturally extends.
        # However, the original purpose of keyword_end_token_index was to refine *within* the match.
        # For now, let's cap it at match_end_idx if it goes beyond, or reconsider if this capping is good.
        # The prompt asks for the index *after* the end of the argument.
        # If last_arg_token_idx is valid, return last_arg_token_idx + 1
        # This could potentially be beyond match_end_idx if a dobj's subtree extends far.
        # The original function used `spacy_doc[... : end].text.strip()`, implying `end` was exclusive.
        # If `last_arg_token_idx` is the actual last token, then `last_arg_token_idx + 1` is the correct slice end.

        # We need to ensure the returned index is at least after the keyword.
        # And typically, it should be within the matched span or slightly after if the argument extends.
        # The primary goal is to get the *argument* of the keyword.

        # If the rightmost token of the argument is keyword_token.i, it means no argument was found after it.
        if last_arg_token_idx > keyword_token.i :
            return last_arg_token_idx + 1
        else: # Argument didn't extend beyond keyword, or was empty
             return keyword_token.i + 1


    # Fallback: No direct object or relevant complement found for the keyword.
    # Return the index of the token immediately following the keyword.
    # This is a conservative fallback, assuming the argument is just the keyword itself (e.g. "search!").
    # Or, if the original match was longer, match_end_idx might be better.
    # The prompt suggested `match[1] + 1` (start_token_idx + 1) or `match[2]` (match_end_idx).
    # If keyword_token is found, `keyword_token.i + 1` is more precise than `match_start_idx + 1`.
    return keyword_token.i + 1

    # Commenting out the old logic:
    # start_token_idx = match[1]; first_token_text = doc[match_start_idx].lower_
    # if first_token_text in ["search", "find", "save", "remember", "store", "summarize", "tl;dr"]: return start_token_idx + 1
    # if first_token_text == "look" and doc[start_token_idx+1].lower_ == "up": return start_token_idx + 2
    # if first_token_text == "give" and doc[start_token_idx+4].lower_ == "of": return start_token_idx + 5
    # return match[2]
