# Context Agent Module (context_agent.py) - The "Prefrontal Cortex"

# 1. Purpose of the file/module:
# This module represents the "prefrontal cortex" or central executive of the
# ContextKernel AI system. It is responsible for high-level cognitive tasks,
# primarily:
#   - Understanding semantic intent from incoming user queries, system events,
#     or other forms of raw input.
#   - Making decisions about how to handle the input.
#   - Routing tasks to the appropriate specialized modules within the kernel
#     (e.g., LLM-Retriever for fetching information, LLM-Listener for processing
#     and storing new information, or other custom tools/plugins).
#   - Orchestrating the flow of information and control between different components.
#   - Potentially managing conversation state and context over multiple turns.

# 2. Core Logic:
# The core logic of the Context Agent typically involves several stages:
#   - Input Processing: Receiving and pre-processing input (e.g., text cleaning,
#     basic parsing).
#   - Intent Detection: Analyzing the input to determine the underlying goal or
#     purpose. This can range from simple keyword matching to sophisticated
#     NLP-based classification using machine learning models.
#   - Parameter Extraction: Identifying key entities or parameters within the input
#     that are necessary for fulfilling the intent (e.g., dates, names, locations,
#     specific search terms).
#   - Routing Decision: Based on the detected intent and extracted parameters,
#     the agent decides which module, tool, or workflow is best suited to handle
#     the request. This can be driven by a rules engine, a trained model, or a
#     hybrid approach.
#   - Task Dispatching: Formulating a task or instruction for the selected module
#     and dispatching it. This might involve transforming the input into a format
#     expected by the target module.
#   - Response Handling (Optional): It might receive a response from the dispatched
#     module and decide on further actions, or directly formulate a response to
#     the original input source.
#   - Orchestration: Managing the sequence of operations if multiple modules need
#     to be invoked to fulfill a complex request.

# 3. Key Inputs/Outputs:
#   - Inputs:
#     - User Queries: Natural language text from users (e.g., chat messages, search queries).
#     - System Events: Notifications or data from other parts of the system or
#       external services.
#     - Structured Data: Potentially, inputs could also be more structured commands or API calls.
#     - Current Context: Information about the ongoing interaction or state.
#   - Outputs:
#     - Dispatched Tasks: Instructions or calls to other modules within the ContextKernel
#       (e.g., a query for the LLMRetriever, data for the LLMListener to process).
#     - Direct Responses: In some cases, the Context Agent might generate a direct
#       response to the user (e.g., an acknowledgement, a clarification question).
#     - Updated Context/State: Modifications to the conversational or system state.

# 4. Dependencies/Needs:
#   - LLM Models: Access to language models for tasks like intent detection,
#     entity extraction, query understanding, or even for making routing decisions.
#   - Interfaces to Core Logic Modules: Mechanisms to call and pass data to
#     `LLMRetriever`, `LLMListener`, memory systems, and any custom tools/plugins.
#   - Configuration:
#     - Routing Rules/Models: Definitions for how intents map to modules.
#     - LLM model configurations (e.g., API keys, model names).
#   - Memory System: Access to STM/LTM to retrieve context that might inform
#     intent detection or routing.

# 5. Real-world solutions/enhancements:

#   Intent Detection Libraries/Techniques:
#   - spaCy: For rule-based matching (e.g., using `Matcher`, `PhraseMatcher`),
#     dependency parsing, NER, and text classification for simpler intent models.
#     (https://spacy.io/)
#   - NLTK (Natural Language Toolkit): Provides tools for text processing, classification,
#     and other NLP tasks. (https://www.nltk.org/)
#   - Transformer-based Models (Hugging Face):
#     - Zero-shot/Few-shot Classification: Use models like BART or T5 fine-tuned on NLI
#       (Natural Language Inference) to classify intent with few or no examples.
#     - Fine-tuned Models: Fine-tune models like BERT, RoBERTa, DistilBERT on a specific
#       intent recognition dataset for higher accuracy.
#     (https://huggingface.co/transformers/)
#   - Rasa NLU: Open-source framework specifically for intent recognition and entity
#     extraction in conversational AI. (https://rasa.com/docs/rasa/nlu-training-data/)
#   - Scikit-learn: For traditional ML approaches to intent classification (e.g., SVM,
#     Logistic Regression on TF-IDF features).

#   Routing Strategies:
#   - Rule-based Routing: Define explicit `IF intent THEN module` rules. Can be implemented
#     with simple conditional logic or dedicated rules engines.
#     - Example: `IF intent == "get_weather" AND "location" in entities THEN route_to_weather_tool`.
#   - Model-based Routing: Train a classification model where the input is the query/context
#     and the output classes are the target modules or tools.
#   - Hybrid Approach: Use rules for common, well-defined intents and a model for more
#     ambiguous cases or for selecting among multiple potentially relevant tools.
#   - Cost-based Routing: If multiple modules can handle a request, route based on factors
#     like predicted latency, computational cost, or reliability.

#   Orchestration Analogies & Tools:
#   - API Gateways (e.g., Kong, Apigee, AWS API Gateway): Manage and route incoming API
#     requests to backend services, similar to how the agent routes tasks.
#   - Workflow Orchestrators (e.g., Apache Airflow, Prefect, Camunda, AWS Step Functions):
#     Define, schedule, and monitor complex workflows involving multiple steps/tasks.
#     The Context Agent could be seen as a dynamic, AI-driven orchestrator.
#   - Business Process Management (BPM) Engines: Software for modeling, executing, and
#     monitoring business processes, which often involve routing and decision logic.

#   State Management:
#   - If handling multi-turn conversations, the Context Agent needs to manage conversation
#     state (e.g., previous turns, user context, active goals).
#   - This state can be stored in an STM (e.g., Redis) or passed around.
#   - State information is crucial for resolving anaphora, understanding follow-up
#     questions, and maintaining coherent interactions.

#   Learning/Adaptive Routing:
#   - Reinforcement Learning: The agent could learn optimal routing strategies over time
#     based on feedback (e.g., task success, user satisfaction).
#   - Feedback Loops: Incorporate feedback on the quality or relevance of module outputs
#     to refine routing rules or retrain routing models.

import logging
from typing import Dict, Any, List, Optional as TypingOptional # Renamed to avoid conflict with Pydantic's Optional
from pydantic import BaseModel, Field, BaseSettings
import spacy
from spacy.matcher import Matcher

# Configuration Model
class ContextAgentConfig(BaseSettings):
    spacy_model_name: str = "en_core_web_sm"
    low_confidence_threshold: float = 0.6
    default_intent_confidence: float = 0.5  # For fallback mechanisms
    high_confidence_threshold: float = 0.8 # For spaCy rule matches

    class Config:
        env_prefix = 'CONTEXT_AGENT_' # Example: CONTEXT_AGENT_SPACY_MODEL_NAME

# Pydantic Models for Data Structuring
class IntentExtractionResult(BaseModel):
    intent: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: TypingOptional[float] = None
    original_input: TypingOptional[str] = None # For error cases or context
    spacy_doc: Any = None # To store the spaCy Doc object, not serialized by default
    matched_patterns: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True # Allow Any type for spacy_doc

class RoutingDecision(BaseModel):
    target_module: str
    task_parameters: Dict[str, Any] = Field(default_factory=dict)
    original_intent: TypingOptional[IntentExtractionResult] = None # For context/error

class TaskResult(BaseModel):
    status: str  # e.g., "success", "error", "partial_success"
    data: TypingOptional[Any] = None
    message: TypingOptional[str] = None
    error_details: TypingOptional[Dict[str, Any]] = None # For structured error info

class ContextAgent:
    def __init__(self, llm_service, memory_system, agent_config: ContextAgentConfig, state_manager=None):
        """
        Initializes the ContextAgent.

        Args:
            llm_service: Stub for the LLM service.
            memory_system: Stub for the memory system.
            agent_config: Configuration object for the ContextAgent.
            state_manager: Optional. Instance for managing conversation state.
        """
        self.llm_service = llm_service
        self.memory_system = memory_system
        self.agent_config = agent_config # Store the config object
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        try:
            self.logger.info(f"Loading spaCy model: {self.agent_config.spacy_model_name}")
            self.nlp = spacy.load(self.agent_config.spacy_model_name)
            self.matcher = Matcher(self.nlp.vocab)
            self._initialize_matchers()
            self.logger.info(f"ContextAgent initialized with spaCy model '{self.agent_config.spacy_model_name}' and matchers.")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model '{self.agent_config.spacy_model_name}' or initializing matchers: {e}", exc_info=True)
            self.nlp = None
            self.matcher = None

        if self.state_manager:
            self.logger.info("ContextAgent initialized with StateManager.")
        else:
            self.logger.info("ContextAgent initialized (no StateManager).") # Kept as INFO as it's a significant config detail

    def _initialize_matchers(self):
        """Initializes spaCy Matcher rules."""
        if not self.matcher:
            return

        # Pattern for "search_info"
        pattern_search_1 = [{"LOWER": "search"}, {"IS_ASCII": True, "OP": "+"}] # "search <something>"
        pattern_search_2 = [{"LOWER": "find"}, {"IS_ASCII": True, "OP": "+"}] # "find <something>"
        pattern_search_3 = [{"LOWER": "look"}, {"LOWER": "up"}, {"IS_ASCII": True, "OP": "+"}] # "look up <something>"
        self.matcher.add("search_info", [pattern_search_1, pattern_search_2, pattern_search_3])

        # Pattern for "save_info"
        pattern_save_1 = [{"LOWER": "save"}, {"IS_ASCII": True, "OP": "+"}] # "save <something>"
        pattern_save_2 = [{"LOWER": "remember"}, {"IS_ASCII": True, "OP": "+"}] # "remember <something>"
        pattern_save_3 = [{"LOWER": "store"}, {"IS_ASCII": True, "OP": "+"}] # "store <something>"
        self.matcher.add("save_info", [pattern_save_1, pattern_save_2, pattern_save_3])

        # Pattern for "summarization_intent"
        pattern_summarize_1 = [{"LOWER": "summarize"}, {"IS_ASCII": True, "OP": "+"}] # "summarize <something>"
        pattern_summarize_2 = [{"LOWER": "tl;dr"}, {"IS_ASCII": True, "OP": "+"}] # "tl;dr <something>"
        pattern_summarize_3 = [{"LOWER": "give"}, {"LOWER": "me"}, {"LOWER": "a"}, {"LOWER": "summary"}, {"LOWER": "of"}, {"IS_ASCII": True, "OP": "+"}]
        self.matcher.add("summarization_intent", [pattern_summarize_1, pattern_summarize_2, pattern_summarize_3])

    def process_input(self, raw_input: any) -> str:
        """
        Processes the raw input by converting to string, cleaning, and logging.

        Args:
            raw_input: The input to process. Can be of any type that can be
                       converted to a string.

        Returns:
            The processed string input, or an empty string if processing fails.
        """
        self.logger.debug(f"Received raw input for processing: {raw_input}")

        try:
            if not isinstance(raw_input, str):
                # Attempt conversion for non-string types
                processed_input = str(raw_input)
                self.logger.debug(f"Successfully converted input of type {type(raw_input)} to string.")
            else:
                processed_input = raw_input

            # Basic text cleaning
            processed_input = processed_input.lower()
            processed_input = processed_input.strip()

            self.logger.debug(f"Successfully processed input: '{processed_input}'") # Changed to debug as it's verbose for every call
            return processed_input
        except Exception as e:
            self.logger.error(f"Error during input processing for raw_input='{raw_input}': {e}", exc_info=True)
            # Return a default or signal error; returning empty string as per previous logic
            return "" # Or raise a custom exception e.g., InputProcessingError(str(e))

    async def detect_intent(self, processed_input: str) -> IntentExtractionResult:
        """
        Detects the intent and extracts entities from the processed input.
        This is a placeholder implementation.

        Args:
            processed_input: The cleaned and processed input string.

        Returns:
            An IntentExtractionResult object.
        """
        self.logger.debug(f"Attempting to detect intent for: '{processed_input}'")
        if not self.nlp or not self.matcher:
            self.logger.error("spaCy nlp model or matcher not initialized. Falling back to basic intent detection.")
            # Fallback to a very basic keyword check if spaCy is not available
            if "search" in processed_input: intent_val = "search_info"
            elif "save" in processed_input: intent_val = "save_info"
            elif "summarize" in processed_input: intent_val = "summarization_intent"
            else: intent_val = "unknown_intent"
            return IntentExtractionResult(
                intent=intent_val,
                entities={"error_message": "spaCy model not available"},
                original_input=processed_input,
                confidence=self.agent_config.default_intent_confidence # Use config
            )

        try:
            doc = self.nlp(processed_input)
            matches = self.matcher(doc)

            intent_val = "unknown_intent"
            entities_val = {}
            # Use default_intent_confidence as a starting point, will be overridden by successful match
            confidence_val = self.agent_config.default_intent_confidence
            matched_patterns_details = []

            if matches:
                # For simplicity, take the first match's intent.
                # More sophisticated logic could prioritize or combine intents.
                match_id, start, end = matches[0]
                intent_val = self.nlp.vocab.strings[match_id]
                span = doc[start:end]  # The matched span
                matched_text = span.text
                self.logger.debug(f"Matched span: {matched_text} for intent: {intent_val}")

                # Simple entity extraction: consider the text after the matched keyword(s) as the entity
                # This is a basic approach and can be significantly improved.
                if intent_val == "search_info":
                    # Example: "search for cats" -> query: "cats"
                    # Find the token where the main keyword ends and extract the rest
                    keyword_end_token_index = 0
                    if doc[start].lower_ in ["search", "find"]: # first token of match
                        keyword_end_token_index = start + 1
                    elif doc[start].lower_ == "look" and doc[start+1].lower_ == "up": # "look up"
                         keyword_end_token_index = start + 2

                    if keyword_end_token_index < end:
                         entities_val["query"] = doc[keyword_end_token_index:end].text.strip()

                elif intent_val == "save_info":
                    keyword_end_token_index = 0
                    if doc[start].lower_ in ["save", "remember", "store"]:
                        keyword_end_token_index = start + 1

                    if keyword_end_token_index < end:
                        entities_val["data"] = doc[keyword_end_token_index:end].text.strip()

                elif intent_val == "summarization_intent":
                    keyword_end_token_index = 0
                    if doc[start].lower_ == "summarize":
                         keyword_end_token_index = start + 1
                    elif doc[start].lower_ == "tl;dr": # "tl;dr"
                         keyword_end_token_index = start + 1
                    elif doc[start].lower_ == "give": # "give me a summary of"
                        keyword_end_token_index = start + 5

                    if keyword_end_token_index < end:
                         entities_val["text_to_summarize"] = doc[keyword_end_token_index:end].text.strip()

                # Use high_confidence_threshold for successful spaCy rule matches
                confidence_val = self.agent_config.high_confidence_threshold
                for match_id, start, end in matches:
                    matched_patterns_details.append({
                        "pattern_name": self.nlp.vocab.strings[match_id],
                        "matched_text": doc[start:end].text,
                        "start_token": start,
                        "end_token": end
                    })
            else:
                self.logger.debug(f"No spaCy Matcher rules matched for: '{processed_input}'")
                # Optional: Could add simple keyword check as a fallback if no spaCy rules match
                if "search" in processed_input: intent_val = "search_info"
                elif "save" in processed_input: intent_val = "save_info"
                elif "summarize" in processed_input: intent_val = "summarization_intent"
                # else: intent_val remains "unknown_intent" as initialized
                # Confidence remains self.agent_config.default_intent_confidence if only basic keyword match
                if intent_val != "unknown_intent": # Log if basic keyword match changed intent
                    self.logger.debug(f"No spaCy match, fell back to keyword match for intent: '{intent_val}'")


            self.logger.info(f"Detected intent='{intent_val}', entities={entities_val}, confidence={confidence_val} (spaCy match: {bool(matches)}).")
            return IntentExtractionResult(
                intent=intent_val,
                entities=entities_val,
                confidence=confidence_val,
                original_input=processed_input,
                spacy_doc=doc,
                matched_patterns=matched_patterns_details
            )
        except Exception as e:
            self.logger.error(f"Error during spaCy intent detection for input='{processed_input}': {e}", exc_info=True)
            return IntentExtractionResult(
                intent="intent_detection_error",
                entities={"error_message": str(e)},
                original_input=processed_input,
                confidence=0.0
            )

    def decide_route(self, intent_result: IntentExtractionResult) -> RoutingDecision:
        """
        Decides the target module and task parameters based on intent_result.
        This is a placeholder implementation with rule-based routing.

        Args:
            intent_result: The IntentExtractionResult object.

        Returns:
            A RoutingDecision object.
        """
        self.logger.debug(f"Attempting to decide route for intent_result: Intent='{intent_result.intent}', Confidence='{intent_result.confidence}', Entities='{intent_result.entities}'")

        target_module_val = "ErrorHandler" # Default to ErrorHandler
        task_parameters_val = {"original_intent_info": intent_result.dict(exclude_none=True)} # Start with full intent info

        try:
            current_intent = intent_result.intent
            current_entities = intent_result.entities
            confidence = intent_result.confidence

            if current_intent == "intent_detection_error":
                task_parameters_val["error_message"] = "Intent detection failed."
                task_parameters_val["details"] = intent_result.entities.get("error_message", "No specific error details from detection.")
                self.logger.error(f"Routing to ErrorHandler due to intent_detection_error. Details: {task_parameters_val['details']}")

            elif confidence is None or confidence < self.agent_config.low_confidence_threshold:
                task_parameters_val["error_message"] = "Intent unclear or confidence too low."
                task_parameters_val["confidence_score"] = confidence
                task_parameters_val["detected_intent"] = current_intent
                self.logger.warn(
                    f"Routing to ErrorHandler: Intent '{current_intent}' confidence {confidence} is below threshold {self.agent_config.low_confidence_threshold}."
                )
            # Clearer routing based on high-confidence intents
            elif current_intent == "search_info":
                target_module_val = "LLMRetriever"
                task_parameters_val = current_entities
                self.logger.info(f"Routing to LLMRetriever for intent '{current_intent}'.")
            elif current_intent == "save_info":
                target_module_val = "LLMListener"
                task_parameters_val = current_entities
                self.logger.info(f"Routing to LLMListener for intent '{current_intent}'.")
            elif current_intent == "summarization_intent":
                target_module_val = "LLMListener" # Or a dedicated SummarizationService
                task_parameters_val = {
                    "raw_data": current_entities.get("text_to_summarize"),
                    "context_instructions": {"process_for_summarization": True, "summarize": True}
                }
                self.logger.info(f"Routing to LLMListener for intent '{current_intent}'.")
            elif current_intent == "unknown_intent":
                task_parameters_val["error_message"] = "Unknown intent detected."
                self.logger.warn(f"Routing to ErrorHandler due to 'unknown_intent'.")
            else: # Default fallback for any other unrecognized intent string not caught above
                task_parameters_val["error_message"] = f"Unrecognized high-confidence intent string: {current_intent}"
                self.logger.warn(f"Routing to ErrorHandler due to unrecognized high-confidence intent: '{current_intent}'.")

            # Ensure original_intent is part of the final decision for context
            return RoutingDecision(
                target_module=target_module_val,
                task_parameters=task_parameters_val,
                original_intent=intent_result
            )
        except Exception as e:
            self.logger.error(f"Error during routing decision for intent='{intent_result.intent}': {e}", exc_info=True)
            return RoutingDecision(
                target_module="ErrorHandler",
                task_parameters={"error_message": "Critical error during routing decision process.", "details": str(e)},
                original_intent=intent_result
            )

    async def dispatch_task(self, route: RoutingDecision) -> TaskResult:
        """
        Dispatches the task based on the routing decision.
        This is a stubbed implementation.

        Args:
            route: The RoutingDecision object.

        Returns:
            A TaskResult object.
        """
        target_module = route.target_module
        task_params = route.task_parameters
        self.logger.info(f"Attempting to dispatch task to module='{target_module}'.")
        self.logger.debug(f"Task parameters for dispatch: {task_params}")

        # Default error details in case of early exit or unhandled exception
        error_details_payload = {"target_module": target_module, "task_parameters": task_params}

        try:
            if target_module == "LLMRetriever":
                if not self.llm_service:
                    self.logger.error("LLMService (self.llm_service) is not available for LLMRetriever.")
                    error_details_payload["reason"] = "LLMService not configured."
                    return TaskResult(status="error", message="LLMRetriever service not configured.", error_details=error_details_payload)
                if not hasattr(self.llm_service, 'retrieve'):
                    self.logger.error("LLMService does not have a 'retrieve' method for LLMRetriever.")
                    error_details_payload["reason"] = "LLMService 'retrieve' method missing."
                    return TaskResult(status="error", message="LLMRetriever service method 'retrieve' not found.", error_details=error_details_payload)

                query = task_params.get('query')
                if query is None:
                    self.logger.error("'query' not found in task_params for LLMRetriever.")
                    error_details_payload["reason"] = "'query' parameter missing."
                    return TaskResult(status="error", message="Missing 'query' parameter for LLMRetriever.", error_details=error_details_payload)

                # Prepare kwargs by excluding 'query' if it's already explicitly passed
                kwargs_params = {k: v for k, v in task_params.items() if k != 'query'}
                self.logger.debug(f"Calling self.llm_service.retrieve(query='{query}', **{kwargs_params})")

                try:
                    response_data = await self.llm_service.retrieve(query=query, **kwargs_params)
                    self.logger.info(f"LLMRetriever call successful. Response data type: {type(response_data)}")
                    self.logger.debug(f"LLMRetriever response data: {response_data}")
                    return TaskResult(status="success", data=response_data, message="LLMRetriever processed successfully.")
                except Exception as service_exc:
                    self.logger.error(f"Exception during LLMRetriever service call: {service_exc}", exc_info=True)
                    error_details_payload["exception_type"] = type(service_exc).__name__
                    error_details_payload["exception_message"] = str(service_exc)
                    return TaskResult(status="error", message=f"Error during LLMRetriever execution: {service_exc}", error_details=error_details_payload)

            elif target_module == "LLMListener":
                if not self.llm_service:
                    self.logger.error("LLMService (self.llm_service) is not available for LLMListener.")
                    error_details_payload["reason"] = "LLMService not configured."
                    return TaskResult(status="error", message="LLMListener service not configured.", error_details=error_details_payload)
                if not hasattr(self.llm_service, 'process'):
                    self.logger.error("LLMService does not have a 'process' method for LLMListener.")
                    error_details_payload["reason"] = "LLMService 'process' method missing."
                    return TaskResult(status="error", message="LLMListener service method 'process' not found.", error_details=error_details_payload)

                raw_data = task_params.get('raw_data')
                context_instructions = task_params.get('context_instructions')
                # raw_data could be None if only context_instructions are provided for certain operations

                # Prepare kwargs by excluding 'raw_data' and 'context_instructions'
                kwargs_params = {k: v for k, v in task_params.items() if k not in ['raw_data', 'context_instructions']}
                self.logger.debug(f"Calling self.llm_service.process(data='{raw_data}', context_instructions={context_instructions}, **{kwargs_params})")

                try:
                    response_data = await self.llm_service.process(data=raw_data, context_instructions=context_instructions, **kwargs_params)
                    self.logger.info(f"LLMListener call successful. Response data type: {type(response_data)}")
                    self.logger.debug(f"LLMListener response data: {response_data}")
                    return TaskResult(status="success", data=response_data, message="LLMListener processed successfully.")
                except Exception as service_exc:
                    self.logger.error(f"Exception during LLMListener service call: {service_exc}", exc_info=True)
                    error_details_payload["exception_type"] = type(service_exc).__name__
                    error_details_payload["exception_message"] = str(service_exc)
                    return TaskResult(status="error", message=f"Error during LLMListener execution: {service_exc}", error_details=error_details_payload)

            elif target_module == "ErrorHandler":
                self.logger.info(f"Handling error with ErrorHandler. Error details: {task_params}")
                # task_params itself becomes the error_details for ErrorHandler
                return TaskResult(status="error",
                                  message=task_params.get("error_message", "Error handled by ErrorHandler."),
                                  error_details=task_params)

            else: # Unknown module
                self.logger.warn(f"Unknown module for dispatch: '{target_module}'. Parameters: {task_params}")
                error_details_payload["reason"] = f"Unknown target module: {target_module}"
                return TaskResult(status="error", message=f"Unknown module: {target_module}", error_details=error_details_payload)

        except Exception as e: # Catch-all for unexpected errors within dispatch_task itself
            self.logger.error(f"Critical unhandled exception in dispatch_task for module '{target_module}': {e}", exc_info=True)
            error_details_payload["exception_type"] = type(e).__name__
            error_details_payload["exception_message"] = str(e)
            error_details_payload["reason"] = "Critical unhandled error in dispatch logic."
            return TaskResult(
                status="error",
                message=f"Critical error dispatching to {target_module}: {str(e)}",
                error_details=error_details_payload
            )

    async def handle_request(self, raw_input: any, conversation_id: str = None, current_context: dict = None) -> TaskResult:
        """
        Orchestrates the processing of a raw input to produce a response,
        potentially using conversation state and context.

        Args:
            raw_input: The initial input from the user or system.
            conversation_id: Optional. Identifier for the current conversation.
            current_context: Optional. Dictionary containing current contextual information.

        Returns:
            A TaskResult object representing the outcome of the request.
        """
        self.logger.info(f"--- Starting new request handling --- Conversation ID: {conversation_id if conversation_id else 'N/A'} ---")
        self.logger.debug(f"Received raw_input: '{raw_input}', conversation_id: '{conversation_id}', current_context: {current_context}")

        try:
            # Placeholder for state retrieval
            if self.state_manager and conversation_id:
                self.logger.debug(f"State Manager: Attempting to retrieve state for conversation_id: {conversation_id} (placeholder)")
                # retrieved_state = await self.state_manager.get_state(conversation_id)
                # if retrieved_state: current_context = {**retrieved_state, **(current_context or {})} # Example merge

            # 1. Process Input
            self.logger.debug("Step 1/5: Processing input...")
            processed_input = self.process_input(raw_input)
            if not processed_input and raw_input is not None and raw_input != "":
                self.logger.warn("Input processing resulted in an empty string for non-empty input. This might indicate an issue.")
                # Early exit for failed input processing if critical
                # return TaskResult(status="error", message="Input processing failed.", error_details={"raw_input": raw_input})
            self.logger.info(f"Step 1/5: Processed input: '{processed_input}'")

            # 2. Detect Intent
            self.logger.debug("Step 2/5: Detecting intent...")
            if current_context:
                self.logger.debug("Context available, could be used to refine intent detection.")
            intent_extraction_result = await self.detect_intent(processed_input)
            self.logger.info(f"Step 2/5: Detected intent: {intent_extraction_result.intent}, Entities: {intent_extraction_result.entities}, Confidence: {intent_extraction_result.confidence}, Matched Patterns: {len(intent_extraction_result.matched_patterns)}")
            if intent_extraction_result.intent == "intent_detection_error":
                 self.logger.warn(f"Intent detection failed. Details: {intent_extraction_result.entities.get('error_message')}")
                 # Routing to ErrorHandler will happen in decide_route based on this intent_extraction_result

            # 3. Decide Route
            self.logger.debug("Step 3/5: Deciding route...")
            if current_context:
                self.logger.debug("Context available, could be used to refine routing decisions.")
            routing_decision = self.decide_route(intent_extraction_result)
            self.logger.info(f"Step 3/5: Decided route: {routing_decision.dict()}") # Log Pydantic model as dict

            # 4. Dispatch Task
            self.logger.debug("Step 4/5: Dispatching task...")
            if current_context and routing_decision.target_module not in ["ErrorHandler"]:
                 self.logger.debug(f"Context available, could be merged into task_parameters for {routing_decision.target_module}.")
                # routing_decision.task_parameters['current_context'] = current_context # Example
            task_result = await self.dispatch_task(routing_decision)
            self.logger.info(f"Step 4/5: Response from dispatched task to '{routing_decision.target_module}': {task_result.dict()}") # Log Pydantic model

            # Placeholder for state saving
            if self.state_manager and conversation_id:
                # Example: updated_state = {**(current_context or {}), "last_intent": intent_extraction_result.intent, "last_task_status": task_result.status}
                self.logger.debug(f"State Manager: Attempting to save state for conversation_id: {conversation_id} (placeholder)")
                # await self.state_manager.save_state(conversation_id, updated_state)

            self.logger.info(f"Step 5/5: Final TaskResult for Conversation ID {conversation_id if conversation_id else 'N/A'}: {task_result.dict()}")
            self.logger.info(f"--- Finished request handling --- Conversation ID: {conversation_id if conversation_id else 'N/A'} ---")
            return task_result

        except Exception as e:
            self.logger.error(f"Critical unhandled exception in handle_request for Conversation ID {conversation_id if conversation_id else 'N/A'}: {e}", exc_info=True)
            return TaskResult(
                status="error",
                message=f"Critical error during request handling: {str(e)}",
                error_details={"conversation_id": conversation_id, "exception_type": type(e).__name__}
            )
