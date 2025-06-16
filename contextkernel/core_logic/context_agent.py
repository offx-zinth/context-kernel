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
from typing import Dict, Any, Optional as TypingOptional # Renamed to avoid conflict with Pydantic's Optional
from pydantic import BaseModel, Field

# Pydantic Models for Data Structuring
class IntentExtractionResult(BaseModel):
    intent: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: TypingOptional[float] = None
    original_input: TypingOptional[str] = None # For error cases or context

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
    def __init__(self, llm_service, memory_system, config, state_manager=None):
        """
        Initializes the ContextAgent.

        Args:
            llm_service: Stub for the LLM service.
            memory_system: Stub for the memory system.
            config: Stub for the configuration.
            state_manager: Optional. Instance for managing conversation state.
        """
        self.llm_service = llm_service
        self.memory_system = memory_system
        self.config = config
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        if self.state_manager:
            self.logger.info("ContextAgent initialized with StateManager.")
        else:
            self.logger.info("ContextAgent initialized (no StateManager).") # Kept as INFO as it's a significant config detail

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
        try:
            intent_val = "unknown_intent"
            entities_val = {}
            confidence_val = 0.5 # Placeholder confidence

            # Placeholder logic for intent detection
            if "search" in processed_input:
                intent_val = "search_info"
                entities_val = {"query": processed_input.replace("search", "", 1).strip()}
                confidence_val = 0.8 # Higher confidence for keyword match
            elif "save" in processed_input:
                intent_val = "save_info"
                entities_val = {"data": processed_input.replace("save", "", 1).strip()}
                confidence_val = 0.85
            elif "summarize" in processed_input:
                intent_val = "summarization_intent"
                # Basic extraction: find "summarize " and take the rest.
                # More robust extraction would be needed for varied phrasing.
                summary_query_parts = processed_input.split("summarize ", 1)
                text_to_summarize = summary_query_parts[1] if len(summary_query_parts) > 1 else processed_input
                entities_val = {"text_to_summarize": text_to_summarize.strip()}
                confidence_val = 0.9 # High confidence for keyword match

            self.logger.debug(f"Successfully detected intent='{intent_val}', entities={entities_val}, confidence={confidence_val}")
            return IntentExtractionResult(
                intent=intent_val,
                entities=entities_val,
                confidence=confidence_val,
                original_input=processed_input
            )
        except Exception as e:
            self.logger.error(f"Error during intent detection for input='{processed_input}': {e}", exc_info=True)
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
        self.logger.debug(f"Attempting to decide route for intent_result: {intent_result.dict()}")
        try:
            target_module_val = ""
            task_parameters_val = {}
            current_intent = intent_result.intent
            current_entities = intent_result.entities

            if current_intent == "search_info":
                target_module_val = "LLMRetriever"
                task_parameters_val = current_entities
            elif current_intent == "save_info":
                target_module_val = "LLMListener"
                task_parameters_val = current_entities # e.g., {"data": "text to save"}
            elif current_intent == "summarization_intent":
                target_module_val = "LLMListener" # Or a dedicated SummarizationService if we had one
                # LLMListener's process_data expects `raw_data` and `context_instructions`.
                # We need to map `entities_val` ({"text_to_summarize": ...}) to this.
                # The `llm_service.process` method (which MockContextAgentLLMService implements)
                # will receive these task_parameters_val.
                # It will then need to call LLMListener.process_data appropriately.
                task_parameters_val = {
                    "raw_data": current_entities.get("text_to_summarize"),
                    "context_instructions": {"process_for_summarization": True, "summarize": True} # Signal to LLMListener
                }
            elif current_intent == "unknown_intent":
                target_module_val = "ErrorHandler"
                task_parameters_val = {"error_message": "Unknown intent detected.", "original_intent_info": intent_result.dict()}
                self.logger.warn(f"Routing to {target_module_val} due to '{current_intent}'. Params: {task_parameters_val}")
            elif current_intent == "intent_detection_error":
                target_module_val = "ErrorHandler"
                task_parameters_val = {"error_message": "Intent detection failed.", "original_intent_info": intent_result.dict()}
                self.logger.error(f"Routing to {target_module_val} due to intent detection error. Params: {task_parameters_val}")
            else: # Default fallback for any other unrecognized intent string
                target_module_val = "ErrorHandler"
                task_parameters_val = {"error_message": f"Unrecognized intent string: {current_intent}", "original_intent_info": intent_result.dict()}
                self.logger.warn(f"Routing to {target_module_val} due to unrecognized intent string '{current_intent}'. Params: {task_parameters_val}")

            self.logger.debug(f"Successfully decided route: target_module='{target_module_val}', task_parameters={task_parameters_val}")
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
        self.logger.debug(f"Attempting to dispatch task to module='{target_module}' with parameters={task_params}")

        try:
            if target_module == "LLMRetriever":
                if self.llm_service and hasattr(self.llm_service, 'retrieve'):
                    self.logger.debug(f"Calling self.llm_service.retrieve with {task_params}")
                    # Actual call to service; response_data would be its result
                    response_data = await self.llm_service.retrieve(task_params)
                    # Mocking service call for now
                    # mock_data = f"Retrieved data for query: '{task_params.get('query', 'N/A')}'"
                    return TaskResult(status="success", data=response_data, message="LLMRetriever processed successfully.")
                else:
                    self.logger.warn(f"LLMRetriever (self.llm_service or retrieve method) not available. Task params: {task_params}")
                    return TaskResult(status="error", message="LLMRetriever service not available.", error_details=task_params)

            elif target_module == "LLMListener":
                if self.llm_service and hasattr(self.llm_service, 'process'):
                    self.logger.debug(f"Calling self.llm_service.process with {task_params}")
                    response_data = await self.llm_service.process(task_params)
                    # mock_data = f"Processed data: '{task_params.get('data', 'N/A')}'"
                    return TaskResult(status="success", data=response_data, message="LLMListener processed successfully.")
                else:
                    self.logger.warn(f"LLMListener (self.llm_service or process method) not available. Task params: {task_params}")
                    return TaskResult(status="error", message="LLMListener service not available.", error_details=task_params)

            elif target_module == "ErrorHandler":
                self.logger.info(f"Handling error with ErrorHandler. Parameters: {task_params}")
                # task_params here usually contains error_message and original_intent_info or details
                return TaskResult(status="error", message=task_params.get("error_message", "Error handled by ErrorHandler."),
                                  error_details=task_params) # task_params itself becomes the error_details

            else: # Unknown module
                self.logger.warn(f"Unknown module for dispatch: {target_module}. Parameters: {task_params}")
                return TaskResult(status="error", message=f"Unknown module: {target_module}", error_details=task_params)

        except Exception as e:
            self.logger.error(f"Exception during dispatch to '{target_module}' with params='{task_params}': {e}", exc_info=True)
            return TaskResult(
                status="error",
                message=f"Exception during dispatch to {target_module}: {str(e)}",
                error_details={"target_module": target_module, "task_parameters": task_params, "exception": str(e)}
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
            self.logger.info(f"Step 2/5: Detected intent: {intent_extraction_result.dict()}") # Log Pydantic model as dict
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
