import logging
from typing import Any, Dict, Optional, Union
from abc import ABC, abstractmethod

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ContextKernel imports
from contextkernel.utils.config import AppSettings #, get_settings # get_settings will be used in main.py to load and attach
from contextkernel.utils.state_manager import AbstractStateManager # Assuming this is the abstract base class
# Placeholder for ContextAgent - this would typically be imported from core_logic
# from contextkernel.core_logic.context_agent import ContextAgent
from contextkernel.core_logic.exceptions import (
    ConfigurationError,
    MemoryAccessError,
    CoreLogicError,
    ExternalServiceError,
    # ApplicationError # A generic base for CK errors if defined
)


# Configure basic logging - This might be superseded by setup_logging in main.py
# logging.basicConfig(level=logging.INFO) # Commented out to allow main.py to control logging config
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ContextKernel API",
    description="API for interacting with the ContextKernel system.",
    version="0.1.0", # This could be sourced from AppSettings later
)

# --- Placeholder for ContextAgent (until actual class is available) ---
class ContextAgent(ABC):
    @abstractmethod
    async def handle_chat(
        self,
        chat_message: 'ChatMessage',
        session_id: Optional[str], # Or pass state_manager if session handling is internal
        state_manager: AbstractStateManager
    ) -> 'ContextResponse': # Exact return type might vary
        pass

    @abstractmethod
    async def ingest_data(self, data: 'IngestData', settings: AppSettings) -> 'IngestResponse':
        pass

    @abstractmethod
    async def get_context_details(self, context_id: str, state_manager: AbstractStateManager) -> Optional['ContextResponse']:
        pass


# --- Pydantic Models (Request/Response Schemas) ---
class ChatMessage(BaseModel):
    user_id: str = Field(..., example="user123", description="Unique identifier for the user.")
    session_id: Optional[str] = Field(None, example="session456", description="Optional session identifier for context continuity.")
    message: str = Field(..., example="Hello, ContextKernel!", description="The message content from the user.")
    metadata: Optional[Dict[str, Any]] = Field(None, example={"source": "webapp"}, description="Optional metadata accompanying the message.")


class ContextResponse(BaseModel):
    context_id: str = Field(..., example="ctx_abc123", description="Unique identifier for the context or session.")
    data: Dict[str, Any] = Field(..., example={"reply": "Hello from ContextKernel!", "history_summary": "..."},
                                 description="The primary response data, often including the kernel's reply and relevant context.")
    metadata: Optional[Dict[str, Any]] = Field(None, example={"timestamp": "2023-10-27T10:30:00Z", "llm_model_used": "gpt-4"},
                                               description="Optional metadata about the response, like timestamps or model info.")


class IngestData(BaseModel):
    source_uri: Optional[str] = Field(None, example="https://example.com/data.txt",
                              description="URI of the data source (e.g., URL, file path).")
    data_type: str = Field(default="text", example="text",
                                     description="Type of the data (e.g., 'text', 'pdf', 'json').")
    content: Optional[str] = Field(None,
                                   example="This is the content to be ingested...",
                                   description="Direct content to ingest. If provided, source_uri might be for reference.")
    document_id: Optional[str] = Field(None, example="doc_manual_id_123", description="Optional pre-assigned document ID.")
    metadata: Optional[Dict[str, Any]] = Field(None, example={"tags": ["ai", "kernel"], "source_system": "crm"},
                                               description="Optional metadata for the ingested data.")

class IngestResponse(BaseModel):
    document_id: str = Field(..., example="doc_xyz789", description="Unique identifier for the ingested document.")
    status: str = Field(..., example="success", description="Status of the ingestion process (e.g., 'success', 'pending', 'failed').")
    message: Optional[str] = Field(None, example="Data ingested successfully and is being processed.", description="Optional details about the ingestion status.")


# --- Dependency Provider Functions ---
# These functions assume that instances are attached to `request.app.state`
# in `main.py` during application startup.

async def get_state_manager(request: Request) -> AbstractStateManager:
    if not hasattr(request.app.state, 'state_manager') or request.app.state.state_manager is None:
        logger.error("StateManager not initialized or not found in app.state.")
        raise ConfigurationError("StateManager is not available.")
    return request.app.state.state_manager

async def get_context_agent(request: Request) -> ContextAgent:
    if not hasattr(request.app.state, 'context_agent') or request.app.state.context_agent is None:
        logger.error("ContextAgent not initialized or not found in app.state.")
        raise ConfigurationError("ContextAgent is not available.")
    return request.app.state.context_agent

async def get_settings_dependency(request: Request) -> AppSettings:
    if not hasattr(request.app.state, 'settings') or request.app.state.settings is None:
        logger.error("AppSettings not initialized or not found in app.state.")
        raise ConfigurationError("Application settings are not available.")
    return request.app.state.settings


# --- Global Exception Handlers ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException caught: Status {exc.status_code}, Detail: {exc.detail}")
    # For HTTPException, FastAPI's default handling is usually sufficient.
    # Re-raise if no custom JSON structure is needed beyond what HTTPException provides.
    # return JSONResponse(status_code=exc.status_code, content={"error": "Client error", "details": exc.detail})
    raise exc # Re-raise to use FastAPI's default handling or if another handler further up wants it.

@app.exception_handler(MemoryAccessError)
async def memory_access_error_handler(request: Request, exc: MemoryAccessError):
    logger.error(f"MemoryAccessError: {exc}", exc_info=True) # Log with stack trace
    return JSONResponse(
        status_code=500,
        content={"error": "Storage access error", "details": str(exc)},
    )

@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    logger.error(f"ConfigurationError: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, # Or 503 Service Unavailable if it implies a service cannot run
        content={"error": "Configuration issue", "details": str(exc)},
    )

@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(request: Request, exc: ExternalServiceError):
    logger.error(f"ExternalServiceError: {exc}", exc_info=True)
    return JSONResponse(
        status_code=502,  # Bad Gateway
        content={"error": "External service error", "details": str(exc)},
    )

@app.exception_handler(CoreLogicError)
async def core_logic_error_handler(request: Request, exc: CoreLogicError):
    logger.error(f"CoreLogicError: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Core application error", "details": str(exc)},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.critical(f"Unhandled generic exception: {exc}", exc_info=True) # Critical for unexpected errors
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected internal server error occurred", "details": "An internal error occurred. Please try again later."}, # Avoid leaking str(exc) for generic unknown errors
    )


# --- API Endpoints ---

# Define StreamMessage here if it's not imported from main.py or a shared types module
# This should ideally be imported from where it's defined (e.g., main.py or a types.py)
# For now, defining a local version for this file's context.
# from contextkernel.main import StreamMessage # Ideal import
from dataclasses import dataclass
import asyncio # For asyncio.Queue

@dataclass
class StreamMessage: # Local definition, ensure it matches the one in main.py
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


async def get_conversation_queue(request: Request) -> asyncio.Queue:
    if not hasattr(request.app.state, 'conversation_queue') or request.app.state.conversation_queue is None:
        logger.error("Conversation queue not initialized or not found in app.state.")
        # This is a server configuration error, so 500 or 503 might be appropriate
        raise HTTPException(status_code=503, detail="Message queue is not available.")
    return request.app.state.conversation_queue

class ChatAcceptedResponse(BaseModel):
    status: str = "Message received for processing"
    session_id: Optional[str] = None
    message_id: Optional[str] = None # Could be useful for client to track message

@app.post("/chat",
            response_model=ChatAcceptedResponse,
            status_code=202, # HTTP 202 Accepted
            tags=["Chat"],
            summary="Accepts a chat message for asynchronous processing.")
async def chat_endpoint(
    chat_message: ChatMessage,
    queue: asyncio.Queue = Depends(get_conversation_queue)
    # state_manager: AbstractStateManager = Depends(get_state_manager) # State manager might not be directly used here anymore
):
    """
    Receives a chat message and puts it onto the conversation stream (queue)
    for asynchronous processing by the cognitive loops. Returns an immediate
    202 Accepted response.
    """
    logger.info(f"Received POST /chat request for user '{chat_message.user_id}', session '{chat_message.session_id}', message: '{chat_message.message[:50]}...'")

    # Create a unique ID for this message if needed for tracking, or use session_id
    # For now, session_id from ChatMessage can be part of metadata.
    # A more robust system might generate a unique message ID here.

    stream_msg = StreamMessage(
        content=chat_message.message,
        source=f"api_chat_user_{chat_message.user_id}",
        metadata={
            "user_id": chat_message.user_id,
            "session_id": chat_message.session_id,
            "received_at": datetime.datetime.utcnow().isoformat(), # Using standard library datetime
            **(chat_message.metadata or {})
        }
    )

    try:
        await queue.put(stream_msg)
        logger.info(f"Message from user '{chat_message.user_id}' (session: {chat_message.session_id}) enqueued successfully.")
        return ChatAcceptedResponse(
            session_id=chat_message.session_id,
            # message_id= ... # if a unique message ID was generated
        )
    except asyncio.QueueFull:
        logger.error(f"Conversation queue is full. Cannot accept message from user '{chat_message.user_id}'.")
        raise HTTPException(status_code=503, detail="Server is busy, please try again later. Queue is full.")
    except Exception as e:
        logger.error(f"Unexpected error enqueuing message in /chat for user {chat_message.user_id}: {e}", exc_info=True)
        # This is an internal server error if putting to queue fails unexpectedly
        raise HTTPException(status_code=500, detail="Failed to enqueue message for processing.")


@app.post("/ingest", response_model=IngestResponse, status_code=202, tags=["Data Ingestion"])
async def ingest_endpoint(
    data: IngestData,
    agent: ContextAgent = Depends(get_context_agent),
    settings: AppSettings = Depends(get_settings_dependency)
):
    """
    Ingests new data into the ContextKernel via the ContextAgent.
    """
    logger.info(f"Received POST /ingest request: Source URI '{data.source_uri}', Type: '{data.data_type}', Doc ID: '{data.document_id}'")
    if not data.content and not data.source_uri:
        logger.warning(f"Bad request to /ingest: either 'content' or 'source_uri' must be provided.")
        raise HTTPException(status_code=400, detail="Either 'content' or 'source_uri' must be provided.")

    try:
        response = await agent.ingest_data(data=data, settings=settings)
        logger.info(f"Successfully processed /ingest request: Document ID '{response.document_id}', Status: '{response.status}'")
        return response
    except Exception as e:
        logger.error(f"Error during /ingest for source {data.source_uri or 'direct content'}: {e}", exc_info=True)
        # Re-raise the original exception to be caught by the global generic_exception_handler
        raise e


@app.get("/context/{context_id}", response_model=ContextResponse, tags=["Context Retrieval"])
async def get_context_endpoint(
    context_id: str,
    agent: ContextAgent = Depends(get_context_agent), # Or state_manager if context is simple state
    state_manager: AbstractStateManager = Depends(get_state_manager)
):
    """
    Retrieves specific context or session information using its unique ID via the ContextAgent.
    """
    logger.info(f"Received GET /context/{context_id} request")
    try:
        # Assuming ContextAgent has a method to get detailed context/session information
        # This might involve just the state_manager or more complex logic in the agent
        context_data = await agent.get_context_details(context_id=context_id, state_manager=state_manager)

        if context_data is None:
            logger.warning(f"Context ID '{context_id}' not found for /context request.")
            raise HTTPException(status_code=404, detail=f"Context with ID '{context_id}' not found.")

        logger.info(f"Successfully retrieved /context/{context_id}")
        return context_data
    except HTTPException: # Re-raise HTTPExceptions (like 404) directly
        raise
    except Exception as e:
        logger.error(f"Error during /context/{context_id} retrieval: {e}", exc_info=True)
        # Re-raise the original exception to be caught by the global generic_exception_handler
        raise e


# To run this FastAPI application (locally for development):
# (Instructions remain similar, assuming main.py will handle Uvicorn setup)
# ...

if __name__ == "__main__":
    # This block is primarily for direct execution testing of this api.py file.
    # In a full application, main.py would typically orchestrate Uvicorn.
    import uvicorn
    from contextkernel.utils.config import get_settings # For standalone run

    # --- Mockups for standalone running ---
    class MockStateManager(AbstractStateManager):
        _store = {}
        async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]: return self._store.get(session_id)
        async def save_state(self, session_id: str, state: Dict[str, Any]) -> None: self._store[session_id] = state
        async def delete_state(self, session_id: str) -> None:
            if session_id in self._store: del self._store[session_id]
        async def close(self): pass

    class MockContextAgent(ContextAgent):
        async def handle_chat(self, chat_message: ChatMessage, session_id: Optional[str], state_manager: AbstractStateManager) -> ContextResponse:
            logger.info(f"MockContextAgent: Handling chat for user {chat_message.user_id}")
            new_context_id = session_id or f"session_{chat_message.user_id}_{abs(hash(chat_message.message))}"
            response_data = {"reply": f"Mock echo: {chat_message.message}", "history": []}

            current_state = await state_manager.get_state(new_context_id)
            if current_state:
                response_data["history"] = current_state.get("history", [])
            response_data["history"].append(chat_message.message)
            await state_manager.save_state(new_context_id, {"history": response_data["history"]})

            return ContextResponse(context_id=new_context_id, data=response_data)

        async def ingest_data(self, data: IngestData, settings: AppSettings) -> IngestResponse:
            logger.info(f"MockContextAgent: Ingesting data from {data.source_uri or 'direct content'}")
            doc_id = data.document_id or f"doc_{abs(hash(data.source_uri or data.content))}"
            return IngestResponse(document_id=doc_id, status="success", message="Mock data ingested.")

        async def get_context_details(self, context_id: str, state_manager: AbstractStateManager) -> Optional[ContextResponse]:
            logger.info(f"MockContextAgent: Retrieving context for {context_id}")
            state = await state_manager.get_state(context_id)
            if state:
                return ContextResponse(context_id=context_id, data=state)
            return None

    logger.info("Starting Uvicorn server for ContextKernel API (standalone test mode)...")

    # In a real scenario, main.py would set these up.
    # For standalone testing of api.py, we initialize them here.
    try:
        app.state.settings = get_settings()
        app.state.state_manager = MockStateManager()
        app.state.context_agent = MockContextAgent()
        logger.info("Mock services and settings initialized for standalone run.")
    except Exception as e:
        logger.error(f"Failed to initialize mock services for standalone run: {e}", exc_info=True)
        raise

    uvicorn.run(app, host="0.0.0.0", port=8001) # Use a different port like 8001 for standalone

[end of contextkernel/interfaces/api.py]
