import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ContextKernel API",
    description="API for interacting with the ContextKernel system.",
    version="0.1.0",
)


# --- Pydantic Models ---
class ChatMessage(BaseModel):
    """Request model for sending a chat message."""
    user_id: str = Field(..., example="user123", description="Unique identifier for the user.")
    session_id: Optional[str] = Field(None, example="session456", description="Optional session identifier.")
    message: str = Field(..., example="Hello, ContextKernel!", description="The message content from the user.")
    metadata: Optional[Dict[str, Any]] = Field(None, example={"source": "webapp"}, description="Optional metadata.")


class ContextResponse(BaseModel):
    """Response model for context-related endpoints."""
    context_id: str = Field(..., example="ctx_abc123", description="Unique identifier for the context.")
    data: Dict[str, Any] = Field(..., example={"reply": "Hello from ContextKernel!", "history_summary": "..."},
                                 description="The context data, often including the kernel's reply.")
    metadata: Optional[Dict[str, Any]] = Field(None, example={"timestamp": "2023-10-27T10:30:00Z"},
                                               description="Optional metadata about the response.")


class IngestData(BaseModel):
    """Request model for ingesting data."""
    source_uri: str = Field(..., example="https://example.com/data.txt",
                              description="URI of the data source (e.g., URL, file path).")
    data_type: Optional[str] = Field("text", example="text",
                                     description="Type of the data (e.g., 'text', 'pdf', 'json').")
    content: Optional[str] = Field(None,
                                   example="This is the content to be ingested...",
                                   description="Direct content to ingest. If provided, source_uri might be for reference.")
    metadata: Optional[Dict[str, Any]] = Field(None, example={"tags": ["ai", "kernel"]},
                                               description="Optional metadata for the ingested data.")


class IngestResponse(BaseModel):
    """Response model for data ingestion status."""
    document_id: str = Field(..., example="doc_xyz789", description="Unique identifier for the ingested document.")
    status: str = Field(..., example="success", description="Status of the ingestion process.")
    message: Optional[str] = Field(None, example="Data ingested successfully.", description="Optional details.")


# --- Stub Functions for Core Logic ---
async def process_chat_message_stub(chat_message: ChatMessage) -> ContextResponse:
    """Stub for core logic to process a chat message."""
    logger.info(f"Stub: Processing chat message for user {chat_message.user_id}")
    # Simulate processing and generating a response
    return ContextResponse(
        context_id=f"ctx_{chat_message.session_id or 'new'}_{str(hash(chat_message.message))[:6]}",
        data={
            "reply": f"Echo: {chat_message.message}",
            "conversation_history": ["Previous message 1", "Previous message 2", chat_message.message],
            "user_intent": "greeting"  # Example derived information
        },
        metadata={
            "timestamp": "2024-05-21T12:00:00Z",
            "processing_time_ms": 50
        }
    )


async def ingest_data_stub(ingest_data: IngestData) -> IngestResponse:
    """Stub for core logic to ingest data."""
    logger.info(f"Stub: Ingesting data from source: {ingest_data.source_uri}")
    # Simulate data ingestion
    doc_id = f"doc_{str(hash(ingest_data.source_uri + (ingest_data.content or '')))[:10]}"
    return IngestResponse(
        document_id=doc_id,
        status="success",
        message=f"Data from '{ingest_data.source_uri}' (type: {ingest_data.data_type}) processed."
    )


async def retrieve_context_stub(context_id: str) -> Optional[ContextResponse]:
    """Stub for core logic to retrieve context."""
    logger.info(f"Stub: Retrieving context for context_id: {context_id}")
    # Simulate database lookup
    if context_id == "ctx_abc123_valid":
        return ContextResponse(
            context_id=context_id,
            data={
                "retrieved_information": "This is some cached context data.",
                "original_query": "Tell me about context.",
            },
            metadata={"retrieved_at": "2024-05-21T12:05:00Z"}
        )
    elif context_id.startswith("ctx_session456"): # Example of a dynamic context from chat
        return ContextResponse(
            context_id=context_id,
            data={
                "retrieved_information": f"This is context related to session in {context_id}",
                "history_summary": "User said hello."
            },
            metadata={"retrieved_at": "2024-05-21T12:05:00Z"}
        )
    return None


# --- API Endpoints ---

@app.post("/chat", response_model=ContextResponse, tags=["Chat"])
async def chat_endpoint(chat_message: ChatMessage):
    """
    Receives a chat message, processes it through the ContextKernel,
    and returns a contextual response.
    """
    logger.info(f"Received request for /chat: User '{chat_message.user_id}', Message: '{chat_message.message[:50]}...'")
    response = await process_chat_message_stub(chat_message)
    logger.info(f"Sending response for /chat: Context ID '{response.context_id}'")
    return response


@app.post("/ingest", response_model=IngestResponse, status_code=202, tags=["Data Ingestion"])
async def ingest_endpoint(data: IngestData):
    """
    Ingests new data into the ContextKernel from a given source.
    """
    logger.info(f"Received request for /ingest: Source URI '{data.source_uri}', Type: '{data.data_type}'")
    if not data.content and not data.source_uri: # Basic validation
        raise HTTPException(status_code=400, detail="Either 'content' or 'source_uri' must be provided.")
    response = await ingest_data_stub(data)
    logger.info(f"Sending response for /ingest: Document ID '{response.document_id}', Status: '{response.status}'")
    return response


@app.get("/context/{context_id}", response_model=ContextResponse, tags=["Context Retrieval"])
async def get_context_endpoint(context_id: str):
    """
    Retrieves specific context information using its unique ID.
    """
    logger.info(f"Received request for /context/{context_id}")
    context_data = await retrieve_context_stub(context_id)
    if context_data is None:
        logger.warning(f"Context ID '{context_id}' not found for /context request.")
        raise HTTPException(status_code=404, detail=f"Context with ID '{context_id}' not found.")
    logger.info(f"Sending response for /context/{context_id}: Found.")
    return context_data

# To run this FastAPI application (locally for development):
# 1. Save this file as `api.py` in `contextkernel/interfaces/`.
# 2. Make sure you have FastAPI and Uvicorn installed:
#    `pip install fastapi uvicorn`
# 3. Navigate to the directory containing `contextkernel` (e.g., your project root).
# 4. Run Uvicorn:
#    `uvicorn contextkernel.interfaces.api:app --reload`
# 5. Access the API docs at http://127.0.0.1:8000/docs

if __name__ == "__main__":
    # This block is for direct execution (e.g., `python api.py`)
    # It's more common to run FastAPI apps with Uvicorn as shown above.
    import uvicorn
    logger.info("Starting Uvicorn server for ContextKernel API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# End of api.py
# Ensure this replaces the old "api.py loaded" print and comments.
