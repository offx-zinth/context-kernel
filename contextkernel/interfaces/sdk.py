import httpx
import json
import logging # Added logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__) # Added logger instance

class ContextKernelClientError(Exception):
    """
    Base exception for ContextKernelClient errors.
    Encapsulates errors originating from API responses or client-side issues.
    """
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        """
        Initializes the ContextKernelClientError.

        Args:
            message: The primary error message.
            status_code: The HTTP status code from the API response, if applicable.
            response_data: The parsed JSON error data from the API response, if applicable.
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

    def __str__(self):
        base_str = super().__str__()
        if self.status_code:
            base_str += f" (Status Code: {self.status_code})"
        if self.response_data:
            base_str += f" | API Response: {json.dumps(self.response_data)}"
        return base_str

class ContextKernelClient:
    """
    An asynchronous client for interacting with the ContextKernel API.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initializes the ContextKernelClient.

        Args:
            base_url: The base URL of the ContextKernel API (e.g., "http://localhost:8001/").
            timeout: Default timeout for HTTP requests in seconds.
        """
        if not base_url.endswith('/'):
            base_url += '/'
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"ContextKernelClient initialized with base_url: {self.base_url}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Returns an active httpx.AsyncClient, creating one if necessary."""
        if self._client is None or self._client.is_closed:
            logger.debug("Initializing httpx.AsyncClient.")
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self):
        """Closes the underlying httpx.AsyncClient if it's open."""
        if self._client and not self._client.is_closed:
            logger.debug("Closing httpx.AsyncClient.")
            await self._client.aclose()
            self._client = None
            logger.debug("httpx.AsyncClient closed.")
        else:
            logger.debug("httpx.AsyncClient already closed or not initialized.")


    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Makes an asynchronous HTTP request to the specified endpoint.

        Args:
            method: HTTP method (e.g., "GET", "POST").
            endpoint: API endpoint path (e.g., "chat", "context/some_id").
            **kwargs: Additional arguments to pass to httpx.request (e.g., json=payload).

        Returns:
            The JSON response from the API as a dictionary.

        Raises:
            ContextKernelClientError: If the request fails, returns a non-2xx status code,
                                      or if the response cannot be decoded as JSON.
        """
        client = await self._get_client()
        url = client.build_request(method, endpoint).url
        payload_summary = kwargs.get("json", {}).keys() if "json" in kwargs else "No JSON payload"
        logger.debug(f"Sending API request: {method} {url} (Payload keys: {payload_summary})")

        try:
            response = await client.request(method, endpoint, **kwargs)
            logger.debug(f"Received API response: Status {response.status_code} for {url}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                response_data = e.response.json()
            except json.JSONDecodeError:
                response_data = {"error_message": e.response.text[:200]} if e.response.text else {"error_message": "No response body"}
            # The error details are already logged when the exception is caught by the calling application.
            # Logging it here too would be redundant if the app logs exceptions.
            # However, if we want SDK to always log API errors regardless of app:
            # logger.error(f"API request failed: {e.response.status_code} for {url}. Response: {response_data}", exc_info=False)
            raise ContextKernelClientError(
                message=f"API request failed: {e.response.status_code} {e.response.reason_phrase} for {e.request.url}",
                status_code=e.response.status_code,
                response_data=response_data
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error for {url}: {e}", exc_info=True)
            raise ContextKernelClientError(f"Request error for {e.request.url}: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {url}: {e.msg}", exc_info=True)
            raise ContextKernelClientError(f"Failed to decode JSON response: {e.msg}", response_data={"raw_response": e.doc}) from e


    async def chat(self, message: str, user_id: str, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sends a chat message to the ContextKernel.
        (See _request for logging details)
        """
        payload = {
            "user_id": user_id,
            "message": message,
        }
        if session_id:
            payload["session_id"] = session_id
        if metadata:
            payload["metadata"] = metadata
        logger.info(f"Sending chat message for user_id: {user_id}, session_id: {session_id}")
        response = await self._request("POST", "chat", json=payload)
        logger.info(f"Chat response received for user_id: {user_id}, session_id: {session_id}")
        return response

    async def ingest_document(
        self,
        source_uri: str,
        content: Optional[str] = None,
        data_type: str = "text",
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingests a document or data content into the ContextKernel.
        (See _request for logging details)
        """
        payload: Dict[str, Any] = {
            "source_uri": source_uri,
            "data_type": data_type,
        }
        if content is not None:
            payload["content"] = content
        if document_id is not None:
            payload["document_id"] = document_id
        if metadata:
            payload["metadata"] = metadata

        if content is None and source_uri is None:
             raise ValueError("Either content or source_uri must be provided for ingestion.")
        logger.info(f"Ingesting document: {source_uri or document_id or 'direct_content'}")
        response = await self._request("POST", "ingest", json=payload)
        logger.info(f"Ingest response received for: {source_uri or document_id or 'direct_content'}")
        return response

    async def retrieve_context(self, context_id: str) -> Dict[str, Any]:
        """
        Retrieves specific context information using its unique ID.
        (See _request for logging details)
        """
        if not context_id:
            raise ValueError("context_id cannot be empty.")
        logger.info(f"Retrieving context for context_id: {context_id}")
        response = await self._request("GET", f"context/{context_id}")
        logger.info(f"Retrieve context response received for context_id: {context_id}")
        return response

    async def __aenter__(self):
        logger.debug("ContextKernelClient entering async context.")
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.debug("ContextKernelClient exiting async context. Closing client.")
        await self.close()


if __name__ == "__main__":
    import asyncio
    # Setup basic logging for the SDK when run directly for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def main():
        logger.info("Starting ContextKernelClient SDK example...")
        client = ContextKernelClient(base_url="http://localhost:8001")

        async with client:
            try:
                # 1. Ingest some data
                logger.info("--- SDK Example: Ingesting document ---")
                ingest_payload = {
                    "source_uri": "sdk://test_document.txt",
                    "content": "This is a test document about AI and kernels, ingested via SDK.",
                    "data_type": "text",
                    "document_id": "sdk_doc_001",
                    "metadata": {"category": "sdk_testing"}
                }
                ingest_response = await client.ingest_document(**ingest_payload)
                print(f"Ingest Response: {ingest_response}\n")

                # 2. Send a chat message
                logger.info("--- SDK Example: Sending chat message ---")
                chat_payload = {
                    "user_id": "sdk_user_002",
                    "session_id": "sdk_session_456",
                    "message": "Hello, Kernel! This is a test from the SDK.",
                    "metadata": {"source": "sdk_example_script"}
                }
                chat_response = await client.chat(**chat_payload)
                print(f"Chat Response: {chat_response}\n")
                context_id_from_chat = chat_response.get("context_id")

                # 3. Retrieve context
                if context_id_from_chat:
                    logger.info(f"--- SDK Example: Retrieving context with ID: {context_id_from_chat} ---")
                    retrieved_context = await client.retrieve_context(context_id_from_chat)
                    print(f"Retrieved Context: {retrieved_context}\n")

                # 4. Test error handling
                non_existent_id = "this_id_definitely_does_not_exist_123"
                logger.info(f"--- SDK Example: Retrieving non-existent context ID: {non_existent_id} ---")
                try:
                    await client.retrieve_context(non_existent_id)
                    print(f"ERROR: Call to retrieve_context with '{non_existent_id}' should have failed but didn't.\n")
                except ContextKernelClientError as e:
                    # Log the caught error using the SDK's logger for demonstration
                    logger.warning(f"SDK caught expected error for '{non_existent_id}': {e}")
                    # Print details as before for console output
                    print(f"Successfully caught expected error for '{non_existent_id}':")
                    print(f"  Error Type: {type(e).__name__}")
                    print(f"  Status Code: {e.status_code}")
                    print(f"  Message: {e}")
                    if e.response_data:
                        print(f"  API Error Field: {e.response_data.get('error')}")
                        print(f"  API Details Field: {e.response_data.get('details')}")
                    else:
                        print("  Response Data: None (This is unexpected for a 404 from the new API)")
                    print("")

            except ContextKernelClientError as e:
                logger.error(f"An SDK client error occurred: {e}", exc_info=True)
                print(f"An SDK client error occurred: {e}")
            except httpx.ConnectError as e:
                logger.critical(f"Connection error: Could not connect to API at {client.base_url}", exc_info=True)
                print(f"Connection error: Could not connect to the API at {client.base_url}. "
                      "Please ensure the ContextKernel API server (api.py) is running on port 8001.")
            except Exception as e:
                logger.error(f"An unexpected error occurred in SDK example: {e}", exc_info=True)
                print(f"An unexpected error occurred: {type(e).__name__} - {e}")
        logger.info("ContextKernelClient SDK example finished.")

    asyncio.run(main())

[end of contextkernel/interfaces/sdk.py]
