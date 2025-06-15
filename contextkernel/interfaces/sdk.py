import httpx
import json
from typing import Any, Dict, Optional

class ContextKernelClientError(Exception):
    """Base exception for ContextKernelClient errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

    def __str__(self):
        base_str = super().__str__()
        if self.status_code:
            base_str += f" (Status Code: {self.status_code})"
        if self.response_data:
            base_str += f" | Response: {json.dumps(self.response_data)}"
        return base_str

class ContextKernelClient:
    """
    A client for interacting with the ContextKernel API.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initializes the ContextKernelClient.

        Args:
            base_url: The base URL of the ContextKernel API (e.g., "http://localhost:8000").
            timeout: Default timeout for HTTP requests in seconds.
        """
        if not base_url.endswith('/'):
            base_url += '/'
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Returns an active httpx.AsyncClient, creating one if necessary."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self):
        """Closes the underlying httpx.AsyncClient."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Makes an asynchronous HTTP request to the specified endpoint.

        Args:
            method: HTTP method (e.g., "GET", "POST").
            endpoint: API endpoint path.
            **kwargs: Additional arguments to pass to httpx.request.

        Returns:
            The JSON response as a dictionary.

        Raises:
            ContextKernelClientError: If the request fails or returns a non-2xx status code.
        """
        client = await self._get_client()
        try:
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                response_data = e.response.json()
            except json.JSONDecodeError:
                response_data = {"error_details": e.response.text[:200]} # Truncate long non-JSON errors
            raise ContextKernelClientError(
                message=f"API request failed: {e.response.status_code} {e.response.reason_phrase} for {e.request.url}",
                status_code=e.response.status_code,
                response_data=response_data
            ) from e
        except httpx.RequestError as e:
            raise ContextKernelClientError(f"Request error for {e.request.url}: {e}") from e
        except json.JSONDecodeError as e:
            raise ContextKernelClientError(f"Failed to decode JSON response: {e}") from e


    async def chat(self, message: str, user_id: str, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sends a chat message to the ContextKernel.

        Args:
            message: The message content from the user.
            user_id: Unique identifier for the user.
            session_id: Optional session identifier.
            metadata: Optional metadata.

        Returns:
            A dictionary containing the kernel's response.
        """
        payload = {
            "user_id": user_id,
            "message": message,
        }
        if session_id:
            payload["session_id"] = session_id
        if metadata:
            payload["metadata"] = metadata

        return await self._request("POST", "chat", json=payload)

    async def ingest_document(
        self,
        source_uri: str,
        content: Optional[str] = None,
        data_type: Optional[str] = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingests a document (or data) into the ContextKernel.

        Args:
            source_uri: URI of the data source (e.g., URL, file path reference).
            content: Optional direct content to ingest.
            data_type: Type of the data (e.g., 'text', 'pdf').
            metadata: Optional metadata for the ingested data.

        Returns:
            A dictionary containing the ingestion status.
        """
        payload = {
            "source_uri": source_uri,
            "data_type": data_type,
        }
        if content is not None:
            payload["content"] = content
        if metadata:
            payload["metadata"] = metadata

        return await self._request("POST", "ingest", json=payload)

    async def retrieve_context(self, context_id: str) -> Dict[str, Any]:
        """
        Retrieves specific context information using its unique ID.

        Args:
            context_id: The unique identifier of the context to retrieve.

        Returns:
            A dictionary containing the context data.
        """
        return await self._request("GET", f"context/{context_id}")

    async def __aenter__(self):
        await self._get_client() # Ensure client is initialized
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        # This example assumes the API server (from api.py) is running on localhost:8000
        # uvicorn contextkernel.interfaces.api:app --reload

        client = ContextKernelClient(base_url="http://localhost:8000")

        async with client: # Using async context manager
            try:
                # 1. Ingest some data
                print("Ingesting document...")
                ingest_payload = {
                    "source_uri": "https://example.com/my_document.txt",
                    "content": "This is a test document about AI and kernels.",
                    "data_type": "text",
                    "metadata": {"category": "testing"}
                }
                ingest_response = await client.ingest_document(**ingest_payload)
                print(f"Ingest Response: {ingest_response}\n")
                doc_id = ingest_response.get("document_id")

                # 2. Send a chat message
                print("Sending chat message...")
                chat_payload = {
                    "user_id": "sdk_user_001",
                    "session_id": "sdk_session_123",
                    "message": "Hello, Kernel! Tell me about the document.",
                    "metadata": {"source": "sdk_test"}
                }
                chat_response = await client.chat(**chat_payload)
                print(f"Chat Response: {chat_response}\n")
                context_id_from_chat = chat_response.get("context_id")

                # 3. Retrieve context (using context_id from chat)
                if context_id_from_chat:
                    print(f"Retrieving context with ID: {context_id_from_chat}...")
                    try:
                        retrieved_context = await client.retrieve_context(context_id_from_chat)
                        print(f"Retrieved Context: {retrieved_context}\n")
                    except ContextKernelClientError as e:
                        print(f"Caught expected error for '{context_id_from_chat}': {e} (This may be expected with current stubs)\n")

                # 4. Retrieve context (using a known valid test ID from api.py stubs)
                print("Retrieving known context 'ctx_abc123_valid'...")
                known_context = await client.retrieve_context("ctx_abc123_valid")
                print(f"Known Context Response: {known_context}\n")


                # 5. Example of handling an error (e.g., context not found)
                print("Attempting to retrieve non-existent context...")
                try:
                    await client.retrieve_context("non_existent_context_id_12345")
                except ContextKernelClientError as e:
                    print(f"Caught expected error: {e}\n")

            except ContextKernelClientError as e:
                print(f"An SDK client error occurred: {e}")
            except httpx.ConnectError as e:
                print(f"Connection error: Could not connect to the API at {client.base_url}. "
                      "Please ensure the ContextKernel API server is running.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    # To run this example:
    # 1. Make sure the API server from `api.py` is running.
    #    (e.g., in another terminal: `python -m uvicorn contextkernel.interfaces.api:app --reload`)
    # 2. Run this script: `python contextkernel/interfaces/sdk.py`
    asyncio.run(main())

# End of sdk.py
# Ensure this replaces the old "sdk.py loaded" print and comments.
