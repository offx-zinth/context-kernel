import asyncio

class MockVectorDB:
    def __init__(self):
        self._vectors = {}  # Stores vectors with doc_id as key
        self._documents = {} # Stores document content with doc_id as key
        self._next_id = 1

    async def add_document(self, document_text: str, vector: list[float], metadata: dict = None):
        await asyncio.sleep(0.01)  # Simulate async I/O
        doc_id = str(self._next_id)
        self._next_id += 1

        self._vectors[doc_id] = vector
        self._documents[doc_id] = {"text": document_text, "metadata": metadata or {}}
        # print(f"MockVectorDB: Added document {doc_id} with vector {vector} and metadata {metadata}")
        return doc_id

    async def search_vectors(self, query_vector: list[float], top_k: int = 5, filters: dict = None):
        await asyncio.sleep(0.02)  # Simulate async I/O

        # This is a very simplistic mock search.
        # It doesn't actually compare vectors but returns some mock results.
        # It will try to filter by metadata if filters are provided.

        results = []
        for doc_id, doc_data in self._documents.items():
            if filters:
                match = True
                for key, value in filters.items():
                    if doc_data["metadata"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Simplified similarity score - can be enhanced if needed
            similarity_score = 0.95 - (len(results) * 0.05) # decreasing score for mock

            results.append({
                "id": doc_id,
                "text": doc_data["text"],
                "metadata": doc_data["metadata"],
                "score": similarity_score
            })

            if len(results) >= top_k:
                break

        # print(f"MockVectorDB: Searched with vector {query_vector}, top_k {top_k}, filters {filters}. Found {len(results)} results.")
        return results

    async def get_document(self, doc_id: str):
        await asyncio.sleep(0.005)
        return self._documents.get(doc_id)

    async def update_document(self, doc_id: str, document_text: str = None, vector: list[float] = None, metadata: dict = None):
        await asyncio.sleep(0.01)
        if doc_id not in self._documents:
            return False # Or raise error

        if document_text is not None:
            self._documents[doc_id]["text"] = document_text
        if vector is not None:
            self._vectors[doc_id] = vector
        if metadata is not None:
            self._documents[doc_id]["metadata"].update(metadata)
        return True

    async def delete_document(self, doc_id: str):
        await asyncio.sleep(0.01)
        if doc_id in self._documents:
            del self._documents[doc_id]
            if doc_id in self._vectors:
                del self._vectors[doc_id]
            return True
        return False

    def clear_all_data(self):
        self._vectors = {}
        self._documents = {}
        self._next_id = 1
