import asyncio

class MockLLM:
    async def generate_text(self, prompt: str, max_tokens: int = 50) -> str:
        await asyncio.sleep(0.01)  # Simulate async operation
        return f"Mock generated text for prompt: '{prompt[:30]}...' (max_tokens: {max_tokens})"

    async def summarize(self, text: str, max_length: int = 100) -> str:
        await asyncio.sleep(0.01)  # Simulate async operation
        return f"Mock summary for text: '{text[:50]}...' (max_length: {max_length})"

    async def classify_intent(self, text: str) -> dict:
        await asyncio.sleep(0.01) # Simulate async operation
        # Simulate a simple intent classification
        if "summarize" in text.lower():
            return {"intent": "summarization", "confidence": 0.9, "entities": {"document_type": "transcript"}}
        elif "question" in text.lower() or "what is" in text.lower():
            return {"intent": "question_answering", "confidence": 0.85, "entities": {}}
        else:
            return {"intent": "general_query", "confidence": 0.7, "entities": {}}

    async def generate_embedding(self, text: str) -> list[float]:
        await asyncio.sleep(0.01)
        # Simple embedding based on text length
        return [float(len(text)) / 100.0] * 10 # Return a 10-dimensional embedding
