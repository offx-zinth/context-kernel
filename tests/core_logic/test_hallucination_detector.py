import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock # Use AsyncMock for async methods

# Adjust import path based on actual project structure
# Assuming HallucinationDetector and LLMRetriever are in contextkernel.core_logic
from contextkernel.core_logic.hallucination_detector import HallucinationDetector, ValidationResult
from contextkernel.core_logic.llm_retriever import LLMRetriever # Import base class for type hinting

class MockLLMClient:
    def __init__(self):
        self.last_prompt = None
        # Predefined responses based on keywords for simplicity
        self.responses = {
            "Paris is the capital of France.": "ACCURATE",
            "The sky is green.": "The sky is typically blue due to Rayleigh scattering. It is not green.",
            "The moon is made of cheese.": "This statement is factually incorrect. The moon is composed of rock and minerals.",
            "Error case": Exception("Simulated LLM API error")
        }
        self.default_response = "Statement could not be verified by mock LLM."

    async def complete(self, prompt: str) -> str: # Changed to async
        self.last_prompt = prompt
        for key_phrase, response_val in self.responses.items():
            if key_phrase in prompt:
                if isinstance(response_val, Exception):
                    raise response_val
                return response_val
        return self.default_response

class MockLLMRetriever(LLMRetriever):
    def __init__(self):
        # Mock the retriever's dependencies if necessary for its constructor, or simplify
        super().__init__(retriever_config=MagicMock(), ltm_interface=MagicMock(), stm_interface=MagicMock(), graphdb_interface=MagicMock())
        self.last_query = None
        self.retrieval_responses = {} # query_substring: response_list

    async def retrieve(self, query: str, top_k: int = 3, **kwargs) -> list: # Made async, return type simplified for mock
        self.last_query = query
        for key_substring, response_list in self.retrieval_responses.items():
            if key_substring in query:
                return response_list
        return []

    def set_retrieve_response(self, query_substring: str, response: list):
        self.retrieval_responses[query_substring] = response


class TestHallucinationDetectorAsync(unittest.IsolatedAsyncioTestCase):

    async def test_detect_factually_accurate_statement(self):
        mock_llm = MockLLMClient()
        mock_retriever = MockLLMRetriever()
        detector = HallucinationDetector(llm_client=mock_llm, retriever=mock_retriever)

        statement = "Paris is the capital of France."
        result = await detector.detect(statement)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.explanation, "LLM assessed as accurate.")
        self.assertEqual(result.past_occurrences, []) # Expect empty if no retrieval triggered or no results

        expected_prompt_substring = "Critically evaluate the following statement for factual accuracy."
        self.assertIn(expected_prompt_substring, mock_llm.last_prompt)
        self.assertIn(statement, mock_llm.last_prompt)
        self.assertIsNone(mock_retriever.last_query) # Retriever should not be called if initially valid

    async def test_detect_factually_inaccurate_statement_no_retrieval_results(self):
        mock_llm = MockLLMClient()
        mock_retriever = MockLLMRetriever()
        detector = HallucinationDetector(llm_client=mock_llm, retriever=mock_retriever)

        statement = "The sky is green."
        expected_llm_explanation = "The sky is typically blue due to Rayleigh scattering. It is not green."
        result = await detector.detect(statement)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, expected_llm_explanation)
        self.assertEqual(result.past_occurrences, [])

        self.assertIsNotNone(mock_llm.last_prompt)
        self.assertIsNotNone(mock_retriever.last_query) # Retriever should be called
        self.assertIn(statement[:100], mock_retriever.last_query) # Check if query includes part of statement

    async def test_detect_factually_inaccurate_statement_with_retrieval_results(self):
        mock_llm = MockLLMClient()
        mock_retriever = MockLLMRetriever()

        statement = "The moon is made of cheese."
        mock_llm_explanation = "This statement is factually incorrect. The moon is composed of rock and minerals."
        retrieved_data = [{"id": "err_moon_cheese_001", "correction": "The moon is made of rock, not cheese."}]
        mock_retriever.set_retrieve_response("moon is made of cheese", retrieved_data)

        detector = HallucinationDetector(llm_client=mock_llm, retriever=mock_retriever)
        result = await detector.detect(statement)

        self.assertFalse(result.is_valid)
        self.assertIn(mock_llm_explanation, result.explanation)
        self.assertIn("Similar past occurrences found.", result.explanation)
        self.assertEqual(result.past_occurrences, retrieved_data)

        self.assertIsNotNone(mock_retriever.last_query)

    async def test_detect_llm_api_error_with_retrieval(self):
        mock_llm = MockLLMClient() # Error case is handled by its predefined responses
        mock_retriever = MockLLMRetriever()
        retrieved_data = [{"id": "err_generic_002", "details": "Past error related to 'Error case'"}]
        mock_retriever.set_retrieve_response("Error case", retrieved_data)

        detector = HallucinationDetector(llm_client=mock_llm, retriever=mock_retriever)
        statement = "Error case" # Triggers exception in mock LLM

        result = await detector.detect(statement)

        self.assertFalse(result.is_valid)
        self.assertIn("Error during LLM call: Simulated LLM API error", result.explanation)
        self.assertIn("Similar past occurrences found.", result.explanation) # Retriever should still be called
        self.assertEqual(result.past_occurrences, retrieved_data)

    async def test_detect_retriever_api_error(self):
        mock_llm = MockLLMClient()
        mock_retriever = MockLLMRetriever()
        mock_retriever.retrieve = AsyncMock(side_effect=Exception("Simulated retriever error")) # Make retrieve method itself raise error

        detector = HallucinationDetector(llm_client=mock_llm, retriever=mock_retriever)
        statement = "The sky is green." # This will make LLM return invalid, triggering retrieval

        result = await detector.detect(statement)

        self.assertFalse(result.is_valid) # Still false due to LLM
        self.assertIn("The sky is typically blue", result.explanation) # LLM explanation
        self.assertIn("Retriever error: Simulated retriever error", result.explanation) # Retriever error appended
        self.assertEqual(result.past_occurrences, []) # No occurrences as retriever failed

    async def test_empty_statement_triggers_retrieval(self):
        mock_llm = MockLLMClient() # Returns default "Statement could not be verified"
        mock_retriever = MockLLMRetriever()
        retrieved_data = [{"id": "empty_info", "details": "Information about empty statements"}]
        mock_retriever.set_retrieve_response("Information about hallucination or correction regarding: ", retrieved_data) # Query for empty string

        detector = HallucinationDetector(llm_client=mock_llm, retriever=mock_retriever)
        statement = ""

        result = await detector.detect(statement)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, "Statement could not be verified by mock LLM. | Similar past occurrences found.")
        self.assertEqual(result.past_occurrences, retrieved_data)

if __name__ == '__main__':
    unittest.main()
