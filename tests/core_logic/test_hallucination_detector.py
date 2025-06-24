import unittest
from unittest.mock import MagicMock

from core_logic.hallucination_detector import HallucinationDetector, ValidationResult

class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses if responses is not None else {}
        self.last_prompt = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        # Simulate LLM behavior based on keywords in the prompt
        if "Paris is the capital of France." in prompt:
            return "Paris is the capital of France."
        elif "The sky is green." in prompt:
            return "The sky is typically blue due to Rayleigh scattering. It is not green."
        elif "Error case" in prompt:
            raise Exception("Simulated LLM API error")
        else:
            return "Statement could not be verified."

class TestHallucinationDetector(unittest.TestCase):

    def test_detect_factually_accurate_statement(self):
        mock_llm = MockLLMClient()
        detector = HallucinationDetector(llm_client=mock_llm)
        statement = "Paris is the capital of France."
        result = detector.detect(statement)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.explanation, "")
        expected_prompt = (
            "Critically evaluate the following statement for factual accuracy. "
            "If it is accurate, repeat it verbatim. If not, explain the error. "
            f"Statement: \"{statement}\""
        )
        self.assertEqual(mock_llm.last_prompt, expected_prompt)

    def test_detect_factually_inaccurate_statement(self):
        mock_llm = MockLLMClient()
        detector = HallucinationDetector(llm_client=mock_llm)
        statement = "The sky is green."
        result = detector.detect(statement)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, "The sky is typically blue due to Rayleigh scattering. It is not green.")
        expected_prompt = (
            "Critically evaluate the following statement for factual accuracy. "
            "If it is accurate, repeat it verbatim. If not, explain the error. "
            f"Statement: \"{statement}\""
        )
        self.assertEqual(mock_llm.last_prompt, expected_prompt)

    def test_detect_statement_unable_to_verify(self):
        mock_llm = MockLLMClient()
        detector = HallucinationDetector(llm_client=mock_llm)
        statement = "This is a neutral statement." # Assumes mock will return "Statement could not be verified."
        result = detector.detect(statement)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, "Statement could not be verified.")

    def test_detect_llm_api_error(self):
        mock_llm = MockLLMClient()
        detector = HallucinationDetector(llm_client=mock_llm)
        statement = "Error case" # Triggers exception in mock
        result = detector.detect(statement)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, "Error during LLM call: Simulated LLM API error")

    def test_empty_statement(self):
        mock_llm = MockLLMClient()
        detector = HallucinationDetector(llm_client=mock_llm)
        statement = ""
        # The current implementation would send "" to the LLM.
        # Depending on LLM, it might return empty, or some generic response.
        # MockLLM returns "Statement could not be verified." for unknown inputs.
        result = detector.detect(statement)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, "Statement could not be verified.")

if __name__ == '__main__':
    unittest.main()
