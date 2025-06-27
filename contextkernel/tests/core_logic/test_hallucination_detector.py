import unittest
from unittest.mock import MagicMock

from contextkernel.core_logic.hallucination_detector import HallucinationDetector, ValidationResult

class TestHallucinationDetector(unittest.TestCase):

    def test_detect_valid_chunk(self):
        mock_llm_client = MagicMock()
        statement = "Paris is the capital of France."
        # Configure the mock LLM to return the statement verbatim for valid chunks
        mock_llm_client.complete.return_value = statement

        detector = HallucinationDetector(llm_client=mock_llm_client)
        result = detector.detect(statement)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.explanation, "")
        mock_llm_client.complete.assert_called_once()
        # Check that the prompt was constructed as expected (optional, but good for rigor)
        args, _ = mock_llm_client.complete.call_args
        self.assertIn(statement, args[0])
        self.assertIn("Critically evaluate", args[0])

    def test_detect_invalid_chunk(self):
        mock_llm_client = MagicMock()
        statement = "The sky is green."
        explanation = "The sky is typically blue due to Rayleigh scattering, not green."
        # Configure the mock LLM to return an explanation for invalid chunks
        mock_llm_client.complete.return_value = explanation

        detector = HallucinationDetector(llm_client=mock_llm_client)
        result = detector.detect(statement)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, explanation)
        mock_llm_client.complete.assert_called_once()
        args, _ = mock_llm_client.complete.call_args
        self.assertIn(statement, args[0])

    def test_detect_llm_error(self):
        mock_llm_client = MagicMock()
        statement = "This will cause an error."
        error_message = "LLM API is down."
        # Configure the mock LLM to raise an exception
        mock_llm_client.complete.side_effect = Exception(error_message)

        detector = HallucinationDetector(llm_client=mock_llm_client)
        result = detector.detect(statement)

        self.assertFalse(result.is_valid) # Or True if we want to treat LLM errors as "unable to verify"
                                         # Current HallucinationDetector code treats it as invalid.
        self.assertIn(f"Error during LLM call: {error_message}", result.explanation)
        mock_llm_client.complete.assert_called_once()

    def test_detect_empty_chunk(self):
        mock_llm_client = MagicMock()
        statement = ""
        # LLM might return empty or a canned response for empty input.
        # Let's assume it returns a specific response indicating it can't process empty.
        llm_response_for_empty = "Cannot evaluate an empty statement."
        mock_llm_client.complete.return_value = llm_response_for_empty

        detector = HallucinationDetector(llm_client=mock_llm_client)
        result = detector.detect(statement)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.explanation, llm_response_for_empty)

if __name__ == '__main__':
    unittest.main()
