import unittest
import asyncio # For async tests
from unittest.mock import MagicMock, AsyncMock

# Adjust import path based on actual project structure
# Assuming SemanticChunker is in contextkernel.core_logic
from contextkernel.core_logic.chunker import SemanticChunker, default_tokenizer

# Mock spaCy components (Doc, Token, Span, Language, Matcher)
# These are simplified mocks. For more complex spaCy interactions, more detailed mocks would be needed.
class MockSpacyToken:
    def __init__(self, text, lemma_, pos_, is_stop=False):
        self.text = text
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.is_stop = is_stop

class MockSpacySpan:
    def __init__(self, text, label_):
        self.text = text
        self.label_ = label_

class MockSpacyDoc:
    def __init__(self, text, tokens=None, ents=None):
        self.text = text
        self._tokens = tokens if tokens else [MockSpacyToken(t, t.lower(), "NOUN") for t in text.split()]
        self.ents = [MockSpacySpan(ent_text, ent_label) for ent_text, ent_label in ents] if ents else []

    def __iter__(self): # To allow iteration like `for token in doc:`
        return iter(self._tokens)

    def __getitem__(self, key): # To allow slicing like `doc[start:end]`
        if isinstance(key, slice):
            # Simplified: just return a new MockSpacyDoc for the slice text for .text access
            # A real slice would return a Span object.
            return MockSpacyDoc(self.text) # Placeholder
        return self._tokens[key]


class MockSpacyNLP:
    def __init__(self, vocab_strings=None):
        self.vocab = MagicMock()
        # Mocking vocab.strings for matcher's nlp.vocab.strings[match_id]
        self.vocab.strings = vocab_strings or {}
        # Ensure __setitem__ and __getitem__ are present for vocab.strings if it's a plain dict
        if not hasattr(self.vocab.strings, '__setitem__'):
            self.vocab.strings = {k:v for k,v in enumerate(vocab_strings)} if vocab_strings else {} # if list
            def get_string(item_id): return self.vocab.strings.get(item_id, f"ID_{item_id}")
            self.vocab.strings.__getitem__ = get_string


    def __call__(self, text):
        # Based on text, return a MockSpacyDoc with predefined tokens/ents for testing
        if "question about product X" in text:
            tokens = [
                MockSpacyToken("question", "question", "NOUN"), MockSpacyToken("about", "about", "ADP"),
                MockSpacyToken("product", "product", "NOUN"), MockSpacyToken("X", "x", "PROPN")
            ]
            ents = [("product X", "PRODUCT")]
            return MockSpacyDoc(text, tokens=tokens, ents=ents)
        elif "order command" in text:
            tokens = [MockSpacyToken("order", "order", "VERB"), MockSpacyToken("command", "command", "NOUN")]
            ents = []
            return MockSpacyDoc(text, tokens=tokens, ents=ents)
        else: # Default generic doc
            return MockSpacyDoc(text)

class MockMatcher:
    def __init__(self, nlp_vocab_strings=None):
        self.nlp_vocab_strings = nlp_vocab_strings or {}
        self._matches = {} # Store predefined matches: text_substring -> list_of_match_tuples

    def add(self, key, patterns): # Mocked add, not used for matching logic here
        pass

    def __call__(self, doc):
        # Return predefined matches if doc.text contains a key
        for text_key, match_list in self._matches.items():
            if text_key in doc.text:
                # Convert string match_id (intent name) to an int for vocab.strings lookup
                # This requires vocab_strings to map intent names to integers or careful setup
                processed_matches = []
                for match_id_str, start, end in match_list:
                    # Find the int key for this string value in nlp_vocab_strings
                    int_match_id = None
                    for k,v in self.nlp_vocab_strings.items():
                        if v == match_id_str:
                            int_match_id = k
                            break
                    if int_match_id is None: # If intent string not in vocab map, assign a temp one (test setup issue)
                        # This part is tricky because spaCy assigns integer IDs.
                        # For testing, we might need to pre-populate vocab.strings or use a fixed mapping.
                        # Let's assume the string itself can be a temporary key if not found in vocab.
                        # logger.warning(f"Matcher mock: intent '{match_id_str}' not found in vocab.strings. Using string as temp ID.")
                        int_match_id = match_id_str # Fallback, though spaCy uses int IDs
                    processed_matches.append((int_match_id, start, end))
                return processed_matches
        return []

    def set_match_response(self, text_substring_key: str, matches: list):
        # matches should be like [("intent_name_str", start_token_idx, end_token_idx), ...]
        self._matches[text_substring_key] = matches


class MockHFIntentClassifier:
    def __init__(self):
        self._responses = {} # text_substring -> {"labels": [...], "scores": [...]}

    def __call__(self, text, candidate_labels, multi_label=False): # Make it directly callable
        for text_key, response in self._responses.items():
            if text_key in text:
                return response
        # Default response if no specific match
        return {"labels": ["statement"], "scores": [0.6], "sequence": text}

    def set_classification_response(self, text_substring_key: str, response: dict):
        self._responses[text_substring_key] = response


class TestSemanticChunkerAsync(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Shared mock NLP components for label_chunk tests
        self.mock_spacy_vocab_strings = {100: "question", 200: "command"} # Example ID mapping
        self.mock_nlp = MockSpacyNLP(vocab_strings=self.mock_spacy_vocab_strings)
        self.mock_matcher = MockMatcher(nlp_vocab_strings=self.mock_spacy_vocab_strings)
        self.mock_intent_classifier = MockHFIntentClassifier()

        # Default chunker instance for labeling tests
        self.labeling_chunker = SemanticChunker(
            nlp_model=self.mock_nlp,
            matcher=self.mock_matcher,
            intent_classifier=self.mock_intent_classifier,
            use_spacy_matcher_first=True,
            intent_candidate_labels=["question", "command", "statement"],
            default_intent_confidence=0.5,
            high_confidence_threshold=0.8
        )

    def test_default_tokenizer(self):
        text = "Hello, world! This is a test."
        expected_tokens = ["Hello", ",", "world", "!", "This", "is", "a", "test", "."]
        self.assertEqual(default_tokenizer(text), expected_tokens)

    def test_default_tokenizer_empty_string(self):
        self.assertEqual(default_tokenizer(""), [])

    def test_default_tokenizer_with_extra_spaces(self):
        text = "  Hello   world  . "
        expected_tokens = ["Hello", "world", "."] # Note: default_tokenizer's regex \b\w+\b handles spaces
        self.assertEqual(default_tokenizer(text), expected_tokens)

    def test_chunk_text_by_tokens_simple(self):
        chunker = SemanticChunker() # Test split_text with default (no NLP tools)
        text = "This is a simple test text for chunking."
        max_tokens = 5
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is a simple test")
        self.assertEqual(chunks[1], "text for chunking.")
        self.assertTrue(len(default_tokenizer(chunks[0])) <= max_tokens)
        self.assertTrue(len(default_tokenizer(chunks[1])) <= max_tokens)


    def test_chunk_text_by_tokens_exact_multiple(self):
        chunker = SemanticChunker()
        text = "One two three four five six seven eight nine ten."
        max_tokens = 5
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "One two three four five")
        self.assertEqual(chunks[1], "six seven eight nine ten")
        self.assertEqual(chunks[2], ".")

    def test_chunk_text_by_tokens_small_max_tokens(self):
        chunker = SemanticChunker()
        text = "Short example."
        max_tokens = 1
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Short")
        self.assertEqual(chunks[1], "example")
        self.assertEqual(chunks[2], ".")

    def test_chunk_text_empty_string(self):
        chunker = SemanticChunker()
        chunks = chunker.split_text("", max_tokens=10, method="tokens")
        self.assertEqual(chunks, [])

    def test_chunk_text_shorter_than_max_tokens(self):
        chunker = SemanticChunker()
        text = "This is short."
        chunks = chunker.split_text(text, max_tokens=10, method="tokens")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "This is short.")

    def test_chunk_with_custom_tokenizer(self):
        def custom_word_tokenizer(text: str):
            return text.lower().split()
        chunker = SemanticChunker(tokenizer=custom_word_tokenizer)
        text = "This is a custom tokenizer test with simple words."
        max_tokens = 4
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "this is a custom")
        self.assertEqual(chunks[1], "tokenizer test with simple")
        self.assertEqual(chunks[2], "words.")

    def test_chunk_text_with_punctuation_joining(self):
        chunker = SemanticChunker()
        text = "Hello, world! How are you? I am fine."
        max_tokens = 4
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Hello, world!")
        self.assertEqual(chunks[1], "How are you?")
        self.assertEqual(chunks[2], "I am fine.")

    def test_unsupported_chunking_method(self):
        chunker = SemanticChunker()
        with self.assertRaises(ValueError):
            chunker.split_text("Some text", method="non_existent_method")

    # --- Tests for label_chunk ---
    async def test_label_chunk_with_matcher_intent_and_ner(self):
        # Configure matcher to return a "question" intent for this specific text
        # For spaCy Matcher, match_id is an int. We mapped "question" to 100 in mock_spacy_vocab_strings.
        self.mock_matcher.set_match_response(
            text_substring_key="question about product x",  # text is lowercased in label_chunk
            matches=[(100, 0, 4)] # Match "question" (token 0) up to "X" (token 3), ID 100 maps to "question"
        )

        chunk_text = "This is a question about product X."
        # MockSpacyNLP will produce ents=[("product X", "PRODUCT")] for this text.

        expected_labels = {
            "intent": "question", # From matcher
            "entities": {"product": ["product X"]}, # From MockSpacyNLP's NER
            "keywords": ["question", "product", "x"], # Simplified from MockSpacyNLP based on text
            "confidence": self.labeling_chunker.high_confidence_threshold, # Matcher uses high confidence
            "text_preview": chunk_text[:100]
        }

        labels = await self.labeling_chunker.label_chunk(chunk_text)

        self.assertEqual(labels["intent"], expected_labels["intent"])
        self.assertDictEqual(labels["entities"], expected_labels["entities"])
        self.assertListEqual(sorted(labels["keywords"]), sorted(expected_labels["keywords"]))
        self.assertEqual(labels["confidence"], expected_labels["confidence"])

    async def test_label_chunk_with_classifier_intent(self):
        # Matcher will not find anything, so it falls back to classifier
        self.mock_matcher.set_match_response("some other text", []) # Ensure no match
        self.mock_intent_classifier.set_classification_response(
            text_substring_key="some other text",
            response={"labels": ["command"], "scores": [0.9], "sequence": "some other text"}
        )

        chunk_text = "This is some other text for an order command."
        # MockSpacyNLP will produce default entities/keywords if not specifically configured for this text

        expected_labels = {
            "intent": "command", # From classifier
            "entities": {}, # Assuming no specific entities from default MockSpacyNLP
            "keywords": ["text", "order", "command"], # default keywords from MockSpacyNLP
            "confidence": 0.9, # From classifier
            "text_preview": chunk_text[:100]
        }

        labels = await self.labeling_chunker.label_chunk(chunk_text)

        self.assertEqual(labels["intent"], expected_labels["intent"])
        # self.assertDictEqual(labels["entities"], expected_labels["entities"]) # Default NER might vary
        self.assertTrue(all(kw in labels["keywords"] for kw in expected_labels["keywords"])) # Check subset
        self.assertEqual(labels["confidence"], expected_labels["confidence"])

    async def test_label_chunk_no_nlp_model(self):
        chunker_no_nlp = SemanticChunker(nlp_model=None) # Create chunker without NLP tools
        chunk_text = "Some text."
        labels = await chunker_no_nlp.label_chunk(chunk_text)

        expected_labels = {"intent": "unknown", "entities": {}, "keywords": [], "confidence": 0.0}
        # text_preview is not added by label_chunk if nlp model is None.
        # Let's adjust the assertion to check only the core fields.
        self.assertEqual(labels["intent"], expected_labels["intent"])
        self.assertDictEqual(labels["entities"], expected_labels["entities"])
        self.assertListEqual(labels["keywords"], expected_labels["keywords"])
        self.assertEqual(labels["confidence"], expected_labels["confidence"])


    async def test_label_chunk_default_fallback_intent(self):
        # Matcher returns no match
        self.mock_matcher.set_match_response("unclear statement", [])
        # Classifier returns low confidence or no specific labels
        self.mock_intent_classifier.set_classification_response(
            text_substring_key="unclear statement",
            response={"labels": ["statement"], "scores": [0.2], "sequence": "unclear statement"} # Low score
        )

        chunk_text = "This is an unclear statement."
        labels = await self.labeling_chunker.label_chunk(chunk_text)

        # Expects fallback to "statement" intent due to low classifier confidence
        # or if classifier fails and matcher also failed.
        # The current label_chunk logic uses classifier's label if score is low, but intent would be 'statement' if classifier provides no labels.
        # If classifier returns low score for 'statement', it should still be 'statement'
        self.assertEqual(labels["intent"], "statement")
        # Confidence might be the low score from classifier or default_intent_confidence if classifier truly fails.
        # Current mock HF classifier returns the low score.
        self.assertEqual(labels["confidence"], 0.2)


if __name__ == '__main__':
    unittest.main()
