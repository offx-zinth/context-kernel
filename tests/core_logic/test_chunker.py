import unittest
from core_logic.chunker import SemanticChunker, default_tokenizer

class TestSemanticChunker(unittest.TestCase):

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
        chunker = SemanticChunker()
        text = "This is a simple test text for chunking."
        # Tokens: This, is, a, simple, test, text, for, chunking, . (9 tokens)
        max_tokens = 5
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        # Expected:
        # Chunk 1: "This is a simple test" (5 tokens: This, is, a, simple, test)
        # Chunk 2: "text for chunking." (4 tokens: text, for, chunking, .)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is a simple test")
        self.assertEqual(chunks[1], "text for chunking.")
        self.assertTrue(len(default_tokenizer(chunks[0])) <= max_tokens)
        # The last chunk can be smaller
        self.assertTrue(len(default_tokenizer(chunks[1])) <= max_tokens)


    def test_chunk_text_by_tokens_exact_multiple(self):
        chunker = SemanticChunker()
        text = "One two three four five six seven eight nine ten."
        # Tokens: One, two, three, four, five, six, seven, eight, nine, ten, . (11 tokens)
        # If max_tokens = 5
        # "One two three four five"
        # "six seven eight nine ten"
        # "."
        max_tokens = 5
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "One two three four five")
        self.assertEqual(chunks[1], "six seven eight nine ten")
        self.assertEqual(chunks[2], ".")

    def test_chunk_text_by_tokens_small_max_tokens(self):
        chunker = SemanticChunker()
        text = "Short example." # Tokens: Short, example, . (3)
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
        text = "This is short." # Tokens: This, is, short, . (4)
        chunks = chunker.split_text(text, max_tokens=10, method="tokens")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "This is short.")

    def test_chunk_with_custom_tokenizer(self):
        # Custom tokenizer that splits by spaces only and counts words
        def custom_word_tokenizer(text: str):
            return text.lower().split()

        chunker = SemanticChunker(tokenizer=custom_word_tokenizer)
        text = "This is a custom tokenizer test with simple words."
        # Words: this, is, a, custom, tokenizer, test, with, simple, words. (9 words)
        max_tokens = 4 # Here "tokens" means "words" due to custom_word_tokenizer
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        # Expected:
        # "this is a custom"
        # "tokenizer test with simple"
        # "words."
        # Note: The _join_tokens method in SemanticChunker is basic.
        # It adds spaces between all tokens it's given.
        # If custom_word_tokenizer produced "words.", the joiner would treat "words." as one token.
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "this is a custom")
        self.assertEqual(chunks[1], "tokenizer test with simple")
        self.assertEqual(chunks[2], "words.") # The custom tokenizer splits "words." into "words."

    def test_chunk_text_with_punctuation_joining(self):
        chunker = SemanticChunker() # Uses default_tokenizer
        text = "Hello, world! How are you? I am fine."
        # Tokens: Hello, ,, world, !, How, are, you, ?, I, am, fine, . (12 tokens)
        max_tokens = 4
        chunks = chunker.split_text(text, max_tokens=max_tokens, method="tokens")
        # Expected based on current _join_tokens logic:
        # Chunk 1: "Hello, world!" (Tokens: Hello, ,, world, !)
        # Chunk 2: "How are you?" (Tokens: How, are, you, ?)
        # Chunk 3: "I am fine." (Tokens: I, am, fine, .)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Hello, world!")
        self.assertEqual(chunks[1], "How are you?")
        self.assertEqual(chunks[2], "I am fine.")

    def test_unsupported_chunking_method(self):
        chunker = SemanticChunker()
        with self.assertRaises(ValueError):
            chunker.split_text("Some text", method="non_existent_method")

if __name__ == '__main__':
    unittest.main()
