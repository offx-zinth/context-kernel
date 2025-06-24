import re
from typing import List, Callable

# Placeholder for a more sophisticated tokenizer if needed (e.g., from Hugging Face)
# For now, we'll use a simple whitespace and punctuation-based split for tokens.
def default_tokenizer(text: str) -> List[str]:
    """Splits text into tokens based on whitespace and basic punctuation."""
    # Remove punctuation and then split by space
    # text_no_punct = re.sub(r'[^\w\s]', '', text)
    # tokens = text_no_punct.lower().split()
    # More robust tokenization might be needed for true "semantic" chunking later
    tokens = re.findall(r'\b\w+\b|[.,;!?()]', text) # Keeps punctuation as separate tokens
    return tokens

class SemanticChunker:
    def __init__(self, tokenizer: Callable[[str], List[str]] = None):
        """
        Initializes the SemanticChunker.

        Args:
            tokenizer: A function that takes a string and returns a list of tokens.
                       If None, a default simple tokenizer will be used.
        """
        self.tokenizer = tokenizer if tokenizer else default_tokenizer

    def _chunk_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """
        Splits text into chunks based on a maximum number of tokens.
        This is a basic implementation. More advanced semantic chunking would
        require embedding models or NLP libraries.
        """
        if not text:
            return []

        tokens = self.tokenizer(text)
        if not tokens:
            return []

        chunks = []
        current_chunk_tokens = []
        current_token_count = 0

        for token in tokens:
            # This logic is a bit simplistic for "tokens" if they include punctuation.
            # A more sophisticated approach might count actual words or use a proper tokenizer's output.
            token_len = 1 # Each item from self.tokenizer is considered one token for now

            if current_token_count + token_len > max_tokens and current_chunk_tokens:
                # Join tokens to form the chunk string.
                # We need to be careful about how tokens are joined back, especially with punctuation.
                # This basic join might not perfectly reconstruct original spacing.
                chunk_str = self._join_tokens(current_chunk_tokens)
                chunks.append(chunk_str)
                current_chunk_tokens = []
                current_token_count = 0

            current_chunk_tokens.append(token)
            current_token_count += token_len

        if current_chunk_tokens:
            chunk_str = self._join_tokens(current_chunk_tokens)
            chunks.append(chunk_str)

        return chunks

    def _join_tokens(self, tokens: List[str]) -> str:
        """
        Joins a list of tokens back into a string.
        This basic version adds spaces, which might not be ideal for all tokenizers.
        """
        # A more sophisticated join would handle punctuation spacing better.
        # For example, not putting a space before a comma or period.
        text = ""
        for i, token in enumerate(tokens):
            if i > 0 and token not in [".", ",", ";", "!", "?", ")"] and (len(text) > 0 and text[-1] not in ["("]):
                text += " "
            text += token
        return text.strip()

    def split_text(self, text: str, max_tokens: int = 200, method: str = "tokens") -> List[str]:
        """
        Splits text into chunks.

        Args:
            text: The text to be chunked.
            max_tokens: The maximum number of tokens per chunk (applies if method is "tokens").
            method: The chunking method ("tokens", "sentence", "semantic").
                    Currently, only "tokens" is implemented.

        Returns:
            A list of text chunks.
        """
        if method == "tokens":
            return self._chunk_text_by_tokens(text, max_tokens)
        # elif method == "sentence":
        #     # Placeholder for sentence-based chunking
        #     # return self._chunk_text_by_sentences(text, max_tokens_hint) # Sentences might also have a token limit
        #     raise NotImplementedError("Sentence-based chunking is not yet implemented.")
        # elif method == "semantic":
        #     # Placeholder for semantic chunking using embedding models
        #     raise NotImplementedError("Semantic chunking is not yet implemented.")
        else:
            raise ValueError(f"Unsupported chunking method: {method}")

if __name__ == '__main__':
    chunker = SemanticChunker()

    sample_text = (
        "This is a sample text. It has several sentences. "
        "We want to split this into smaller chunks. For example, based on token count. "
        "Let's see how it works. This should be fun and educational! Right?"
    )

    print("Original Text:")
    print(sample_text)
    print("-" * 30)

    # Test token-based chunking
    token_chunks = chunker.split_text(sample_text, max_tokens=10, method="tokens")
    print("\nToken-based chunks (max_tokens=10):")
    for i, chunk in enumerate(token_chunks):
        print(f"Chunk {i+1}: \"{chunk}\" (Tokens: {len(chunker.tokenizer(chunk))})")

    print("-" * 30)
    token_chunks_larger = chunker.split_text(sample_text, max_tokens=20, method="tokens")
    print("\nToken-based chunks (max_tokens=20):")
    for i, chunk in enumerate(token_chunks_larger):
        print(f"Chunk {i+1}: \"{chunk}\" (Tokens: {len(chunker.tokenizer(chunk))})")

    # Example with different tokenizer (e.g. simple word split)
    def simple_word_tokenizer(text: str) -> List[str]:
        return text.lower().split()

    word_chunker = SemanticChunker(tokenizer=simple_word_tokenizer)
    word_chunks = word_chunker.split_text(sample_text, max_tokens=10, method="tokens")
    print("\nWord-based chunks (custom tokenizer, max_tokens=10):")
    # Note: The _join_tokens method is very basic and might not reconstruct perfectly with all tokenizers
    for i, chunk in enumerate(word_chunks):
         print(f"Chunk {i+1}: \"{chunk}\" (Tokens: {len(word_chunker.tokenizer(chunk))})")

    # Test empty string
    empty_chunks = chunker.split_text("", max_tokens=10)
    print(f"\nChunks from empty string: {empty_chunks}")

    # Test text shorter than max_tokens
    short_text = "This is short."
    short_chunks = chunker.split_text(short_text, max_tokens=10)
    print(f"\nChunks from short text: {short_chunks} (Tokens: {len(chunker.tokenizer(short_chunks[0])) if short_chunks else 0})")
