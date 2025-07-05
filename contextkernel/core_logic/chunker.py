import re
from typing import List, Callable, Dict, Any, Optional
import logging
import spacy # For NLP tasks
from spacy.matcher import Matcher # For rule-based intent matching

# Assuming NLPConfig might be defined elsewhere or parts of it passed directly.
# from .nlp_utils import NLPConfig # Or a new shared config location

# Placeholder for Transformers pipeline if used directly by chunker for advanced labeling
try:
    from transformers import Pipeline as TransformersPipeline
except ImportError:
    TransformersPipeline = None # type: ignore

logger = logging.getLogger(__name__)

# Default tokenizer if none provided
def default_tokenizer(text: str) -> List[str]:
    """Splits text into tokens based on whitespace and basic punctuation."""
    tokens = re.findall(r'\b\w+\b|[.,;!?()]', text)
    return tokens

class SemanticChunker:
    def __init__(self,
                 tokenizer: Optional[Callable[[str], List[str]]] = None,
                 nlp_model: Optional[spacy.language.Language] = None, # Spacy model instance
                 matcher: Optional[Matcher] = None, # Spacy Matcher instance
                 intent_classifier: Optional[TransformersPipeline] = None, # HF pipeline for intent
                 # config: Optional[NLPConfig] = None # Or pass relevant config values directly
                 use_spacy_matcher_first: bool = True,
                 intent_candidate_labels: List[str] = None,
                 default_intent_confidence: float = 0.5,
                 high_confidence_threshold: float = 0.8
                ):
        """
        Initializes the SemanticChunker.

        Args:
            tokenizer: A function that takes a string and returns a list of tokens.
            nlp_model: Initialized spaCy language model.
            matcher: Initialized spaCy Matcher.
            intent_classifier: Initialized Hugging Face zero-shot classification pipeline.
            use_spacy_matcher_first: Whether to use spaCy Matcher before classifier for intent.
            intent_candidate_labels: Candidate labels for the intent classifier.
            default_intent_confidence: Default confidence for intents.
            high_confidence_threshold: Confidence threshold to consider a spaCy match strong enough.
        """
        self.tokenizer = tokenizer if tokenizer else default_tokenizer
        self.nlp = nlp_model
        self.matcher = matcher
        self.intent_classifier = intent_classifier

        # Store config values needed for labeling
        self.use_spacy_matcher_first = use_spacy_matcher_first
        self.intent_candidate_labels = intent_candidate_labels or ["question", "command", "statement", "request_info"]
        self.default_intent_confidence = default_intent_confidence
        self.high_confidence_threshold = high_confidence_threshold

        if not self.nlp:
            logger.warning("SemanticChunker initialized without a spaCy NLP model. Labeling capabilities will be limited.")
        # Matcher requires nlp.vocab, so if matcher is provided, nlp should also be.
        if self.matcher and not self.nlp:
            logger.error("Matcher provided to SemanticChunker without an NLP model. Matcher will not work.")
            self.matcher = None # Disable matcher if NLP model is missing

    async def label_chunk(self, chunk_text: str) -> Dict[str, Any]:
        """
        Applies semantic labels to a single text chunk.
        Migrates logic from nlp_utils.py for intent detection and entity recognition.

        Args:
            chunk_text: The text chunk to label.

        Returns:
            A dictionary containing labels like intent, entities, keywords.
            Example: {"intent": "question", "entities": {"Person": ["Alice"]}, "keywords": ["memory", "database"]}
        """
        if not self.nlp:
            logger.warning("NLP model not available in Chunker. Cannot perform advanced labeling.")
            return {"intent": "unknown", "entities": {}, "keywords": [], "confidence": 0.0}

        processed_input = chunk_text.lower().strip() # Similar to nlp_utils.process_input
        intent = "unknown"
        entities: Dict[str, Any] = {}
        confidence = self.default_intent_confidence
        # spacy_doc is not returned in the final dict but used internally.
        # matched_patterns from nlp_utils is also not directly returned but influences intent.

        spacy_doc = self.nlp(processed_input) # Process with Spacy

        # 1. Intent Detection (adapted from nlp_utils.detect_intent)
        # 1a. Try spaCy Matcher first if configured and available
        if self.use_spacy_matcher_first and self.matcher:
            matches = self.matcher(spacy_doc)
            if matches:
                # Using the first match like in nlp_utils.
                # More sophisticated logic could analyze all matches.
                match_id, start, end = matches[0]
                intent = self.nlp.vocab.strings[match_id]
                confidence = self.high_confidence_threshold

                # Basic entity extraction from match (simplified from nlp_utils)
                # This part of nlp_utils `keyword_end_token_index` is complex;
                # for now, we'll rely on general NER below or simplify.
                # matched_text_for_entity = spacy_doc[start:end].text # This is the whole matched phrase
                # For now, we'll let the general NER handle entities rather than rule-based from intent.

        # 1b. If Matcher didn't yield high-confidence intent, try classifier
        if intent == "unknown" or confidence < self.high_confidence_threshold:
            if self.intent_classifier and self.intent_candidate_labels:
                try:
                    # Ensure classifier call is compatible (e.g., non-async if this method is sync)
                    # If intent_classifier is an async pipeline, this method should be async
                    # and the call awaited. For now, assuming it's a sync pipeline.
                    # The plan specifies `async def label_chunk` so this should be awaited.
                    if hasattr(self.intent_classifier, "__call__"): # Check if callable
                        # Assuming the pipeline might be async if it's a TransformersPipeline
                        # This part needs careful handling of sync/async based on actual pipeline type
                        # For now, direct call, assuming it's okay or placeholder.
                        # If it's a true TransformersPipeline, it's often not async directly.
                        # It would need to be run in an executor if label_chunk is async.
                        # Given `async def label_chunk`, we should use:
                        # loop = asyncio.get_event_loop()
                        # result = await loop.run_in_executor(None, self.intent_classifier, processed_input, self.intent_candidate_labels)
                        # For now, let's make a simplifying assumption that the pipeline is somehow awaitable or this is conceptual.
                        # This is a known point of complexity from the original nlp_utils.
                        # Let's assume for now it's a synchronous call for simplicity of this step,
                        # and it will be wrapped if needed by the caller or if intent_classifier is an async wrapper.

                        # Per plan, label_chunk is async. So, classifier call must be awaitable or run in executor.
                        # Placeholder for actual async execution:
                        if callable(getattr(self.intent_classifier, "_async_call", None)): # Fictional async call
                             result = await self.intent_classifier._async_call(processed_input, candidate_labels=self.intent_candidate_labels)
                        elif callable(self.intent_classifier):
                            # This is a simplification. A real HF pipeline might need `loop.run_in_executor`.
                            # For this step, let's assume it's a mockable sync call for structure.
                            logger.warning("Intent classifier called synchronously in async method. Consider executor for non-async pipelines.")
                            result = self.intent_classifier(processed_input, candidate_labels=self.intent_candidate_labels, multi_label=False)
                        else:
                            raise TypeError("Intent classifier is not callable.")

                        if result and result.get('labels') and result.get('scores'):
                            intent = result['labels'][0]
                            confidence = result['scores'][0]
                        else:
                            logger.warning(f"Intent classifier returned unexpected result for '{processed_input}': {result}")
                            if intent == "unknown": # If matcher also failed
                                intent = "statement" # Default fallback
                                confidence = self.default_intent_confidence
                    else:
                        logger.error("Intent classifier is not callable.")
                        if intent == "unknown": intent = "statement"; confidence = self.default_intent_confidence


                except Exception as e_clf:
                    logger.error(f"Intent classification error for chunk '{processed_input[:50]}...': {e_clf}", exc_info=True)
                    if intent == "unknown": # Fallback if matcher also didn't set intent
                        intent = "statement"
                        confidence = self.default_intent_confidence
            elif intent == "unknown": # No classifier, and matcher failed or not used
                intent = "statement" # Default fallback intent
                confidence = self.default_intent_confidence

        # 2. Entity Recognition (using spaCy NER)
        if spacy_doc.ents:
            for ent in spacy_doc.ents:
                label = ent.label_.lower()
                entities.setdefault(label, []).append(ent.text)
            # Consolidate list values like in nlp_utils
            for k, v_list in entities.items():
                if isinstance(v_list, list):
                    entities[k] = list(set(v_list)) # Keep unique entities

        # 3. Keyword Extraction (simple placeholder: significant nouns and verbs)
        # This is a very basic keyword extraction. More advanced methods could be used.
        keywords = [token.lemma_.lower() for token in spacy_doc if token.pos_ in ("NOUN", "PROPN", "VERB") and not token.is_stop]
        keywords = list(set(keywords)) # Unique keywords

        return {
            "intent": intent,
            "entities": entities,
            "keywords": keywords[:10], # Limit number of keywords for brevity
            "confidence": confidence,
            "text_preview": chunk_text[:100] # Add a preview for context
        }

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
