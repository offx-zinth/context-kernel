import logging
import dataclasses # Keep for dataclasses.replace if used in summarize
import re
# import os # Marked as unused
# import json # Marked as unused
from typing import Optional, List, Dict, Any, Union

from pydantic_settings import BaseSettings, SettingsConfigDict # For Config model
from core_logic.chunker import SemanticChunker # Added for new chunker

try:
    from transformers import AutoTokenizer, pipeline, PreTrainedTokenizerFast, PreTrainedTokenizer
except ImportError:
    AutoTokenizer = pipeline = PreTrainedTokenizerFast = PreTrainedTokenizer = None # type: ignore

try:
    import nltk # type: ignore
except ImportError:
    nltk = None # type: ignore

try:
    from gensim.summarization import summarize as gensim_summarize # type: ignore
except ImportError:
    gensim_summarize = None # type: ignore

from .exceptions import SummarizationError, ConfigurationError, ExternalServiceError

logger = logging.getLogger(__name__)

class SummarizerConfig(BaseSettings):
    hf_abstractive_model_name: str = "facebook/bart-large-cnn"
    hf_api_key_env_var: Optional[str] = None
    hf_tokenizer_name: Optional[str] = None
    extractive_method: str = "gensim_textrank"
    style: str = "abstractive" # "abstractive" or "extractive"
    desired_length_type: str = "percentage"
    desired_length_value: Union[float, int] = 0.2
    chunk_size: int = 1024
    chunk_overlap: int = 200
    max_length_abstractive: int = 512
    min_length_abstractive: int = 30
    additional_params: Optional[Dict[str, Any]] = None

    model_config = SettingsConfigDict(env_prefix='SUMMARIZER_')


class Summarizer:
    def __init__(self, default_config: Optional[SummarizerConfig] = None):
        if default_config:
            self.default_config = default_config
        else:
            self.default_config = SummarizerConfig()

        self.tokenizer: Union[None, PreTrainedTokenizer, PreTrainedTokenizerFast] = None
        if AutoTokenizer is None: # Check if transformers is available
            # This is a critical issue if a tokenizer is needed for any configured operation.
            logger.error("Transformers library not available. Tokenizer cannot be initialized.")
            # Depending on how strictly Summarizer needs a tokenizer, could raise ConfigurationError here.
            # For now, allow initialization, but methods requiring tokenizer will fail.
        else:
            tokenizer_name = self.default_config.hf_tokenizer_name or self.default_config.hf_abstractive_model_name
            if not tokenizer_name:
                logger.warning("No tokenizer name or hf_abstractive_model_name in config. Tokenizer not loaded.")
                # Not raising ConfigurationError here, as some extractive methods might not need it.
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    logger.info(f"Tokenizer '{tokenizer_name}' loaded.")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}", exc_info=True)
                    raise ConfigurationError(f"Failed to load tokenizer '{tokenizer_name}': {e}") from e

        self.abstractive_pipeline = None
        if self.default_config.style == "abstractive":
            if pipeline is None:
                raise ConfigurationError("Transformers 'pipeline' function not available for abstractive summarization.")
            if not self.default_config.hf_abstractive_model_name:
                raise ConfigurationError("hf_abstractive_model_name required for abstractive style.")
            if not self.tokenizer: # Abstractive pipeline always needs a tokenizer
                 raise ConfigurationError("Tokenizer required for abstractive pipeline but failed to load or was not specified.")
            try:
                self.abstractive_pipeline = pipeline("summarization", model=self.default_config.hf_abstractive_model_name, tokenizer=self.tokenizer) # type: ignore
                logger.info(f"Abstractive pipeline '{self.default_config.hf_abstractive_model_name}' loaded.")
            except Exception as e:
                logger.error(f"Failed to load abstractive pipeline '{self.default_config.hf_abstractive_model_name}': {e}", exc_info=True)
                raise ExternalServiceError(f"Failed to load abstractive model/pipeline '{self.default_config.hf_abstractive_model_name}': {e}") from e

        if nltk is None:
            logger.warning("NLTK library not available. Sentence tokenization for chunking/fallbacks might fail.")
        else:
            try: nltk.data.find('tokenizers/punkt')
            except LookupError: # More specific than nltk.downloader.DownloadError for find()
                logger.info("NLTK 'punkt' not found. Attempting download...")
                try: nltk.download('punkt', quiet=True)
                except Exception as e: logger.error(f"Failed to download 'punkt': {e}", exc_info=True)
            except Exception as e: logger.error(f"Error checking NLTK 'punkt': {e}", exc_info=True)


    def _preprocess_text(self, text: str) -> str:
        processed_text = re.sub(r'<[^>]+>', '', text)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        return processed_text

    def _chunk_text(self, text: str, config: SummarizerConfig) -> List[str]:
        if not self.tokenizer: raise ConfigurationError("Tokenizer required for chunking but not available.")
        if nltk is None: raise ConfigurationError("NLTK required for sentence tokenization in chunking but not available.")
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError as e: # 'punkt' not found after attempt/check in __init__
            raise ConfigurationError(f"NLTK 'punkt' resource unavailable for sentence tokenization: {e}") from e
        except Exception as e: raise SummarizationError(f"Sentence tokenization failed: {e}") from e
        if not sentences: return []

        chunks, current_chunk_sentences, current_chunk_token_count = [], [], 0
        sentence_overlap = config.chunk_overlap if isinstance(config.chunk_overlap, int) and config.chunk_overlap >=0 else 0

        for sentence in sentences:
            try: sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            except Exception as e: raise SummarizationError(f"Tokenizer encode failed for sentence '{sentence[:30]}...': {e}") from e

            if current_chunk_token_count + sentence_tokens <= config.chunk_size:
                current_chunk_sentences.append(sentence); current_chunk_token_count += sentence_tokens
            else:
                if current_chunk_sentences: chunks.append(" ".join(current_chunk_sentences))
                start_sent_idx = max(0, len(current_chunk_sentences) - sentence_overlap) if sentence_overlap > 0 else 0
                current_chunk_sentences = current_chunk_sentences[start_sent_idx:] + [sentence]
                try: current_chunk_token_count = len(self.tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False))
                except Exception as e: raise SummarizationError(f"Tokenizer encode failed for new chunk: {e}") from e
                if sentence_tokens > config.chunk_size and not new_chunk_start_sentences : # current sentence itself is too big
                    logger.warning(f"Single sentence larger than chunk_size: '{sentence[:50]}...' Will be its own chunk.")
                    if chunks and chunks[-1] != " ".join(current_chunk_sentences[:-1]): # if previous part was added
                        chunks.append(sentence) # Add large sentence as new chunk
                    elif not chunks: # if this is the only sentence
                         chunks.append(sentence)
                    current_chunk_sentences,current_chunk_token_count = [],0 # Reset for next

        if current_chunk_sentences: chunks.append(" ".join(current_chunk_sentences))
        return chunks if chunks else ([text] if not sentences else []) # Fallback for very short texts or single sentence

    def _extractive_summary(self, text: str, config: SummarizerConfig) -> str:
        if not text.strip(): return ""
        if config.extractive_method == "gensim_textrank":
            if gensim_summarize is None: raise ConfigurationError("Gensim library not available for 'gensim_textrank'.")
            try:
                if config.desired_length_type == "percentage":
                    ratio = float(config.desired_length_value); ratio = min(max(0.01, ratio), 1.0)
                    return gensim_summarize(text, ratio=ratio, split=False)
                elif config.desired_length_type == "words":
                    word_count = int(config.desired_length_value); word_count = max(10, word_count) # Min 10 words
                    return gensim_summarize(text, word_count=word_count, split=False)
                else: return gensim_summarize(text, ratio=0.2, split=False) # Default
            except Exception as e: raise SummarizationError(f"Gensim summarization failed: {e}") from e
        else: # Fallback to first N sentences
            logger.warning(f"Unsupported extractive_method '{config.extractive_method}', using first sentence fallback.")
            if nltk is None: raise ConfigurationError("NLTK required for fallback extractive summary.")
            try: sentences = nltk.sent_tokenize(text); return sentences[0] if sentences else ""
            except Exception as e: raise SummarizationError(f"NLTK fallback for extractive summary failed: {e}") from e

    def _abstractive_summary(self, text: str, config: SummarizerConfig) -> str:
        if not self.abstractive_pipeline: raise ConfigurationError("Abstractive pipeline unavailable.")
        if not text.strip(): return ""
        try:
            results = self.abstractive_pipeline(text, max_length=config.max_length_abstractive, min_length=config.min_length_abstractive, do_sample=False, **(config.additional_params or {}))
            if results and isinstance(results, list) and 'summary_text' in results[0]: return results[0]['summary_text']
            raise SummarizationError("Abstractive pipeline returned unexpected output.")
        except Exception as e: raise ExternalServiceError(f"Abstractive pipeline failed: {e}") from e

    def summarize(self, text: str, config: Optional[SummarizerConfig] = None) -> str:
        final_config = config or self.default_config
        if not text.strip(): logger.warning("Empty input to summarize."); return ""
        try:
            processed_text = self._preprocess_text(text)
            # Determine if chunking is needed
            needs_chunking = False
            if self.tokenizer: # Only if tokenizer is available
                try:
                    # Check token count only if tokenizer is confirmed to be working for this text
                    if len(self.tokenizer.encode(processed_text)) > final_config.chunk_size:
                        needs_chunking = True
                except Exception as e_tok: # Catch errors during this specific encode call
                    logger.error(f"Tokenizer error during pre-chunking length check: {e_tok}", exc_info=True)
                    # Decide behavior: attempt full text summarization or raise? For now, log and attempt full.
                    # This might be risky for very long texts.
                    needs_chunking = False # Avoid chunking if unsure about tokenization
            elif len(processed_text) > (final_config.chunk_size * 3): # Heuristic if no tokenizer: avg 3 chars/token
                 logger.warning("Tokenizer unavailable, using character length heuristic for chunking decision.")
                 needs_chunking = True


            text_chunks = self._chunk_text(processed_text, final_config) if needs_chunking else [processed_text]

            chunk_summaries = []
            for chunk in text_chunks:
                if final_config.style == "extractive": chunk_summaries.append(self._extractive_summary(chunk, final_config))
                elif final_config.style == "abstractive": chunk_summaries.append(self._abstractive_summary(chunk, final_config))
                else: raise ConfigurationError(f"Unsupported style: {final_config.style}")

            concatenated_summary = " ".join(s for s in chunk_summaries if s)

            if len(text_chunks) > 1 and self.tokenizer: # Reduce step
                token_count = len(self.tokenizer.encode(concatenated_summary)); threshold = final_config.max_length_abstractive
                if token_count > threshold:
                    logger.info(f"Reducing concatenated summary (tokens: {token_count} > threshold: {threshold}).")
                    # Use a new config for the reduce step, matching the original request's desired length constraints
                    # This re-applies the *original* desired_length_value/type to the concatenated summary.
                    reduce_step_config = dataclasses.replace(final_config)
                    if final_config.style == "abstractive": concatenated_summary = self._abstractive_summary(concatenated_summary, reduce_step_config)
                    elif final_config.style == "extractive": concatenated_summary = self._extractive_summary(concatenated_summary, reduce_step_config)

            return self._postprocess_summary(concatenated_summary, final_config)
        except (ConfigurationError, SummarizationError, ExternalServiceError) as e:
            logger.error(f"Summarization failed: {type(e).__name__} - {e}", exc_info=True); raise
        except Exception as e:
            logger.error(f"Unexpected summarization error: {e}", exc_info=True)
            raise SummarizationError(f"Unexpected error in summarize: {e}") from e

    def _postprocess_summary(self, summary: str, config: SummarizerConfig) -> str:
        processed_summary = summary.strip()
        if processed_summary and not processed_summary.endswith(('.', '!', '?')): processed_summary += "."
        return processed_summary
