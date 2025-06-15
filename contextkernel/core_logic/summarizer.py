# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Summarizer Module (summarizer.py) - The "Mental Note-Taker"

# 1. Purpose of the file/module:
# This module acts as the "Mental Note-Taker" for the ContextKernel. Its primary
# function is to convert raw input text or larger documents into concise, distilled
# summaries or chunked key insights. These summaries can then be used by various
# parts of the system. For example:
#   - The `LLMListener` might use this module to summarize information before
#     storing it in long-term memory (LTM) or the knowledge graph.
#   - The `ContextAgent` might use it to quickly grasp the essence of a long piece
#     of retrieved context before making a decision.
#   - It can help in managing context window limitations for LLMs by providing
#     shorter versions of texts.

# 2. Core Logic:
# The core logic of the Summarizer typically involves:
#   - Receiving Input Text: Takes raw text strings, documents, or potentially URLs
#     pointing to text content.
#   - Pre-processing (Optional): Cleaning the text (e.g., removing HTML tags, irrelevant
#     characters, standardizing whitespace).
#   - Text Chunking (if necessary): If the input text is too long for the chosen
#     summarization model's context window, it needs to be split into smaller, manageable
#     chunks. This might be done by paragraphs, sentences, or fixed token counts.
#   - Summarization Technique Application:
#     - Extractive Summarization: Selecting important sentences or phrases directly
#       from the original text.
#     - Abstractive Summarization: Generating new sentences that capture the essence
#       of the text, often using sequence-to-sequence LLMs.
#   - Iterative Summarization (for chunked text): If the text was chunked, each chunk
#     is summarized. Then, these individual summaries might be concatenated, or a
#     further summarization step might be applied to summarize the summaries themselves
#     (a "map-reduce" style approach).
#   - Post-processing (Optional): Minor cleaning of the summary, ensuring coherence,
#     or attempting to verify factual consistency (a very challenging task).
#   - Outputting Summaries: Returning the final summary or list of summary chunks.

# 3. Key Inputs/Outputs:
#   - Inputs:
#     - Raw Text: A string containing the text to be summarized.
#     - Document(s): Could be file paths or objects representing documents.
#     - Configuration for Summarization:
#       - `desired_length` or `length_ratio` (e.g., percentage of original text).
#       - `style` (e.g., "extractive", "abstractive", "bullet_points").
#       - `chunk_size` and `chunk_overlap` (if chunking is used).
#       - Specific model to use.
#   - Outputs:
#     - Summarized Text: A string containing the summary.
#     - List of Summary Chunks: If the input was chunked and processed iteratively,
#       this might be a list of strings, each being a summary of a chunk.
#     - (Potentially) Metadata about the summarization process (e.g., original length,
#       summary length, time taken).

# 4. Dependencies/Needs:
#   - LLM Models: For abstractive summarization (e.g., BART, T5, Pegasus, GPT-family).
#     Access via APIs (OpenAI, Cohere) or local Hugging Face Transformers.
#   - NLP Libraries:
#     - For text processing, tokenization, sentence splitting (e.g., NLTK, spaCy).
#     - For specific extractive methods (e.g., Gensim for TextRank, Sumy).
#   - Tokenizers: Specifically, the tokenizer corresponding to the LLM being used,
#     to accurately count tokens for chunking and context window management.

# 5. Real-world solutions/enhancements:

#   Summarization Libraries & Models:
#   - Hugging Face Transformers:
#     - `summarization` pipeline: Easy-to-use interface.
#       (https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.SummarizationPipeline)
#     - Specific Models:
#       - BART (`facebook/bart-large-cnn`): Good for general abstractive summarization.
#       - T5 (`t5-small`, `t5-base`, `t5-large`): Text-to-text model, versatile.
#       - Pegasus (`google/pegasus-xsum`): Pre-trained on extreme summarization.
#       - LongT5, LED: Models designed to handle longer context windows.
#   - Gensim: `gensim.summarization.summarize` (TextRank algorithm for extractive).
#     (https://radimrehurek.com/gensim/summarization/summariser.html)
#   - Sumy: Library with multiple extractive algorithms (LexRank, Luhn, LSA, TextRank).
#     (https://pypi.org/project/sumy/)
#   - Langchain: Provides tools and chains for summarization, especially useful for
#     handling long documents with strategies like "map_reduce", "refine", "stuff".
#     (https://python.langchain.com/docs/use_cases/summarization)

#   Summarization Techniques:
#   - Extractive Summarization:
#     - Pros: Preserves factual accuracy (as it uses original sentences), faster.
#     - Cons: Can be less coherent, may not capture overall meaning as well as abstractive.
#     - Use Cases: Legal documents, news articles where exact phrasing is important.
#   - Abstractive Summarization:
#     - Pros: More human-like, can be more concise and coherent, better at synthesis.
#     - Cons: Risk of hallucinations or factual inaccuracies, computationally more intensive.
#     - Use Cases: Creating executive summaries, summarizing discussions for quick understanding.

#   Text Chunking Strategies:
#   - By Paragraphs or Sections: Use natural document structure.
#   - By Sentences: Use NLTK (`sent_tokenize`) or spaCy (`doc.sents`) for sentence splitting.
#   - Fixed Token Count: Use a tokenizer (e.g., from Hugging Face) to count tokens and split.
#   - Recursive Character Text Splitter (e.g., in Langchain): Tries to split on a list of characters
#     (e.g., "\n\n", "\n", " ", "") and keeps chunks of similar size.
#   - Overlap: When chunking, it's often good to have some overlap between chunks to ensure
#     context is not lost at the boundaries.

#   Pre-processing:
#   - HTML Removal: Use libraries like `BeautifulSoup` or `html.parser`.
#   - Special Character Removal/Normalization: Using `regex` or custom logic.
#   - Case Normalization (Optional): May or may not be beneficial depending on the model.

#   Post-processing:
#   - Fact Checking: Extremely hard problem, often requires external knowledge bases or human review.
#     Research areas include using LLMs to self-critique or cross-reference with other sources.
#   - Coherence Refinement: Potentially use another LLM call to smooth transitions or improve
#     the flow of concatenated summaries.
#   - Deduplication: Remove redundant sentences or phrases from the summary.

#   Configurable Summary Length/Detail:
#   - Allow users or other system components to specify the desired output length (e.g.,
#     number of words/sentences, percentage of original) or level of detail.
#   - Models like T5 can be prompted for different summary lengths.
"""Core summarization logic for Context Kernel."""

import logging
import dataclasses
import re
import os
import json
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class SummarizerConfig:
    """Configuration for the summarizer.

    Attributes:
        style: The style of summarization (e.g., "extractive", "abstractive").
        desired_length_type: The type of desired length (e.g., "percentage",
            "words", "tokens").
        desired_length_value: The value corresponding to the desired_length_type.
        chunk_size: The size of chunks for token-based chunking.
        chunk_overlap: The overlap between chunks.
        model_name: The specific model to use for summarization.
        llm_api_key_env: The environment variable for the LLM API key.
        additional_params: Additional parameters for the summarization model.
    """
    style: str = "abstractive"
    desired_length_type: str = "percentage"
    desired_length_value: float = 0.2
    chunk_size: int = 1024
    chunk_overlap: int = 50
    model_name: Optional[str] = None
    llm_api_key_env: Optional[str] = "OPENAI_API_KEY"
    additional_params: Optional[Dict[str, Any]] = None


class Summarizer:
    """Handles text summarization using various strategies."""

    def __init__(self, default_config: Optional[SummarizerConfig] = None):
        """Initializes the Summarizer.

        Args:
            default_config: An optional default configuration for summarization.
                If not provided, a default SummarizerConfig instance will be used.
        """
        if default_config:
            self.default_config = default_config
        else:
            self.default_config = SummarizerConfig()

    def _preprocess_text(self, text: str) -> str:
        """Performs pre-processing on the text, like HTML cleaning and whitespace normalization.

        Args:
            text: The text to preprocess.

        Returns:
            The preprocessed text.
        """
        logger.info("Starting text pre-processing...")

        # Basic HTML tag removal
        # For more robust HTML parsing, a library like BeautifulSoup would be recommended.
        processed_text = re.sub(r'<[^>]+>', '', text)

        # Whitespace normalization (replace multiple whitespaces with a single space)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        logger.info(f"Text after pre-processing (first 100 chars): {processed_text[:100]}")
        return processed_text

    def _chunk_text(self, text: str, config: SummarizerConfig) -> List[str]:
        """Chunks the text into smaller pieces based on the configuration.

        Args:
            text: The text to chunk.
            config: The summarizer configuration containing chunk_size and chunk_overlap.

        Returns:
            A list of text chunks.
        """
        logger.info(f"Starting text chunking with chunk_size={config.chunk_size} and chunk_overlap={config.chunk_overlap}...")
        chunks = []
        start_index = 0
        text_len = len(text)

        # For more advanced chunking, consider:
        # - Tokenizers (e.g., from Hugging Face) for accurate token-based chunking.
        # - NLP libraries (NLTK, spaCy) for sentence-based or paragraph-based chunking.

        while start_index < text_len:
            end_index = start_index + config.chunk_size
            chunk = text[start_index:end_index]
            chunks.append(chunk)

            next_start_index = end_index - config.chunk_overlap
            if next_start_index <= start_index: # Avoid infinite loops if overlap is too large or chunk_size too small
                logger.warning("Chunk overlap is too large or chunk_size too small, advancing by chunk_size to prevent infinite loop.")
                next_start_index = start_index + config.chunk_size

            start_index = next_start_index
            if start_index >= text_len and chunks[-1] != text[text_len-config.chunk_size if text_len > config.chunk_size else 0:]: # Ensure last part is captured if loop condition makes it exit early
                 # This condition is a bit complex, aims to add last chunk if not fully covered.
                 # A simpler way for the last chunk is to ensure the loop runs one more time if start_index was < text_len before update.
                 # The primary loop condition `while start_index < text_len` handles most cases.
                 # The last chunk might be smaller than chunk_size, which is implicitly handled by Python's slicing.
                 pass # Python slicing handles the last chunk correctly.

        # Ensure the very last part of the text is included if it's shorter than chunk_size and loop terminates
        # This is generally handled by the loop logic, but good to be mindful.
        # If, after the loop, the last chunk doesn't reach the end of the text,
        # this could be a point of refinement. However, the current `text[start_index:end_index]`
        # and `start_index = end_index - config.chunk_overlap` should cover it.

        logger.info(f"Created {len(chunks)} chunks.")
        return chunks

    def _extractive_summary(self, text: str, config: SummarizerConfig) -> str:
        """Generates a naive extractive summary of the text.

        This is a placeholder and should be replaced with more sophisticated
        extractive summarization techniques.

        Args:
            text: The text to summarize.
            config: The summarizer configuration.

        Returns:
            A naive extractive summary.
        """
        logger.info("Applying naive extractive summarization...")

        calculated_length = 0
        if config.desired_length_type == "percentage":
            calculated_length = int(len(text) * config.desired_length_value)
        elif config.desired_length_type in ["words", "tokens"]:
            # For this naive approach, treat words/tokens as characters.
            # Proper word/token counting would require a tokenizer.
            calculated_length = int(config.desired_length_value)
            logger.warning("Naive extractive summary is treating 'words' or 'tokens' as character count.")
        else:
            logger.warning(f"Unsupported desired_length_type for extractive summary: {config.desired_length_type}. Defaulting to 20% of text.")
            calculated_length = int(len(text) * 0.2) # Default to 20% if type is unknown

        # Ensure calculated length is not greater than the text length
        calculated_length = min(calculated_length, len(text))

        summary = text[:calculated_length]

        # Comment: This is a very basic placeholder.
        # Real extractive summarization would use techniques like:
        # - TextRank or LexRank algorithms (e.g., via gensim.summarization or sumy).
        # - Selecting important sentences based on scoring (e.g., TF-IDF, embeddings)
        #   after sentence splitting with NLTK/spaCy.

        logger.info(f"Generated naive extractive summary (first 100 chars): {summary[:100]}")
        return summary

    def _abstractive_summary(self, text: str, config: SummarizerConfig) -> str:
        """Generates a stubbed abstractive summary of the text.

        This simulates an API call to an LLM.

        Args:
            text: The text to summarize.
            config: The summarizer configuration.

        Returns:
            A stubbed abstractive summary.
        """
        logger.info("Applying stubbed abstractive summarization...")

        # Simulate API call to an LLM
        logger.info("Simulating API call to an LLM for abstractive summarization.")
        api_key = os.getenv(config.llm_api_key_env if config.llm_api_key_env else "OPENAI_API_KEY") # Default env var if not set
        api_key_found = bool(api_key)
        logger.info(f"API key from env var '{config.llm_api_key_env or "OPENAI_API_KEY"}' found: {api_key_found}")

        mock_payload = {
            "text_to_summarize": text,
            "model": config.model_name or "gpt-3.5-turbo-instruct", # Example default
            "summary_length_type": config.desired_length_type,
            "summary_length_value": config.desired_length_value,
            "additional_params": config.additional_params or {},
        }
        try:
            logger.info(f"Mock LLM API payload: {json.dumps(mock_payload, indent=2, default=str)}")
        except TypeError: # Handle cases where text might not be directly serializable if it were an object
             logger.info(f"Mock LLM API payload (text excerpt): {text[:200]}...")


        # Comment: Real integration would involve:
        # - Using libraries like `openai` (e.g., client.chat.completions.create(...))
        #   or `requests` for HTTP calls to LLM APIs.
        # - For local models, using Hugging Face `transformers.pipeline("summarization", model=config.model_name)`
        #   e.g., `summarizer = pipeline("summarization", model=config.model_name or "facebook/bart-large-cnn")`
        #   `result = summarizer(text, max_length=..., min_length=..., do_sample=False)`

        summary = (
            f"Abstractive summary of '{text[:50]}...' (simulated using model: "
            f"{config.model_name or 'default_LLM'}). API Key ({config.llm_api_key_env or "OPENAI_API_KEY"}) found: {api_key_found}."
        )

        logger.info(f"Generated stubbed abstractive summary: {summary}")
        return summary

    def summarize(self, text: str, config: Optional[SummarizerConfig] = None) -> str:
        """Summarizes the input text based on the provided or default configuration.

        Args:
            text: The text to summarize.
            config: An optional specific configuration for this summarization task.
                If None, the default_config of the Summarizer instance is used.

        Returns:
            The summarized text (currently a placeholder).
        """
        final_config = config if config else self.default_config
        try:
            logger.info(f"Received original text for summarization (first 100 chars): {text[:100]}")
            logger.info(f"Using configuration: {final_config}")
            logger.info("Starting summarization process...")

            # 1. Pre-processing (e.g., cleaning, normalization)
            processed_text = self._preprocess_text(text)

            # 2. Chunking (if necessary, based on text length and model limits)
            text_chunks: List[str]
            if len(processed_text) > final_config.chunk_size:
                text_chunks = self._chunk_text(processed_text, final_config)
                logger.info(f"Text was chunked into {len(text_chunks)} chunks.")
            else:
                text_chunks = [processed_text]
                logger.info("Text is shorter than chunk_size, no chunking applied.")

            # 3. Strategy Selection & Application (e.g., extractive, abstractive)
            chunk_summaries = []
            total_chunks = len(text_chunks)
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1} of {total_chunks}...")
                if final_config.style == "extractive":
                    summary_chunk = self._extractive_summary(chunk, final_config)
                elif final_config.style == "abstractive":
                    summary_chunk = self._abstractive_summary(chunk, final_config)
                else:
                    logger.warning(f"Unsupported summarization style: {final_config.style}. Returning original chunk.")
                    summary_chunk = f"Unknown style: {final_config.style}. Original chunk: {chunk[:100]}..."
                chunk_summaries.append(summary_chunk)

            # 4. Combine summaries (initial step)
            final_summary = " ".join(chunk_summaries)
            logger.info(f"Concatenated summary (first 100 chars): {final_summary[:100]}")

            # TODO: Implement 'reduce' step for map-reduce summarization.
            # If multiple chunks were summarized, the concatenated summaries might be too long or redundant.
            # A further summarization pass could be applied here on 'final_summary',
            # potentially using a different configuration or model designed for summarizing summaries.

            # 5. Post-processing (e.g., final cleaning, length adjustment)
            final_summary = self._postprocess_summary(final_summary, final_config)

            logger.info(f"Summarization complete. Returning post-processed summary (first 100 chars): {final_summary[:100]}")
            return final_summary
        except Exception as e:
            logger.error(f"Error during summarization: {type(e).__name__} - {e}", exc_info=True)
            # exc_info=True in logger.error will include stack trace in logs
            return f"Summarization failed: {type(e).__name__} - {e}"

    def _postprocess_summary(self, summary: str, config: SummarizerConfig) -> str:
        """Performs basic post-processing on the generated summary.

        Args:
            summary: The summary string to post-process.
            config: The summarizer configuration (currently unused in basic version).

        Returns:
            The post-processed summary string.
        """
        logger.info("Starting summary post-processing...")

        processed_summary = summary.strip()

        if processed_summary and not processed_summary.endswith(('.', '!', '?')):
            processed_summary += "."

        # Advanced Post-processing Ideas (add as comments):
        # - Coherence Refinement: Could use another LLM call to improve flow.
        # - Deduplication: Remove redundant sentences/phrases, especially after combining chunks.
        # - Fact Checking: Placeholder for potential future fact-checking mechanisms (very complex).
        # - Length Adherence: More strictly enforce desired_length if current summary deviates significantly
        #   (though this is also a target for the summarization models themselves).

        logger.info(f"Summary after post-processing (first 100 chars): {processed_summary[:100]}")
        return processed_summary
