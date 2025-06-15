# ContextKernel Utility Helpers Module (helpers.py)

# 1. Purpose of the file/module:
# This module is a collection of miscellaneous utility functions and helper classes
# that provide common, reusable logic across different parts of the ContextKernel
# application. These helpers are typically small, focused, and not specific to any
# single component of the kernel but rather offer general-purpose functionalities
# like data manipulation, string processing, ID generation, etc.

# 2. Core Logic:
# The logic within this module will vary depending on the specific helper functions
# included. Examples might include:
#   - Unique ID Generation: Creating universally unique identifiers (UUIDs) or other
#     forms of unique IDs for documents, sessions, or context entries.
#   - Data Chunking: Splitting large lists or texts into smaller, manageable chunks,
#     often useful for processing data in batches (e.g., before sending to an LLM
#     that has token limits).
#   - String Manipulation: Common text processing tasks like sanitization, normalization,
#     or specific formatting.
#   - Dictionary/List Operations: Helper functions for common operations on collections
#     that are not readily available or are cumbersome with standard Python.
#   - Timestamp Utilities: Functions for generating or formatting timestamps.

# 3. Key Inputs/Outputs:
#   - Inputs: Vary by function (e.g., data to be processed, parameters for generation).
#   - Outputs: Vary by function (e.g., a generated ID, a list of chunks, a processed string).

# 4. Dependencies/Needs:
#   - Standard Library: `uuid` for UUID generation, `math` for calculations (e.g., for chunking),
#     `datetime` for timestamp utilities.
#   - (Potentially) Third-party libraries for more complex tasks if needed, though helpers
#     are often kept simple and reliant on the standard library.

# 5. Real-world solutions/enhancements:

#   ID Generation Strategies:
#   - `uuid.uuid4()`: Generates random UUIDs. Most common for general uniqueness.
#   - `uuid.uuid1()`: Generates UUIDs based on host ID and current time (can reveal MAC address).
#   - `uuid.uuid5(namespace, name)`: Generates a UUID based on the SHA-1 hash of a namespace
#     identifier (itself a UUID) and a name. Useful for reproducible UUIDs from specific inputs.
#   - Short IDs: Libraries like `shortuuid` can generate shorter, human-readable unique IDs
#     (e.g., YouTube-like IDs). (https://pypi.org/project/shortuuid/)
#   - Nanoid: Another library for generating short, unique, URL-friendly IDs.
#     (https://pypi.org/project/nanoid/)

#   Data Chunking Techniques:
#   - Fixed-size chunking: Simply dividing a list into sublists of `n` items.
#   - Overlapping chunks (for text): When chunking text for LLMs, it's often useful
#     to have some overlap between chunks to maintain context. Libraries like
#     `Langchain` (text splitters) or `Spacy` (sentence segmentation) can help with
#     more sophisticated text chunking.
#   - Semantic Chunking: More advanced techniques that try to split text based on
#     semantic meaning rather than fixed sizes.

#   Decorator Utilities:
#   - Python's `functools` module (`@wraps`, `@lru_cache`) provides good building blocks.
#   - Custom decorators for logging, timing, retry logic, etc., can be placed here
#     if they are general enough.

#   Considerations for Helper Functions:
#   - Keep them pure: Helper functions should ideally be pure (i.e., their output
#     depends only on their input, and they have no side effects). This makes them
#     easier to test and reason about.
#   - No Application State: They generally should not depend on the application's
#     global state or configuration directly. If they need configuration, it should
#     be passed in as an argument.
#   - Well-Documented: Clear docstrings explaining what each function does, its
#     parameters, and what it returns.
#   - Type Hinted: Use Python type hints for clarity and to aid static analysis.

import uuid
from typing import List, TypeVar, Any, Generator

# Generic type variable for the chunk_list function
T = TypeVar('T')

def generate_unique_id(prefix: str = "id") -> str:
    """
    Generates a universally unique identifier (UUID4) with an optional prefix.

    Args:
        prefix: An optional string prefix for the generated ID.
                Defaults to "id".

    Returns:
        A string representing the unique ID (e.g., "prefix_uuid_string").
    """
    unique_val = uuid.uuid4()
    if prefix:
        return f"{prefix}_{unique_val}"
    return str(unique_val)


def chunk_list(data: List[T], chunk_size: int) -> Generator[List[T], None, None]:
    """
    Splits a list into smaller chunks of a specified size.

    Args:
        data: The list to be chunked.
        chunk_size: The maximum size of each chunk.
                    Must be a positive integer.

    Yields:
        A generator that produces lists, where each list is a chunk
        of the original data. The last chunk may be smaller than chunk_size.

    Raises:
        ValueError: If chunk_size is not a positive integer.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not data:
        return # Yield nothing for an empty list

    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


if __name__ == "__main__":
    print("--- Demonstrating helper functions ---")

    # Demonstrate generate_unique_id
    print("\n1. generate_unique_id():")
    id1 = generate_unique_id()
    id2 = generate_unique_id("user")
    id3 = generate_unique_id("") # Test with empty prefix
    print(f"  Default prefix: {id1}")
    print(f"  'user' prefix: {id2}")
    print(f"  Empty prefix: {id3}")
    assert "id_" in id1
    assert "user_" in id2
    assert "_" not in id3 # uuid4 string itself does not contain underscores typically
    assert len(id1) > 36 # prefix + _ + uuid
    assert len(id2) > 36
    assert len(id3) == 36 # length of uuid4 string

    # Demonstrate chunk_list
    print("\n2. chunk_list():")
    my_list_numbers = list(range(10)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    my_list_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    print(f"  Chunking {my_list_numbers} into chunks of 3:")
    for i, chunk in enumerate(chunk_list(my_list_numbers, 3)):
        print(f"    Chunk {i+1}: {chunk}")
        if i == 0: assert chunk == [0, 1, 2]
        if i == 1: assert chunk == [3, 4, 5]
        if i == 2: assert chunk == [6, 7, 8]
        if i == 3: assert chunk == [9]

    print(f"\n  Chunking {my_list_letters} into chunks of 2:")
    for i, chunk in enumerate(chunk_list(my_list_letters, 2)):
        print(f"    Chunk {i+1}: {chunk}")
        if i == 0: assert chunk == ['a', 'b']
        if i == 1: assert chunk == ['c', 'd']
        if i == 2: assert chunk == ['e', 'f']
        if i == 3: assert chunk == ['g']

    print("\n  Chunking an empty list into chunks of 5:")
    empty_list_chunks = list(chunk_list([], 5))
    print(f"    Result: {empty_list_chunks}")
    assert empty_list_chunks == []

    print("\n  Chunking a list with chunk_size larger than list length:")
    single_chunk = list(chunk_list([1, 2], 5))
    print(f"    Result for chunk_list([1, 2], 5): {single_chunk}")
    assert single_chunk == [[1, 2]]

    print("\n  Testing invalid chunk_size:")
    try:
        list(chunk_list([1, 2, 3], 0))
    except ValueError as e:
        print(f"    Caught expected error for chunk_size=0: {e}")
        assert "positive integer" in str(e)

    try:
        list(chunk_list([1, 2, 3], -1))
    except ValueError as e:
        print(f"    Caught expected error for chunk_size=-1: {e}")
        assert "positive integer" in str(e)

    try:
        list(chunk_list([1, 2, 3], 1.5)) # type: ignore
    except ValueError as e:
        print(f"    Caught expected error for chunk_size=1.5: {e}")
        assert "positive integer" in str(e)

    print("\n--- All demonstrations complete ---")

# End of helpers.py
# Ensure this replaces the old "helpers.py loaded" print and comments.
