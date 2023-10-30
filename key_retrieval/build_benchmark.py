"""
Builds a "key retrieval" benchmark as described in the Code Llama paper.

Each benchmark problem is determined by:

- benchmark_length: a benchmark length in tokens
- key_position: a position in the benchmark in percentage of the benchmark length
- key_function_name: the name of a function that is inserted at the key position
- key: A random value that is inserted at the key position

The benchmark is constructed as follows:

We construct a context of length `benchmark_length` tokens, filled with several
Python functions. We insert a function at `key_position` with the shape:

    def `key_function_name`():
        return `key`

We conclude the prompt with:

    assert `key_function_name`() ==

Thus the only valid completion is `key`.

A few other parameters that matter:

- `FUNCTIONS_DATASET_NAME`: The name of the dataset to use for the functions.
   We are using the MultiPL-T dataset of Python functions from The Stack.
- `FUNCTIONS_DATASET_KWARGS`: Keyword arguments to pass to the dataset.
   We are using the training split.
- `TOKENIZER_NAME`: The name of the tokenizer that we use to compute lengths.
    We are using the StarCoderBase tokenizer.
"""

import random
import argparse
from pathlib import Path
from tqdm import tqdm
import datasets
from typing import Generator, Tuple
from transformers import AutoTokenizer

FUNCTIONS_DATASET_NAME = "nuprl/stack-dedup-python-testgen-starcoder-filter-inferred-v2"
FUNCTIONS_DATASET_KWARGS = {"split": "train"}
TOKENIZER_NAME = "bigcode/starcoderbase-1b"
BENCHMARK_LENGTHS = [4096, 8192, 32768, 65536, 65536 * 2]
KEY_POSITIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Hopefully nobody has used this function name before.
KEY_FUNCTION_NAME = "get_starcoder_hotdog_flavor"
KEY = '"59123"'


def generate_functions(
    tokenizer: AutoTokenizer,
) -> Generator[Tuple[str, int], None, None]:
    """
    Generates functions from the functions dataset. Produces their text and
    their length in tokens.
    """
    functions_dataset = datasets.load_dataset(
        FUNCTIONS_DATASET_NAME, **FUNCTIONS_DATASET_KWARGS
    )
    functions_dataset = functions_dataset.shuffle()
    for example in functions_dataset:
        text = example["content"] + "\n\n"
        token_length = len(tokenizer.encode(text))
        # We will end up concatenating these functions, and retokenizing the
        # concatenated string can split tokens.
        yield (text, token_length + 1)


def build_benchmark_item(
    tokenizer: AutoTokenizer,
    function_generator: Generator[Tuple[str, int], None, None],
    benchmark_length: int,
    key_position: float,
    key_function_name: str,
    key: str,
):
    context_chunks = []
    context_length = 0

    prompt_suffix = f"assert {key_function_name}() =="
    prompt_suffix_len = len(tokenizer.encode(prompt_suffix))
    expected_completion_len = len(tokenizer.encode(" " + key))

    # Add functions to the context until we reach the key position
    (function, function_length) = next(function_generator)
    while (
        context_length + function_length + prompt_suffix_len + expected_completion_len
        < benchmark_length * key_position
    ):
        context_chunks.append(function)
        context_length += function_length
        (function, function_length) = next(function_generator)

    # Add the key function
    key_function = f"def {key_function_name}():\n    return {key}\n\n"
    key_function_length = len(tokenizer.encode(key_function))
    context_chunks.append(key_function)
    context_length += key_function_length

    # Add functions to the context until we reach the end
    while (
        context_length + function_length + prompt_suffix_len + expected_completion_len
        < benchmark_length
    ):
        context_chunks.append(function)
        context_length += function_length
        (function, function_length) = next(function_generator)

    # Add the prompt suffix
    context_chunks.append(prompt_suffix)
    context_length += prompt_suffix_len

    context = "".join(context_chunks)
    assert (
        len(tokenizer.encode(context)) <= benchmark_length
    ), f"Context length {len(tokenizer.encode(context))} exceeds benchmark length {benchmark_length}"
    return {
        "context": context,
        "key": key,
        "context_length": context_length,
        "target_context_length": benchmark_length,
    }


def build_benchmark(output: Path):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    function_generator = generate_functions(tokenizer)

    items = []
    for benchmark_length in tqdm(BENCHMARK_LENGTHS, desc="Benchmark length"):
        for key_position in KEY_POSITIONS:
            item = build_benchmark_item(
                tokenizer=tokenizer,
                function_generator=function_generator,
                benchmark_length=benchmark_length,
                key_position=key_position,
                key_function_name=KEY_FUNCTION_NAME,
                key=KEY,
            )
            items.append(item)

    benchmark_ds = datasets.Dataset.from_list(items)
    benchmark_ds.to_json(output, orient="records", lines=True)
    lengths = benchmark_ds.to_pandas()[["context_length", "target_context_length"]]
    print(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)

    random.seed(42)

    args = parser.parse_args()
    args = vars(args)

    build_benchmark(**args)


if __name__ == "__main__":
    main()
