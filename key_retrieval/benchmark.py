import argparse
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path


# Copied from MultiPL-E
def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.

    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


class Transformers:
    def __init__(self, name, revision, tokenizer_name=None):
        dtype = torch.float16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype, trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or name, revision=revision, padding_side="left", trust_remote_code=True)
        self.tokenizer.pad_token = "<|endoftext|>"

    def completion_tensors(
        self,
        prompts: list,
        max_length: int,
    ):
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt", return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                use_cache=False,
                max_new_tokens=10,
                # max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output

    def decode_single_output(self, output_tensor, prompt):
        detok_hypo_str = self.tokenizer.decode(
            output_tensor, clean_up_tokenization_spaces=False, skip_special_tokens=True,
        )
        return detok_hypo_str[len(prompt):]

    def completions(
        self, prompts: List[str], max_tokens: int
    ):
        prompts = [prompt.strip() for prompt in prompts]
        output_tensors = self.completion_tensors(prompts, max_tokens)
        return [
            stop_at_stop_token(self.decode_single_output(
                output_tensor, prompt),["\n", "<|endoftext|>"])
            for (prompt, output_tensor) in zip(prompts, output_tensors)
        ]


def batch_inputs(input_data, num_completions, batch_size):
    batch = []
    for item in input_data:
        for i in range(num_completions):
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if len(batch) > 0:
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_file", type=Path, required=True)
    parser.add_argument("--results_file", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    args = parser.parse_args()

    model = Transformers(args.model, revision=None)

    input_data = pd.read_json(args.benchmark_file, lines=True)

    input_data = input_data[input_data["target_context_length"] <= args.max_tokens]
    input_data = input_data.to_dict(orient="records")
    batched_inputs = list(batch_inputs(input_data, 1, args.batch_size))
    output_data = [ ]

    for batch in tqdm(batched_inputs, desc="Batch"):
        prompts = [entry["context"] for entry in batch]
        completions = model.completions(prompts, max_tokens=args.max_tokens)
        for (entry, completion) in zip(batch, completions):
            entry["completion"] = completion
            entry["ok"] = entry["key"] == completion.strip()
            output_data.append(entry)
    pd.DataFrame(output_data).to_json(args.results_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
