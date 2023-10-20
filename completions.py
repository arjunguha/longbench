import argparse
import json
from typing import List
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import math
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


# Copied from MultiPL-E
class VLLM:
    def __init__(self, name, revision, tokenizer_name=None, num_gpus=1):
        assert revision is None, "TODO: implement revision"
        dtype = "float16"
        if torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        self.model = LLM(
            model=name,
            tokenizer=tokenizer_name,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95,
        )

    def completions(
        self, prompts: List[str], max_tokens: int, temperature: float, top_p, stop
    ):
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(temperature=temperature,
                                top_p=top_p, max_tokens=max_tokens, stop=stop)
        outputs = self.model.generate(prompts, params, use_tqdm=False)
        return [stop_at_stop_token(o.outputs[0].text, stop) for o in outputs]


def batch_inputs(input_data, num_completions, batch_size):
    batch = [ ]
    for item in input_data:
        for i in range(num_completions):
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = [ ]
    if len(batch) > 0:
        yield batch

def prompt_template(entry):
    p = entry["prompt"]
    f = entry["target_function_name"]
    return f"{p}\n\n# A complete test suite for {f}:\ndef test_{f}():\n    assert {f}("

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-completions", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    vllm = VLLM(args.model_name, None, num_gpus=args.num_gpus)

    
    input_data = pd.read_json(args.input, lines=True)
    input_data = input_data[input_data["approx_token_count"] <= args.max_tokens]
    input_data = input_data.to_dict(orient="records")
    batched_inputs = list(batch_inputs(input_data, args.num_completions, args.batch_size))
    output_data = { item["task_id"]: { 
        "task_id": item["task_id"], 
        "target_function": item["target_function"],
        "target_function_name": item["target_function_name"], 
        "approx_token_count": item["approx_token_count"],
        "mutants": item["mutants"],
        "completions": [] } for item in input_data 
    }


    for batch in tqdm(batched_inputs, desc="Batch"):
        prompts = [prompt_template(entry) for entry in batch]
        completions = vllm.completions(
            prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
        )
        for idx, completion in enumerate(completions):
            task_id = batch[idx]["task_id"]
            output_data[task_id]["completions"].append(completion)

    pd.DataFrame(output_data.values()).to_json(args.output, orient="records", lines=True)

if __name__ == "__main__":
    main()
