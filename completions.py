import argparse
from typing import List
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


# Copied from MultiPL-E
class VLLM:
    def __init__(self, name, revision, tokenizer_name=None, num_gpus=1):
        from vllm import LLM
        dtype = "float16"
        if torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        self.model = LLM(
            model=name,
            tokenizer=tokenizer_name,
            revision=revision,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95,
        )

    def completions(
        self, prompts: List[str], max_tokens: int, temperature: float, top_p, stop, do_sample=True
    ):
        from vllm import SamplingParams
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(temperature=temperature,
                                top_p=top_p, max_tokens=max_tokens, stop=stop)
        outputs = self.model.generate(prompts, params, use_tqdm=False)
        return [stop_at_stop_token(o.outputs[0].text, stop) for o in outputs]


class Transformers:
    def __init__(self, name, revision, tokenizer_name=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
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
        temperature: float,
        top_p: float,
        do_sample=True,
    ):
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt", return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=do_sample,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_length=inputs["input_ids"].shape[1] + max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output

    def decode_single_output(self, output_tensor, prompt):
        detok_hypo_str = self.tokenizer.decode(
            output_tensor, clean_up_tokenization_spaces=False, skip_special_tokens=True,
        )
        return detok_hypo_str[len(prompt):]

    def completions(
        self, prompts: List[str], max_tokens: int, temperature: float, top_p, stop, do_sample=True
    ):
        prompts = [prompt.strip() for prompt in prompts]
        output_tensors = self.completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
            do_sample=do_sample,
        )
        return [
            stop_at_stop_token(self.decode_single_output(
                output_tensor, prompt), stop + ["<|endoftext|>"])
            for (prompt, output_tensor) in zip(prompts, output_tensors)
        ]

class DeepSpeed:
    def __init__(self, name, revision, tokenizer_name=None, num_gpus=1):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import deepspeed
        dtype = torch.float16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype, trust_remote_code=True).cuda()
        self.model = deepspeed.init_inference(self.model, world_size=num_gpus, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or name, revision=revision, padding_side="left", trust_remote_code=True)
        self.tokenizer.pad_token = "<|endoftext|>"

    def completion_tensors(
        self,
        prompts: list,
        max_length: int,
        temperature: float,
        top_p: float,
        do_sample=True,
    ):
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt", return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=do_sample,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_length=inputs["input_ids"].shape[1] + max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output

    def decode_single_output(self, output_tensor, prompt):
        detok_hypo_str = self.tokenizer.decode(
            output_tensor, clean_up_tokenization_spaces=False, skip_special_tokens=True,
        )
        return detok_hypo_str[len(prompt):]

    def completions(
        self, prompts: List[str], max_tokens: int, temperature: float, top_p, stop, do_sample=True
    ):
        prompts = [prompt.strip() for prompt in prompts]
        output_tensors = self.completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
            do_sample=do_sample,
        )
        return [
            stop_at_stop_token(self.decode_single_output(
                output_tensor, prompt), stop + ["<|endoftext|>"])
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
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dont_sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    if args.engine == "vllm":
        engine = VLLM(args.model_name, args.revision, num_gpus=args.num_gpus)
    elif args.engine == "transformers":
        engine = Transformers(args.model_name, args.revision)
    elif args.engine == "deepspeed":
        engine = DeepSpeed(args.model_name, args.revision, num_gpus=args.num_gpus)
    else:
        raise ValueError(f"Unknown engine: {args.engine}")

    input_data = pd.read_json(args.input, lines=True)
    input_data = input_data[input_data["approx_token_count"]
                            <= args.max_tokens]
    input_data = input_data.to_dict(orient="records")
    batched_inputs = list(batch_inputs(
        input_data, args.num_completions, args.batch_size))
    output_data = {item["task_id"]: {
        "task_id": item["task_id"],
        "target_function": item["target_function"],
        "target_function_name": item["target_function_name"],
        "approx_token_count": item["approx_token_count"],
        "mutants": item["mutants"],
        "completions": []} for item in input_data
    }

    for batch in tqdm(batched_inputs, desc="Batch"):
        prompts = [prompt_template(entry) for entry in batch]
        completions = engine.completions(
            prompts,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.dont_sample,
            stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
        )
        for idx, completion in enumerate(completions):
            task_id = batch[idx]["task_id"]
            output_data[task_id]["completions"].append(completion)

    pd.DataFrame(output_data.values()).to_json(
        args.output, orient="records", lines=True)


if __name__ == "__main__":
    main()
