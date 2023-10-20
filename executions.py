
import argparse
from typing import List
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import bounded_subprocess
import tempfile
import json


def run(py_program: str, mutants: List[str]):
    with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
        f.write(py_program)
        f.flush()
        result = bounded_subprocess.run(
            ["python3", f.name], timeout_seconds=30)
        final = {"program": py_program, "stdout": result.stdout,
                 "stderr": result.stderr, "exit_code": result.exit_code, "mutants": []}

    for m in mutants:
        with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
            f.write(m)
            f.flush()
            result = bounded_subprocess.run(
                ["python3", f.name], timeout_seconds=30)
            final["mutants"].append({"program": m, "stdout": result.stdout,
                                     "stderr": result.stderr, "exit_code": result.exit_code})

    return final

def clip_completion(completion: str):
    lines = completion.split("\n")
    print(f"Clipped completion to {len(lines)} -> 10 lines")
    lines = lines[:10]
    
    return "\n".join(lines)

def process_problem(executor, problem: dict):
    # TODO(arjun): Make the prefix and suffix configurable and shared with completions.py
    prefix_suf = f"\n\ndef test_suite():\n    assert {problem['target_function_name']}("
    prefix = problem["target_function"] + prefix_suf
    suffix = "\n\ntest_suite()"
    completions = [ clip_completion(c) for c in problem["completions"] ]
    executions = executor.map(lambda c: run(
        prefix + c + suffix, list(map(lambda m: m + "\n" + prefix_suf + c + suffix, problem["mutants"]))), completions)
    return {"task_id": problem["task_id"], "executions": list(executions)}


def main_with_args(input: Path, output: Path):
    problems = pd.read_json(input, lines=True).to_dict(orient="records")
    with ThreadPoolExecutor() as executor:
        with output.open("w") as f:
            for item in tqdm(map(lambda p: process_problem(executor, p), problems), total=len(problems), desc="Problem"):
                f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="a completions.jsonl file")
    parser.add_argument("--output", type=Path, required=True,
                        help="an executions.jsonl file")
    args = parser.parse_args()
    main_with_args(**vars(args))


if __name__ == "__main__":
    main()
