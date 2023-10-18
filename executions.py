
import argparse
from typing import List
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import bounded_subprocess
import tempfile
import json

def run(py_program: str):
    with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
        f.write(py_program)
        f.flush()
        result = bounded_subprocess.run(["python3", f.name], timeout_seconds=10)
        return { "program": py_program, "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code }


def process_problem(executor, problem: dict):
    prefix = problem["canonical_prompt"]
    suffix = "\n\n" + problem["target_tests"]
    executions = executor.map(lambda c: run(prefix + c + suffix), problem["completions"])
    return { "benchmark_name": problem["benchmark_name"], "executions": list(executions) }
    

def main_with_args(input: Path, output: Path):
    problems = pd.read_json(input, lines=True).to_dict(orient="records")
    with ThreadPoolExecutor() as executor:
        with output.open("w") as f:
            for item in tqdm(executor.map(lambda p: process_problem(executor, p), problems), total=len(problems), desc="Problem"):
                f.write(json.dumps(item) + "\n")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="a completions.jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="an executions.jsonl file")
    args = parser.parse_args()
    main_with_args(**vars(args))


if __name__ == "__main__":
    main()












