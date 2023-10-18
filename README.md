1. Use builder.pynb to create the benchmark. The script relies on randomness,
   so commit the generated benchmark to the repository.

2. Use completions.py to geenerate completions.

   ```
   python3 completions.py --input test2code_long_context.jsonl --output completions.jsonl --model-name /home/arjun/models/starcoderbase-1b --batch-size 50 --num-completions 20
   ```

3. Use executions.py to execution completions.

    ```
    python3 executions.py --input completions.jsonl --output executions.jsonl
    ```

4. Use pass1.ipynb to look at the results.