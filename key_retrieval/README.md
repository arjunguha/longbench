With the model checked out:

```
python3 benchmark.py \
  --model ../../models/sc2-1b-repo-level-ablations-top-level-depth-first_8k_64k \
  --benchmark_file benchmark.jsonl \
  --batch_size 1 \
  --max_tokens 65536 \
  --results_file results_4gpu_sc2-1b-repo-level-ablations-top-level-depth-first_8k_64k.jsonl
```

