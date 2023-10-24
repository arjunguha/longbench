

def mutant_catch_rate(execution):
    if execution["exit_code"] != 0:
        return 0
    mutants = execution["mutants"]
    caught_mutants = sum(1 for mutant in mutants if mutant["exit_code"] != 0)
    return caught_mutants / len(mutants)


def mean_mutant_catch_rate(executions):
    n = len(executions)
    if n == 0:
        return 0
    return sum(mutant_catch_rate(execution) for execution in executions) / n


def test_suite_success_rate(executions):
    n = len(executions)
    if n == 0:
        return 0
    return sum(1 for execution in executions if execution["exit_code"] == 0) / n



def mean(lst):
    return sum(lst) / len(lst)


def get_problem_kind(task_id):
    # returns a tuple (len, kind)
    # example: "LongBench_HumanEval/13_0_first half" -> (0, "first half")
    # example: "LongBench_HumanEval/13_8000_second half" -> (8000, "second half")

    parts = task_id.split("_")
    assert len(parts) >= 2
    assert parts[-1] in ["first half", "second half"]
    assert parts[-2].isdigit()
    return (int(parts[-2]), parts[-1])


def bin_problems(task_ids):
    bins = {}
    for task_id in task_ids:
        kind = get_problem_kind(task_id)
        if kind not in bins:
            bins[kind] = []
        bins[kind].append(task_id)

    return bins


if __name__ == "__main__":
    import argparse
    import datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file", required=True)
    args = parser.parse_args()

    ds = datasets.load_dataset("json", data_files=args.input)["train"]
    bins = bin_problems(ds["task_id"])
    buf = "Length,Kind,Test Suite Success Rate,Mean Mutant Catch Rate\n"
    for (l, kind), b in bins.items():
        exs = ds.filter(lambda x: x["task_id"] in b)
        avg_test_suite_success_rate = mean(
            list(map(test_suite_success_rate, exs["executions"])))
        avg_mean_mutant_catch_rate = mean(
            list(map(mean_mutant_catch_rate, exs["executions"])))
        buf += f"{l},{kind},{avg_test_suite_success_rate},{avg_mean_mutant_catch_rate}\n"

    print(buf)
