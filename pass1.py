

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
    parser.add_argument(
        "--latex", help="output latex table", action="store_true")
    args = parser.parse_args()

    ds = datasets.load_dataset("json", data_files=args.input)["train"]
    bins = bin_problems(ds["task_id"])
    #  \multicolumn{5}{c}{repo\_context\_depth\_first\_8k\_32k} \\ \midrule
    #  & 0 & first  & 0.28 & 0.2225 \\
    #  & 0 & second  & 0.28 & 0.25 \\
    #  & 8000 & first  & 0.14 & 0.1233 \\
    #  & 8000 & second  & 0.36 & 0.3183 \\
    #  & 16000 & first & 0.22 & 0.16 \\
    #  & 16000 & second  & 0.14 & 0.115 \\
    #  & 32000 & first  & 0.1 & 0.1 \\
    #  & 32000 & second & 0.2 & 0.1667 \\ \midrule
    if args.latex:
        name = args.input.split("/")[-1].split(".")[0].replace("_", "\\_")
        buf = "\\multicolumn{5}{c}{" + name + "} \\\\ \\midrule\n"
    else:
        buf = "Length,Kind,Test Suite Success Rate,Mean Mutant Catch Rate\n"

    for (l, kind), b in bins.items():
        exs = ds.filter(lambda x: x["task_id"] in b)
        avg_test_suite_success_rate = mean(
            list(map(test_suite_success_rate, exs["executions"])))
        avg_mean_mutant_catch_rate = mean(
            list(map(mean_mutant_catch_rate, exs["executions"])))
        if args.latex:
            rounded_avg_test_suite_success_rate = round(
                avg_test_suite_success_rate, 2)
            rounded_avg_mean_mutant_catch_rate = round(
                avg_mean_mutant_catch_rate, 2)
            buf += f"& {l} & {kind} & {rounded_avg_test_suite_success_rate} & {rounded_avg_mean_mutant_catch_rate} \\\\\n"
        else:
            buf += f"{l},{kind},{avg_test_suite_success_rate},{avg_mean_mutant_catch_rate}\n"

    if args.latex:
        buf = buf[:-1]
        buf += " \\midrule\n"

    print(buf)
