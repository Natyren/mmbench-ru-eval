import argparse

import pandas as pd


def compute_accuracy(df, aggregate="l2"):
    assert aggregate in ["l1", "l2", "source"], "Aggregate can only be l1 or l2"
    if aggregate == "l1":
        metrics = (
            df[["category", "match"]]
            .groupby("category")
            .agg({"match": ["sum", "count"]})
        )
    elif aggregate == "l2":
        metrics = (
            df[["l2-category", "match"]]
            .groupby("l2-category")
            .agg({"match": ["sum", "count"]})
        )
    else:
        metrics = (
            df[["source", "match"]].groupby("source").agg({"match": ["sum", "count"]})
        )
    metrics.columns = ["sum", "count"]
    metrics.reset_index(inplace=True)
    metrics["accuracy"] = metrics["sum"] / metrics["count"]
    metrics.drop(columns=["sum", "count"], inplace=True)
    return metrics


def process_answers(path, aggregate):
    answers = pd.read_csv(path, sep="\t")
    assert "predict" in answers.columns, "No predict columns in file, check correctness"
    answers["match"] = answers["answer"] == answers["predict"]
    metrics = compute_accuracy(answers, aggregate)
    metrics.to_csv(f"./metrics_{aggregate}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MMBench-RU-evaluation",
        description="Programm checks the predictions on ru MMBench",
    )
    parser.add_argument("-p", "--path")  # option that takes a value
    parser.add_argument("-a", "--aggregate", default="l2")
    args = parser.parse_args()
    process_answers(args.path, args.aggregate)
