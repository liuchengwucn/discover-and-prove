import os
import json
import csv


def load_putnam_formal_dataset(ratio: float = 1.0) -> dict[str, str]:
    path: str = "datasets/putnambench/formal"
    dataset: dict[str, str] = {}
    for file in os.listdir(path):
        if file.endswith(".lean"):
            name = file[:-5]
            with open(os.path.join(path, file), "r") as f:
                content = f.read()
            # remove any line that starts with --
            # because they may contain answers
            lines = [
                line
                for line in content.split("\n")
                if not line.strip().startswith("--")
            ]
            content = "\n".join(lines)
            dataset[name] = content

    if ratio < 1.0:
        keys = list(dataset.keys())
        selected_keys = keys[: int(len(keys) * ratio)]
        dataset = {k: dataset[k] for k in selected_keys}
    return dataset


def load_combi_formal_dataset(ratio: float = 1.0) -> dict[str, str]:
    path = "datasets/combibench/formal"
    dataset: dict[str, str] = {}
    for file in os.listdir(path):
        if file.endswith(".lean"):
            name = file[:-5]
            with open(os.path.join(path, file), "r") as f:
                content = f.read()
            dataset[name] = content

    if ratio < 1.0:
        keys = list(dataset.keys())
        selected_keys = keys[: int(len(keys) * ratio)]
        dataset = {k: dataset[k] for k in selected_keys}
    return dataset


def load_minif2f_formal_dataset(ratio: float = 1.0) -> dict[str, str]:
    path = "datasets/minif2f_hard.json"
    with open(path, "r") as f:
        data = json.load(f)

    dataset = {item["id"]: item["formal_statement"] for item in data}
    if ratio < 1.0:
        keys = list(dataset.keys())
        selected_keys = keys[: int(len(keys) * ratio)]
        dataset = {k: dataset[k] for k in selected_keys}
    return dataset


def load_fimo_formal_dataset(
    ratio: float = 1.0,
    easy_only: bool = False,
) -> dict[str, str]:
    path = "datasets/fimo_hard.json"
    with open(path, "r") as f:
        data = json.load(f)

    if easy_only:
        dataset = {item["id"]: item["formal_statement_easy"] for item in data}
    else:
        dataset = {item["id"]: item["formal_statement"] for item in data}

    if ratio < 1.0:
        keys = list(dataset.keys())
        selected_keys = keys[: int(len(keys) * ratio)]
        dataset = {k: dataset[k] for k in selected_keys}
    return dataset


def load_putnam_informal_dataset(
    path: str = "datasets/putnambench/informal.json",
) -> dict[str, str]:
    with open(path, "r") as f:
        informal_data = json.load(f)
    informal_dict = {
        item["problem_name"]: item["informal_statement"] for item in informal_data
    }
    return informal_dict


def load_combi_informal_dataset(
    path: str = "datasets/combibench/informal.csv",
) -> dict[str, str]:
    informal_dict: dict[str, str] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            theorem_name = row["theorem_name"]
            natural_language = row["natural_language"]
            informal_dict[theorem_name] = natural_language
    return informal_dict


def load_minif2f_informal_dataset():
    path = "datasets/minif2f_hard.json"
    with open(path, "r") as f:
        data = json.load(f)

    informal_dict = {item["id"]: item["math_problem"] for item in data}
    return informal_dict


def load_fimo_informal_dataset():
    path = "datasets/fimo_hard.json"
    with open(path, "r") as f:
        data = json.load(f)

    informal_dict = {item["id"]: item["math_problem"] for item in data}
    return informal_dict


# --- Ground-truth answer loaders (used by the Table 3 answer-accuracy judge) ---
# Each returns {problem_name: ground_truth_answer}. Proof-style problems have an
# empty/"None." answer and are filtered out before judging.


def load_putnam_answer_dataset(
    path: str = "datasets/putnambench/informal.json",
) -> dict[str, str]:
    with open(path, "r") as f:
        informal_data = json.load(f)
    return {
        item["problem_name"]: (item.get("informal_solution") or "")
        for item in informal_data
    }


def load_combi_answer_dataset(
    path: str = "datasets/combibench/informal.csv",
) -> dict[str, str]:
    answer_dict: dict[str, str] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer_dict[row["theorem_name"]] = row.get("answer") or ""
    return answer_dict


def load_minif2f_answer_dataset(
    path: str = "datasets/minif2f_hard.json",
) -> dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f)
    return {item["id"]: (item.get("solution") or "") for item in data}


def load_fimo_answer_dataset(
    path: str = "datasets/fimo_hard.json",
) -> dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f)
    return {item["id"]: (item.get("solution") or "") for item in data}
