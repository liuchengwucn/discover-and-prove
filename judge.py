"""Table 3: Discovery Module answer-accuracy evaluation.

Given an evaluation-results file produced by the main pipeline (``run.py main``),
this judges, for each Hard Mode problem, whether the Discovery Module's
natural-language answer matches the dataset ground-truth answer, using an
LLM-as-judge. This reproduces the answer-accuracy numbers reported in Table 3.

The judge model is configurable. The paper used ``gemini-2.5-flash``; an
open-source alternative (``gpt-oss-120b``) is also provided. Note that LLM
sampling is stochastic, so exact counts may vary slightly between runs.
"""

import json
import os

from tqdm import tqdm

from models import default_model, gpt_oss
from data_loader import (
    load_putnam_informal_dataset,
    load_combi_informal_dataset,
    load_minif2f_informal_dataset,
    load_fimo_informal_dataset,
    load_putnam_answer_dataset,
    load_combi_answer_dataset,
    load_minif2f_answer_dataset,
    load_fimo_answer_dataset,
)

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
with open(os.path.join(PROMPTS_DIR, "judge.txt"), "r", encoding="utf-8") as f:
    JUDGE_PROMPT = f.read()

# The judge used in the paper is gemini-2.5-flash (models.default_model).
JUDGE_MODELS = {
    "gemini": default_model,
    "gpt-oss": gpt_oss,
}


def _load_problems_and_answers(dataset: str) -> tuple[dict[str, str], dict[str, str]]:
    if dataset == "putnam":
        return load_putnam_informal_dataset(), load_putnam_answer_dataset()
    elif dataset == "combi":
        return load_combi_informal_dataset(), load_combi_answer_dataset()
    elif dataset == "minif2f":
        return load_minif2f_informal_dataset(), load_minif2f_answer_dataset()
    elif dataset == "fimo":
        return load_fimo_informal_dataset(), load_fimo_answer_dataset()
    raise ValueError("Unknown dataset: " + dataset)


def judge_answer(problem: str, ground_truth: str, model_solution: str, judge_model) -> bool:
    prompt = JUDGE_PROMPT.replace("{{PROBLEM}}", problem)
    prompt = prompt.replace("{{GROUND_TRUTH_SOLUTION}}", ground_truth)
    prompt = prompt.replace("{{STUDENT_SOLUTION}}", model_solution)
    out = judge_model(prompt)
    response = out[0] if out else ""
    return "yes" in response.lower()


def evaluate_answer_accuracy(
    results_path: str, dataset: str, judge: str = "gemini"
) -> tuple[int, int]:
    with open(results_path, "r") as f:
        results = json.load(f)

    problems, answers = _load_problems_and_answers(dataset)
    judge_model = JUDGE_MODELS[judge]

    total = 0
    correct = 0
    for item in tqdm(results, desc="Judging answers"):
        if not item.get("is_hard"):
            continue
        name = item["name"]
        if name not in answers:
            print(f"[WARN] {name} not found in ground-truth answers; skipping.")
            continue
        problem = (problems.get(name) or "").strip().lower()
        ground_truth = (answers.get(name) or "").strip().lower()
        model_solution = (item.get("informal_solution") or "").strip().lower()

        total += 1
        ok = judge_answer(problem, ground_truth, model_solution, judge_model)
        item["informal_solution_check"] = ok
        if ok:
            correct += 1

    passrate = correct / total * 100 if total else 0.0
    print(f"\nAnswer-accuracy on Hard Mode problems in {results_path}:")
    print(f"  Judge model: {judge}")
    print(f"  Total Hard Mode problems: {total}")
    print(f"  Correct answers: {correct}")
    print(f"  Accuracy: {passrate:.2f}%")

    out_path = results_path.replace(".json", "_with_answer_check.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Annotated results saved to {out_path}")
    return correct, total
