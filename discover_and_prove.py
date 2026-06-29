from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys

from tqdm import tqdm
from goedel_wrapper import process_theorem
from agent import MODEL_NAME, agent
from models import completions_with_backoff, gpt_oss
from data_loader import (
    load_putnam_formal_dataset,
    load_combi_formal_dataset,
    load_minif2f_informal_dataset,
    load_fimo_informal_dataset,
    load_putnam_informal_dataset,
    load_combi_informal_dataset,
    load_minif2f_formal_dataset,
    load_fimo_formal_dataset,
)

with open("prompts/rewrite.txt", "r") as f:
    rewrite_prompt = f.read()


def _extract_lean_code(result: str) -> str:
    """Extract the first fenced code block and drop a leading ``lean``/``lean4``
    language tag, if present."""
    if "```" in result:
        result = result.split("```")[1]
        result = re.sub(r"^lean[0-9]*\n", "", result, count=1)
    return result


def rewrite(
    formal_statement: str, informal_statement: str, informal_solution: str
) -> str:
    prompt = rewrite_prompt.replace("{{LEAN4_STATEMENT}}", formal_statement)
    prompt = prompt.replace("{{NATURAL_LANGUAGE_PROBLEM}}", informal_statement)
    prompt = prompt.replace("{{NATURAL_LANGUAGE_SOLUTION}}", informal_solution or "")

    result = gpt_oss(prompt)
    return _extract_lean_code(result[0] if result else "")


def direct_rewrite(formal_statement: str, informal_statement: str) -> str:
    """Straight Rewriting (Table 5 ablation): ask the model to discover the
    answer and rewrite the Hard Mode statement into an Easy Mode one in a single
    step, with no separately discovered solution provided."""
    prompt = rewrite_prompt.replace("{{LEAN4_STATEMENT}}", formal_statement)
    prompt = prompt.replace("{{NATURAL_LANGUAGE_PROBLEM}}", informal_statement)
    prompt = prompt.replace(
        "{{NATURAL_LANGUAGE_SOLUTION}}",
        "You need to solve the problem first. No solution is provided.",
    )

    result = gpt_oss(prompt)
    return _extract_lean_code(result[0] if result else "")


def check_hard_or_easy(formal_statement: str) -> bool:
    # count "sorry" in the formal statement
    # if there are 2 or more, return True (hard)
    # else return False (easy)
    sorry_count = formal_statement.count("sorry")
    return sorry_count >= 2


def no_agent_solver(informal_statement: str) -> str:
    informal_solution: str = (
        completions_with_backoff(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": informal_statement + "\nLet's think step by step.",
                },
            ],
            temperature=0.1,
            max_tokens=32768,
            reasoning_effort="high",
        )
        .choices[0]
        .message.content
        or ""
    )  # type: ignore
    return informal_solution


def prove(
    name: str,
    formal_statement: str,
    informal_statement: str,
    no_agent: bool = False,
    use_deepseek_v2: bool = False,
    rewrite_mode: str = "ours",
) -> tuple[str, bool, str | None, str | None, list[str], list[str], list[bool]]:
    is_hard = check_hard_or_easy(formal_statement)
    if is_hard:
        if rewrite_mode == "none":
            # No Rewriting (Table 5): feed the Hard Mode statement (two sorrys)
            # straight to the prover without any rewriting.
            informal_solution = None
            easy_statement = formal_statement
        elif rewrite_mode == "straight":
            # Straight Rewriting (Table 5): discover the answer and rewrite the
            # statement in a single step, with no separately discovered solution.
            informal_solution = None
            easy_statement = direct_rewrite(formal_statement, informal_statement)
        else:
            # Ours: discover the answer first, then rewrite the statement with it.
            if no_agent:
                informal_solution = no_agent_solver(informal_statement)
            else:
                informal_solution = agent(informal_statement)
                if informal_solution is None:
                    informal_solution = no_agent_solver(informal_statement)
            easy_statement = rewrite(
                formal_statement, informal_statement, informal_solution
            )
        _, responses, statements, checks = process_theorem(
            name, easy_statement, use_deepseek_v2=use_deepseek_v2
        )
        return (
            name,
            is_hard,
            informal_solution,
            easy_statement,
            responses,
            statements,
            checks,
        )
    else:
        _, responses, statements, checks = process_theorem(
            name, formal_statement, use_deepseek_v2=use_deepseek_v2
        )
        return (name, is_hard, None, None, responses, statements, checks)


def main(
    dataset: str = "putnam",
    ratio: float = 1.0,
    no_agent: bool = False,
    use_deepseek_v2: bool = False,
    rewrite_mode: str = "ours",
):
    if dataset == "putnam":
        formal_dataset = load_putnam_formal_dataset(ratio=ratio)
        informal_dataset = load_putnam_informal_dataset()
    elif dataset == "combi":
        formal_dataset = load_combi_formal_dataset(ratio=ratio)
        informal_dataset = load_combi_informal_dataset()
    elif dataset == "minif2f_annotated":
        formal_dataset = load_minif2f_formal_dataset(ratio=ratio)
        informal_dataset = load_minif2f_informal_dataset()
    elif dataset == "fimo_annotated":
        formal_dataset = load_fimo_formal_dataset(ratio=ratio)
        informal_dataset = load_fimo_informal_dataset()
    else:
        raise ValueError("Unknown dataset: " + dataset)

    # name, is_hard, informal_solution, easy_statement, responses, statements, checks
    results: list[
        tuple[str, bool, str | None, str | None, list[str], list[str], list[bool]]
    ] = []

    if not formal_dataset:
        print("No theorems to process.")
        return

    # Use at most 2 worker threads (kept low to respect API rate limits).
    max_workers = max(1, min(2, len(formal_dataset)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                prove,
                name,
                formal_dataset[name],
                informal_dataset.get(name, ""),
                no_agent,
                use_deepseek_v2,
                rewrite_mode,
            )
            for name in formal_dataset.keys()
        ]

        # Create progress bar with better configuration
        progress_bar = tqdm(
            total=len(formal_dataset),
            desc="Processing theorems",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

        for future in as_completed(futures):
            try:
                (
                    name,
                    is_hard,
                    informal_solution,
                    easy_statement,
                    responses,
                    statements,
                    checks,
                ) = future.result()

                results.append(
                    (
                        name,
                        is_hard,
                        informal_solution,
                        easy_statement,
                        responses,
                        statements,
                        checks,
                    )
                )

                # Save intermediate results with explicit flush
                with open(
                    f"eval_dap_on_{dataset}_intermediate_{ratio}{'_no_agent' if no_agent else ''}{'_deepseek_v2' if use_deepseek_v2 else ''}{('_' + rewrite_mode) if rewrite_mode != 'ours' else ''}.json",
                    "w",
                ) as f:
                    json.dump(
                        [
                            {
                                "name": n,
                                "is_hard": hard,
                                "informal_solution": sol,
                                "easy_statement": stmt,
                                "responses": rsp,
                                "statements": stmts,
                                "checks": chks,
                            }
                            for n, hard, sol, stmt, rsp, stmts, chks in results
                        ],
                        f,
                        indent=4,
                    )
                    f.flush()  # Ensure data is written to disk

                # Update progress bar with more detailed information
                success_count = sum(any(chks) for _, _, _, _, _, _, chks in results)
                progress_bar.set_postfix_str(
                    f"Current: {name} ({'hard' if is_hard else 'easy'}) | "
                    f"Proved: {success_count}/{len(results)} | "
                    f"Success: {'✓' if any(checks) else '✗'}"
                )
                progress_bar.update(1)

            except Exception as e:
                print(f"\nError processing theorem with error: {e}.")
                sys.stdout.flush()
                progress_bar.update(1)

        progress_bar.close()
    print("All done.")
    pass_count = sum(any(chks) for _, _, _, _, _, _, chks in results)
    total_count = len(results)
    print(f"Proved {pass_count} out of {total_count} theorems.")


if __name__ == "__main__":
    # This module exposes the core Discover-and-Prove pipeline. Prefer the CLI in
    # run.py, e.g.:
    #   python run.py main --dataset putnam              # DAP w/ Agent
    #   python run.py main --dataset putnam --no-agent   # DAP w/o Agent
    #   python run.py rewrite-ablation --dataset putnam --mode none
    main(dataset="putnam")
