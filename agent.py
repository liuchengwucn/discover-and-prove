"""
Discovery Module: natural-language solution generation with iterative
self-verification and self-correction.

Adapted from code originally written by Lin Yang and Yichen Huang, MIT License.
https://github.com/lyang36/IMO25/blob/main/code/agent_oai.py
"""

import os
import sys
import json
import logging
from typing import Any

import argparse
from dotenv import load_dotenv
from models import completions_with_backoff

load_dotenv()

# --- CONFIGURATION ---
# The model to use.
MODEL_NAME = "openai/gpt-oss-120b"

# Verbose per-step traces go to this logger at DEBUG level (silent unless the
# caller configures logging, e.g. logging.basicConfig(level=logging.DEBUG)).
logger = logging.getLogger(__name__)


def _log(*args: Any) -> None:
    logger.debug(" ".join(str(arg) for arg in args))


# --- PROMPT LOADING ---
# The prompts documented in the paper appendix are stored as plain-text files in
# the prompts/ directory so that they are the single source of truth.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


step1_prompt = _load_prompt("generation.txt")

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.
"""

correction_prompt = _load_prompt("correction.txt")

verification_system_prompt = _load_prompt("verification.txt")


verification_reminder = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""


def read_file_content(filepath: str) -> str:
    """
    Reads and returns the content of a file.
    Exits if the file cannot be read.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading file '{filepath}': {e}")
        sys.exit(1)


def send_and_extract(
    system_prompt: str,
    question_prompt: str,
    other_prompts: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Combines building the request payload, sending the API request, and extracting the text from the response.
    """
    msg: list[dict[str, str]] = []

    if system_prompt:
        msg.append({"role": "system", "content": system_prompt})
    if history:
        msg.extend(history)

    input_text = question_prompt

    if other_prompts:
        for prompt in other_prompts:
            input_text += f"\n\nAdditional instruction: {prompt}"

    msg.append({"role": "user", "content": input_text})

    response = completions_with_backoff(
        model=MODEL_NAME,
        messages=msg,
        temperature=0.1,
        max_tokens=32768,
        reasoning_effort="high",
    )

    _log(">>>>>> Response:")
    _log(response.to_json(indent=2))

    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    else:
        return ""


def extract_detailed_solution(
    solution: str, marker: str = "Detailed Solution", after: bool = True
):
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    idx = solution.find(marker)
    if idx == -1:
        return ""
    if after:
        return solution[idx + len(marker) :].strip()
    else:
        return solution[:idx].strip()


def verify_solution(problem_statement: str, solution: str, verbose: bool = True):
    dsol = extract_detailed_solution(solution)

    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_reminder}
"""
    if verbose:
        _log(">>>>>>> Start verification.")
        _log(">>>>>>> Verification prompt:")
        _log(
            json.dumps(
                {
                    "system_prompt": verification_system_prompt,
                    "question_prompt": newst,
                },
                indent=4,
            )
        )

    out = send_and_extract(
        system_prompt=verification_system_prompt, question_prompt=newst
    )

    if verbose:
        _log(">>>>>>> Verification results:")
        _log(json.dumps(out, indent=4))

    check_correctness = (
        """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?"""
        + "\n\n"
        + out
    )
    o = send_and_extract(system_prompt="", question_prompt=check_correctness)

    if verbose:
        _log(">>>>>>> Is verification good?")
        _log(json.dumps(o, indent=4))

    bug_report = ""

    if "yes" not in o.lower():
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

    if verbose:
        _log(">>>>>>>Bug report:")
        _log(json.dumps(bug_report, indent=4))

    return bug_report, o


def init_explorations(
    problem_statement: str, verbose: bool = True, other_prompts: list[str] = []
) -> tuple[str, str, str]:
    _log(">>>>>> Initial prompt.")
    _log(
        json.dumps(
            {
                "system_prompt": step1_prompt,
                "question_prompt": problem_statement,
                "other_prompts": other_prompts,
            },
            indent=4,
        )
    )

    output1 = send_and_extract(
        system_prompt=step1_prompt,
        question_prompt=problem_statement,
        other_prompts=other_prompts,
    )

    _log(">>>>>>> First solution: ")
    _log(json.dumps(output1, indent=4))

    _log(">>>>>>> Self improvement start:")
    original_input = problem_statement
    if other_prompts:
        for prompt in other_prompts:
            original_input += f"\n\nAdditional instruction: {prompt}"

    history = [
        {"role": "user", "content": original_input},
        {"role": "assistant", "content": output1},
    ]

    solution = send_and_extract(
        system_prompt=step1_prompt,
        question_prompt=self_improvement_prompt,
        history=history,
    )

    _log(">>>>>>> Corrected solution: ")
    _log(json.dumps(solution, indent=4))

    _log(">>>>>>> Verify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    _log(">>>>>>> Initial verification: ")
    _log(json.dumps(verify, indent=4))
    _log(f">>>>>>> verify results: {good_verify}")

    return solution, verify, good_verify


def agent(problem_statement: str, other_prompts: list[str] = []) -> str | None:
    solution, verify, good_verify = init_explorations(
        problem_statement, True, other_prompts
    )

    error_count = 0
    correct_count = 1
    for i in range(30):
        _log(
            f"Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}"
        )

        try:
            if "yes" not in good_verify.lower():
                # clear
                correct_count = 0
                error_count += 1

                # self improvement
                _log(">>>>>>> Verification does not pass, correcting ...")
                # establish a new prompt that contains the solution and the verification
                original_input = problem_statement
                if other_prompts:
                    for prompt in other_prompts:
                        original_input += f"\n\nAdditional instruction: {prompt}"

                history = [
                    {"role": "user", "content": original_input},
                    {"role": "assistant", "content": solution},
                ]

                _log(">>>>>>> New prompt:")
                _log(
                    json.dumps(
                        {
                            "system_prompt": step1_prompt,
                            "question_prompt": f"{correction_prompt}\n\n{verify}",
                            "history": history,
                        },
                        indent=4,
                    )
                )

                solution = send_and_extract(
                    system_prompt=step1_prompt,
                    question_prompt=f"{correction_prompt}\n\n{verify}",
                    history=history,
                )

                _log(">>>>>>> Corrected solution:")
                _log(json.dumps(solution, indent=4))

            _log(">>>>>>> Verify the solution.")
            verify, good_verify = verify_solution(problem_statement, solution)

            if "yes" in good_verify.lower():
                _log(">>>>>>> Solution is good, verifying again ...")
                correct_count += 1
                error_count = 0

            if correct_count >= 5:
                _log(">>>>>>> Correct solution found.")
                _log(json.dumps(solution, indent=4))
                return solution

            elif error_count >= 10:
                _log(">>>>>>> Failed in finding a correct solution.")
                return None
        except Exception as e:
            _log("Unexpected error:", e, "retry...")

    _log(">>>>>>> Failed in finding a correct solution.")
    return None


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="IMO Problem Solver Agent")
    parser.add_argument(
        "problem_file",
        nargs="?",
        default="problem_statement.txt",
        help="Path to the problem statement file (default: problem_statement.txt)",
    )
    parser.add_argument("--log", "-l", type=str, help="Path to log file (optional)")
    parser.add_argument(
        "--other_prompts", "-o", type=str, help="Other prompts (optional)"
    )
    parser.add_argument(
        "--max_runs",
        "-m",
        type=int,
        default=10,
        help="Maximum number of runs (default: 10)",
    )

    args = parser.parse_args()

    # INFO to the console; full per-step traces (DEBUG) go to --log if given.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.log:
        file_handler = logging.FileHandler(args.log, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(f"Logging detailed traces to file: {args.log}")

    max_runs: int = args.max_runs

    other_prompts: list[str] = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(",")

    _log(">>>>>>> Other prompts:")
    _log(str(other_prompts))

    problem_statement = read_file_content(args.problem_file)

    for i in range(max_runs):
        logger.info(f"Run {i} of {max_runs} ...")
        try:
            sol = agent(problem_statement, other_prompts)
            if sol is not None:
                logger.info(f"Found a correct solution in run {i}.")
                _log(json.dumps(sol, indent=4))
                break
        except Exception as e:
            logger.warning(f"Error in run {i}: {e}")
            continue
