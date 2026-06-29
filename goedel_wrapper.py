import openai
from openai.types.chat import ChatCompletionMessageParam
from kimina_client import KiminaClient, SnippetStatus
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from models import client

# The Proving Module talks to a self-hosted Goedel-Prover-V2 server (any
# OpenAI-compatible endpoint, e.g. vLLM) and a Kimina server that mediates the
# Lean 4 REPL. Configure both via environment variables (see .env.example).
openai_client = openai.OpenAI(
    base_url=os.getenv("GOEDEL_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("GOEDEL_API_KEY", "EMPTY"),
)
kimina_client = KiminaClient(api_url=os.getenv("KIMINA_API_URL", "http://localhost:12332"))
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)

model_id = os.getenv("GOEDEL_MODEL_ID", "Goedel-LM/Goedel-Prover-V2-32B")


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(3),
)
def prove_theorem(
    formal_statement: str, sampling_size: int = 32, use_deepseek_v2: bool = False
) -> list[str]:
    """Sample ``sampling_size`` proof attempts for a Lean 4 statement from the
    prover (Goedel-Prover-V2, or DeepSeek-Prover-V2 when ``use_deepseek_v2``).

    Returns a list of raw model completions, each containing a proof plan and the
    formal proof.
    """

    if use_deepseek_v2:
        prompt = """
Complete the following Lean 4 code:

```lean4
{}```
""".strip()
    else:
        prompt = """
Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt.format(formal_statement)},
    ]

    if use_deepseek_v2:
        response = client.chat.completions.create(
            model="deepseek/deepseek-prover-v2",
            messages=messages,
            n=sampling_size,
            timeout=60 * 60,  # 1 hour
            max_tokens=30000,  # 30k tokens
        )

    else:
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            n=sampling_size,
            timeout=60 * 60,  # 1 hour
            max_tokens=30000,  # 30k tokens
        )

    output_text = [
        response.choices[i].message.content or "" for i in range(sampling_size)
    ]
    return output_text


default_header = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""


def extract_proof(formal_statement: str, output: str) -> str:
    """
    Extract the last ```lean4 ... ``` block from the model's output.

    Args:
        output: The model's response containing proof plan and formal proof

    Returns:
        The extracted Lean 4 proof as a string
    """
    # Extract the last ```lean4 ... ``` block from the output
    try:
        proof_start = output.rindex("```lean4") + len("```lean4")
        proof_end = output.rindex("```", proof_start)
        proof = output[proof_start:proof_end].strip()
        if "import Mathlib" not in proof:
            # extract header from original formal_statement if any
            try:
                header_end = formal_statement.index("theorem")
                header = formal_statement[:header_end].strip()
                proof = header + "\n\n" + proof
            except ValueError:
                proof = default_header + "\n\n" + proof
        return proof
    except ValueError:
        # Extract the last ``` ... ``` block if ```lean4``` is not found
        try:
            proof_start = output.rindex("```") + len("```")
            proof_end = output.rindex("```", proof_start)
            proof = output[proof_start:proof_end].strip()
            if "import Mathlib" not in proof:
                proof = default_header + "\n\n" + proof
            return proof
        except ValueError:
            # If no code block is found, return an empty string
            return ""


def check_statements(statements: list[str]) -> list[bool]:
    response = kimina_client.check(statements, show_progress=False)
    analyses = [r.analyze() for r in response.results]
    return [
        a.status == SnippetStatus.valid and (statements[i].strip() != "")
        for i, a in enumerate(analyses)
    ]


def generate_and_check(
    formal_statement: str, use_deepseek_v2: bool = False
) -> tuple[str, str, bool]:
    try:
        responses = prove_theorem(
            formal_statement, sampling_size=1, use_deepseek_v2=use_deepseek_v2
        )
    except Exception:
        # Prover request failed (timeout, connection error, etc.); treat this
        # sample as an empty (failed) attempt.
        logger.warning("prove_theorem failed for a sample", exc_info=True)
        responses = [""]
    statements = [extract_proof(formal_statement, output) for output in responses]
    checks = check_statements(statements)
    return (responses[0], statements[0], checks[0])


def process_theorem(
    name: str,
    formal_statement: str,
    sampling_size: int = 32,
    use_deepseek_v2: bool = False,
) -> tuple[str, list[str], list[str], list[bool]]:
    """
    Process a single theorem: generate proofs and check them.

    Args:
        name: Name of the theorem
        formal_statement: The Lean 4 theorem statement to prove

    Returns:
        Tuple of (name, responses, statements, checks)
    """
    responses: list[str] = []
    statements: list[str] = []
    checks: list[bool] = []
    with ThreadPoolExecutor(max_workers=min(8, sampling_size)) as executor:
        futures = [
            executor.submit(
                generate_and_check, formal_statement, use_deepseek_v2=use_deepseek_v2
            )
            for _ in range(sampling_size)
        ]

        for future in as_completed(futures):
            rsp, stmt, chk = future.result()
            responses.append(rsp)
            statements.append(stmt)
            checks.append(chk)
            if chk:
                # Early exit: a valid proof was found, cancel remaining samples.
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    return (name, responses, statements, checks)


def test():
    formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat


theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
""".strip()
    name, _, statements, checks = process_theorem(
        "square_equation_solution", formal_statement, sampling_size=32
    )
    print(f"Theorem: {name}")
    for i, (stmt, chk) in enumerate(zip(statements, checks)):
        print(f"Attempt {i + 1} - Valid: {chk}")
        print(stmt)
        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    test()
