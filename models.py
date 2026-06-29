import os
from typing import Any
import openai
from openai.types.chat import ChatCompletion
from functools import partial
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import logging
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_BASE_URL", "")

client = openai.OpenAI(api_key=api_key, base_url=base_url)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def completions_with_backoff(model: str, **kwargs: Any) -> ChatCompletion:
    response = client.chat.completions.create(model=model, **kwargs)  # type: ignore
    assert isinstance(response, ChatCompletion)
    return response


def gpt(
    prompt: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    n: int = 1,
    top_p: int | None = None,
    stop: list[str] = [],
) -> list[str]:
    messages = [{"role": "user", "content": prompt}]
    outputs: list[str] = []
    while n > 0:
        cnt = min(n, 16)
        n -= cnt
        res = completions_with_backoff(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
            top_p=top_p,
        )
        outputs.extend([choice.message.content or "" for choice in res.choices])
    return outputs


gemini_25_flash = partial(gpt, model_name="google/gemini-2.5-flash", max_tokens=8192)
gpt_oss = partial(
    gpt, model_name="openai/gpt-oss-120b", max_tokens=8192, temperature=1.0
)

default_model = gemini_25_flash

if __name__ == "__main__":
    print(gpt_oss("Hello! Who are you?"))
