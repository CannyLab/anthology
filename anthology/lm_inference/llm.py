import os
from multiprocessing import pool

from typing import Optional


import openai
import os
from tiktoken import get_encoding
from tqdm import tqdm
from transformers import AutoTokenizer

from .backoff import retry_with_exponential_backoff
from .survey_system_prompts import PARTY_ID_SYSTEM_PROMPT, PARTY_STRENGTH_SYSTEM_PROMPT

TOGETHER_MODEL_DICT = {
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "mixtral2": "mistralai/Mixtral-8x22B",
    "llama-2-70b": "meta-llama/Llama-2-70b-hf",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "mistral-7b": "mistral-community/Mistral-7B-v0.2",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "mixtral2_chat": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "llama-3-70b-chat": "meta-llama/Llama-3-70B-chat-hf",
}

TOGEHTER_MODEL_DEDICATED_DICT = {
    "llama-3-70b": "kim024680@snu.ac.kr/meta-llama/Meta-Llama-3-70B-234a8a14",  # dedicated endpoint
}


def prepare_llm(api_provider: str, model_name: str) -> str:
    if "together" in api_provider:  # together-ai api
        openai.api_base = "https://api.together.xyz/v1"
        openai.api_key = os.environ.get("TOGETHER_API_KEY")

        # get model name
        if "dedicated" in api_provider:
            model = TOGEHTER_MODEL_DEDICATED_DICT.get(model_name, None)
        else:
            model = TOGETHER_MODEL_DICT.get(model_name, None)

        if model is None:
            raise ValueError(
                f"Invalid model name: {model_name}, available models: {TOGETHER_MODEL_DICT.keys()}"
            )

    elif "localhost" in api_provider:  # vllm
        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "EMPTY"

        # get model name
        models = openai.Model.list()
        model = models["data"][0]["id"]

    elif "openai" in api_provider:  # openai
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        model = model_name
    else:
        raise ValueError(f"Invalid api provider: {api_provider}")

    return model


def get_tokenizer(llm="davinci"):
    # if openai model
    if any(model in llm for model in ["davinci", "curie", "babbage", "ada"]):
        return get_encoding("cl100k_base")
    else:  # if open source models
        from transformers import AutoTokenizer

        if "llama-3" in llm.lower():
            return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
        elif "llama-2" in llm.lower():
            return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-2-70B")
        elif "mixtral2" in llm.lower() or "8x22B" in llm:
            return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
        elif "mixtral" in llm.lower() or "8x7B" in llm:
            return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        elif "mistral-7b" in llm.lower():
            return AutoTokenizer.from_pretrained("mistral-community/Mistral-7B-v0.2")
        else:
            # TODO: Add more LLMs
            raise ValueError(f"invalid model name: {llm}")


@retry_with_exponential_backoff(
    max_retries=20,
    no_retry_on=(
        openai.error.AuthenticationError,
        openai.error.InvalidRequestError,
    ),
)
def prompt_llm(
    prompt,
    max_tokens=64,
    temperature=0,
    stop=None,
    model_name="text-davinci-002",
    top_p=1.0,
    echo=False,
    logprobs=0,
    n=1,
):
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        logprobs=logprobs,
        top_p=top_p,
        echo=echo,
        n=n,
    )

    if n == 1:
        return response.choices[0]
    if n > 1:
        return response.choices
    else:
        raise ValueError("n must be greater than 0")


@retry_with_exponential_backoff(
    max_retries=20,
    no_retry_on=(
        openai.error.AuthenticationError,
        openai.error.InvalidRequestError,
    ),
)
def prompt_llm_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens=64,
    temperature=0,
    stop=None,
    model_name="gpt-4o",
    top_p=1.0,
    echo=False,
    logprobs=0,
):
    if "gpt" in model_name:
        openai_api_base = "https://api.openai.com/v1"
    else:
        openai_api_base = "https://api.together.xyz/v1"
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        top_p=top_p,
        api_key=(
            os.environ.get("OPENAI_API_KEY")
            if "openai" in openai_api_base
            else os.environ.get("TOGETHER_API_KEY")
        ),
        api_base=openai_api_base,
    )
    return response.choices[0].message.content


@retry_with_exponential_backoff(
    max_retries=20,
    no_retry_on=(
        openai.error.AuthenticationError,
        openai.error.InvalidRequestError,
    ),
)
def prompt_llm_chat_with_messages(
    messages: list,
    max_tokens: int = 64,
    temperature: float = 0.0,
    stop: list = [],
    model_name: str = "gpt-4o",
    top_p: float = 1.0,
    echo: bool = False,
    logprobs: bool = True,
    top_logprobs: int = 20,
):
    if "gpt" in model_name:
        openai_api_base = "https://api.openai.com/v1"
    else:
        openai_api_base = "https://api.together.xyz/v1"

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        top_p=top_p,
        api_key=(
            os.environ.get("OPENAI_API_KEY")
            if "openai" in openai_api_base
            else os.environ.get("TOGETHER_API_KEY")
        ),
        api_base=openai_api_base,
    )

    return response.choices[0]


@retry_with_exponential_backoff(
    max_retries=20,
    no_retry_on=(
        openai.error.AuthenticationError,
        openai.error.InvalidRequestError,
    ),
)
def party_id_parser(
    prompt, max_tokens=64, temperature=0, stop=None, model_name="gpt-3.5-turbo"
):
    messages = [
        {"role": "system", "content": PARTY_ID_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    completition = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )

    return completition.choices[0].message


def parallel_generator(
    f,
    parallel_case_ids: list[int],
    cfg: dict,
    output_dict: dict,
    tokenizer: Optional[AutoTokenizer] = None,
):
    # log.info("Generating {} cases".format(len(parallel_case_ids)))
    if tokenizer is None:
        with pool.ThreadPool(cfg.num_processes) as p:
            list(
                tqdm(
                    p.imap(
                        lambda inputs: f(
                            inputs,
                            cfg,
                            output_dict,
                        ),
                        parallel_case_ids,
                    )
                )
            )
    else:
        with pool.ThreadPool(cfg.num_processes) as p:
            list(
                tqdm(
                    p.imap(
                        lambda inputs: f(inputs, cfg, output_dict, tokenizer),
                        parallel_case_ids,
                    )
                )
            )


def parallel_generator_v2(
    f,
    parallel_case_ids: list[int],
    cfg: dict,
    output_dict: dict,
    human_info_list: list,
    tokenizer: Optional[AutoTokenizer] = None,
):
    if tokenizer is None:
        with pool.ThreadPool(cfg.num_processes) as p:
            list(
                tqdm(
                    p.imap(
                        lambda inputs: f(
                            inputs,
                            cfg,
                            output_dict,
                            human_info_list,
                        ),
                        parallel_case_ids,
                    )
                )
            )
    else:
        with pool.ThreadPool(cfg.num_processes) as p:
            list(
                tqdm(
                    p.imap(
                        lambda inputs: f(
                            inputs, cfg, output_dict, human_info_list, tokenizer
                        ),
                        parallel_case_ids,
                    )
                )
            )
