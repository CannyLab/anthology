import logging
import pathlib
import hydra
from threading import Lock
import random

import pandas as pd
import pyreadstat
from omegaconf import OmegaConf

from anthology.lm_inference.llm import (
    prepare_llm,
    get_tokenizer,
    parallel_generator_v2,
    prompt_llm,
    prompt_llm_chat,
)
from anthology.utils.data_utils import save_result_to_csv, publish_result, get_config_path
from anthology.utils.random_utils import set_random_seed
from anthology.survey.atp_baseline_survey import generate_baseline_prompt_prefix

log = logging.getLogger(__name__)

# Multiprocessing
lock = Lock()

# config file path
config_path = get_config_path("generate_backstory")


def filter_demographic_variable(
    input: dict,
    trait_of_interest: list,
) -> dict:
    return {k: v for k, v in input.items() if input[k]["topic"] in trait_of_interest}


def generator_wrapper(
    case_id: int,
    cfg: dict,
    output_dict: dict,
    human_info_list: list,
    tokenizer,
):
    generated_backstory = generate_backstory(
        backstory_idx=case_id,
        cfg=cfg,
        tokenizer=tokenizer,
        human_info=human_info_list[case_id],
    )
    if cfg.debug:
        log.info(f"Generation result:\n{generated_backstory}")
    with lock:
        output_dict[case_id] = generated_backstory

        # Save generated_backstory to csv every freq cases
        if (case_id + 1) % cfg.freq == 0:
            save_result_to_csv(output_dict, cfg.output_data_path)


def generate_backstory(backstory_idx: int, cfg: dict, tokenizer, human_info: dict):
    # Structure of the prompt generating the backstory:
    # The prompt is a concatenation of the following:
    #    a. Backstory preamble question e.g. "Are you a US citizen? Yes."
    #    b. Backstory question e.g. "What is your name?"
    #    c. Backstory answer e.g. "My name is John Doe."
    log.info(f"Generating backstory {backstory_idx}")

    prompt = ""
    preamble_prompt = ""
    preamble_answer = ""
    preamble_answer_logprobs = 0
    preamble_answer_token_length = 0

    backstory_question = cfg.self_generate.backstory_question
    surveyor = cfg.self_generate.surveyor
    respondent = cfg.self_generate.respondent
    connector = cfg.self_generate.connector

    temperature = cfg.llm_parameters.temperature
    max_tokens = cfg.llm_parameters.max_tokens
    top_p = cfg.llm_parameters.top_p
    model_name = cfg.llm_parameters.model_name

    prompt = ""

    if cfg.prompt_style == "qa":
        prompt += "Below you will be asked to complete some demographic questions, and then answer a question.\n\n"

    prompt += f"{human_info['full_passage']}{surveyor}{connector} {backstory_question}\n\n{respondent}{connector}"

    if cfg.debug:
        log.info(f"Prompt: {prompt}")

    llm_parameters = cfg.llm_parameters
    if "chat" in llm_parameters.model_name:
        generated_backstory_result = prompt_llm_chat(
            system_prompt=llm_parameters.system_prompt,
            user_prompt=prompt,
            max_tokens=llm_parameters.max_tokens,
            temperature=llm_parameters.temperature,
            stop=[f"{surveyor}", "Q:", "Interview"],
            model_name=llm_parameters.model_name,
            top_p=llm_parameters.top_p,
            logprobs=1,
        )
    else:
        generated_backstory_result = prompt_llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[f"{surveyor}", "Q:", "Interview"],
            model_name=model_name,
            top_p=top_p,
            echo=False,
            logprobs=1,
        ).text

    temperature = cfg.llm_parameters.temperature
    max_tokens = cfg.llm_parameters.max_tokens
    top_p = cfg.llm_parameters.top_p
    model_name = cfg.llm_parameters.model_name

    result_dict = {
        "full_passage": f"{surveyor}{connector} {backstory_question}\n\n{respondent}{connector}"
        + generated_backstory_result,
        "backstory": generated_backstory_result,
        "input_prompt": prompt,
        "preamble_prompt": preamble_prompt,
        "surveyor": surveyor,
        "respondent": respondent,
        "backstory_generated_token_length": len(
            tokenizer.encode(generated_backstory_result)
        ),
        # "backstory_logprob": backstory_likelihood,
        "total_generated_token_length": len(
            tokenizer.encode(generated_backstory_result)
        ),
        "preamble_answer": preamble_answer,
        "preamble_answer_logprobs": preamble_answer_logprobs,
        "preamble_answer_token_length": preamble_answer_token_length,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    for key in human_info:
        if key not in result_dict:
            result_dict[key] = human_info[key]

    return result_dict


@hydra.main(config_path=str(config_path), config_name="config_baseline_based")
def my_app(cfg):
    output_dict = {}
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Running LLM with config:\n{OmegaConf.to_yaml(cfg.llm_parameters)}")

    set_random_seed(cfg.random_seed)

    llm_cfg = cfg.llm_parameters
    model = prepare_llm(
        api_provider=llm_cfg.api_provider,
        model_name=llm_cfg.model_name,
    )
    model_name = model.split("/")[-1]

    OmegaConf.set_struct(cfg, False)
    cfg.llm_parameters.model_name = model
    tokenizer = get_tokenizer(cfg.llm_parameters.model_name)
    save_file_name = f"{model_name}_generated_backstory_atp_w{cfg.wave}_{cfg.prompt_style}_{cfg.output_data_name}.csv"
    output_data_path = pathlib.Path(cfg.run_dir) / save_file_name
    log.info(f"Output data path: {output_data_path}")
    cfg.output_data_path = output_data_path
    OmegaConf.set_struct(cfg, True)

    survey_data_df, meta = pyreadstat.read_sav(
        cfg.survey_data_path,
    )
    atp_demographics_df = pd.DataFrame(
        filter_demographic_variable(
            dict(pd.read_json(cfg.survey_demographics_metadata_path)),
            trait_of_interest=cfg.trait_of_interest,
        )
    )
    prompt_prefix_list = generate_baseline_prompt_prefix(
        survey_data_df=survey_data_df,
        demographics_df=atp_demographics_df,
        prompt_style=cfg.prompt_style,
        include_answer_forcing=True,
    )
    if cfg.debug:
        prompt_prefix_list = random.sample(prompt_prefix_list, 10)
    else:
        if cfg.num_backstories > 0:
            prompt_prefix_list = random.sample(prompt_prefix_list, cfg.num_backstories)
    log.info(f"Number of backstories to generate: {len(prompt_prefix_list)}")
    log.info(f"Output to be published to: {cfg.save_dir}/{save_file_name}")

    parallel_generator_v2(
        f=generator_wrapper,
        parallel_case_ids=(range(len(prompt_prefix_list))),
        cfg=cfg,
        output_dict=output_dict,
        human_info_list=prompt_prefix_list,
        tokenizer=tokenizer,
    )
    save_result_to_csv(output_dict, output_data_path)

    if not cfg.debug:
        publish_result(
            output=output_dict, publish_dir=cfg.save_dir, filename=save_file_name
        )


if __name__ == "__main__":
    my_app()
