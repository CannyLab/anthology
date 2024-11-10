import logging
import pathlib
import hydra
from threading import Lock
from typing import Type
from multiprocessing import pool

import pandas as pd
from omegaconf import OmegaConf, DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from anthology.utils.random_utils import set_random_seed
from anthology.utils.data_utils import (
    save_result_to_pkl,
    publish_result,
    save_result_to_csv,
    get_config_path,
)
from anthology.demographic_survey.political_affiliation import (
    format_affiliation_surveys,
    affiliation_parser,
)
from anthology.lm_inference.llm import prepare_llm, prompt_llm
from anthology.backstory.backstory import load_backstory

log = logging.getLogger(__name__)

# Multiprocessing
lock = Lock()

# config file path
config_path = get_config_path("surveys")


def parallel_survey(
    f,
    parallel_case_ids: list,
    backstory_df: pd.DataFrame,
    survey_cfg: Type[DictConfig],
    llm_cfg: Type[DictConfig],
    cfg: Type[DictConfig],
    model: str,
    output_dict: dict,
    debug: bool = False,
):
    with pool.ThreadPool(cfg.num_processes) as p:
        list(
            tqdm(
                p.imap(
                    lambda inputs: f(
                        inputs,
                        backstory_df,
                        survey_cfg,
                        llm_cfg,
                        cfg,
                        output_dict,
                        model,
                        debug,
                    ),
                    parallel_case_ids,
                )
            )
        )


def surveyor_wrapper(
    case_id: int,
    backstory_df: pd.DataFrame,
    survey_cfg: Type[DictConfig],
    llm_cfg: Type[DictConfig],
    cfg: Type[DictConfig],
    output_dict: dict,
    model: str,
    debug: bool = False,
):
    output_result = run_political_affiliation(
        case_id=case_id,
        backstory_df=backstory_df.iloc[case_id],
        survey_cfg=survey_cfg,
        llm_cfg=llm_cfg,
        cfg=cfg,
        model=model,
        debug=debug,
    )
    with lock:
        output_dict[case_id] = output_result

        if (case_id + 1) % cfg.freq == 0:
            csv_output_path = str(cfg.output_data_path).replace(".pkl", ".csv")
            csv_output_path = pathlib.Path(csv_output_path)

            save_result_to_csv(output_dict, csv_output_path)
            save_result_to_pkl(output_dict, cfg.output_data_path)


def run_political_affiliation(
    case_id: int,
    backstory_df: pd.DataFrame,
    survey_cfg: Type[DictConfig],
    llm_cfg: Type[DictConfig],
    cfg: Type[DictConfig],
    model: str,
    debug: bool = False,
):
    result_dict = {}

    log.info(f"Running political affiliation survey for case_id: {case_id}")

    # Format the survey
    formatted_question, democrat_first = format_affiliation_surveys(
        question=survey_cfg.party_identification.question,
        available_options=survey_cfg.party_identification.choices,
        consistency_prompt=survey_cfg.party_identification.consistency_prompt,
        randomize_choice=survey_cfg.party_identification.randomize_choices,
    )

    temperature = llm_cfg.temperature
    max_tokens = llm_cfg.max_tokens
    top_p = llm_cfg.top_p

    backstory = backstory_df.full_passage.strip()

    result_dict["temperature"] = temperature
    result_dict["max_tokens"] = max_tokens
    result_dict["top_p"] = top_p
    result_dict["model"] = model
    result_dict["backstory"] = backstory
    result_dict["uid"] = backstory_df.uid

    # Prompt the model
    prompt = f"{backstory}\n\n{formatted_question}"
    if "together" in llm_cfg.api_provider:
        response = []
        for _ in range(cfg.num_sample_response // 10):
            response.extend(
                prompt_llm(
                    model_name=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=["\n", "Question:"],
                    top_p=top_p,
                    logprobs=None,
                    n=10,
                )
            )
    else:
        response = prompt_llm(
            model_name=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n", "Question:"],
            top_p=top_p,
            logprobs=None,
            n=cfg.num_sample_response,
        )
    response = [r.text for r in response]
    parse_response_list = [affiliation_parser(r, democrat_first) for r in response]

    result_dict["political_affiliation_question"] = formatted_question
    result_dict["political_affiliation_prompt"] = prompt
    result_dict["political_affiliation_response_texts"] = response
    result_dict["political_affiliation_choices"] = parse_response_list
    result_dict["political_affiliation_response_distribution"] = [
        parse_response_list.count("democrat"),
        parse_response_list.count("republican"),
        parse_response_list.count("independent"),
        parse_response_list.count("other"),
        parse_response_list.count("prefer_not_to_answer"),
        parse_response_list.count("non_compliant"),
    ]
    result_dict["political_affiliation_option_order"] = (
        "democrat_first" if democrat_first else "republican_first"
    )
    result_dict["trajectory"] = prompt


@hydra.main(config_path=str(config_path), config_name="political_affiliation_config")
def my_app(cfg):
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

    survey_cfg = cfg.questionnaire.party_affiliation

    # Set random seed
    set_random_seed(cfg.random_seed)

    # Load backstory data and demography survey results
    backstory_df = load_backstory(
        backstory_path=pathlib.Path(__file__).parent.parent
        / cfg.backstories.backstory_path,
        backstory_type=cfg.backstories.backstory_type,
        num_backstory=cfg.backstories.num_backstories,
    )
    log.info(f"The number of backstories: {len(backstory_df)}")

    # config llm
    llm_cfg = cfg.llm_parameters
    model = prepare_llm(
        api_provider=llm_cfg.api_provider,
        model_name=llm_cfg.model_name,
    )

    OmegaConf.set_struct(cfg, False)
    llm_cfg.model_name = model
    log.info(f"Model: {model}")
    model_name = model.split("/")[-1]
    file_name = f"political_affiliation_{model_name}_{cfg.output_data_name}.pkl"
    output_data_path = pathlib.Path(cfg.run_dir) / file_name
    cfg.output_data_path = output_data_path
    log.info(f"Output data path: {output_data_path}")
    OmegaConf.set_struct(cfg, True)

    survey_cfg = cfg.questionnaire.party_affiliation

    # config output data path
    output_dict = {}

    parallel_survey(
        f=surveyor_wrapper,
        parallel_case_ids=range(len(backstory_df)),
        backstory_df=backstory_df,
        survey_cfg=survey_cfg,
        llm_cfg=llm_cfg,
        model=model,
        cfg=cfg,
        output_dict=output_dict,
        debug=cfg.debug,
    )

    save_result_to_pkl(output_dict, cfg.output_data_path)
    if not cfg.debug:
        publish_result(
            output=output_dict,
            publish_dir=cfg.save_dir,
            filename=file_name,
        )


if __name__ == "__main__":
    my_app()
