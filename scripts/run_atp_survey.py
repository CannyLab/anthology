import logging
import pathlib
from threading import Lock
from typing import Type
from multiprocessing import pool

import pyreadstat
import hydra
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from anthology.utils.random_utils import set_random_seed
from anthology.utils.data_utils import (
    save_result_to_pkl,
    publish_result,
    save_result_to_csv,
    get_config_path,
)
from anthology.lm_inference.llm import prepare_llm, prompt_llm
from anthology.survey.atp_survey import format_survey, response_parser
from anthology.survey.atp_baseline_survey import generate_baseline_prompt_prefix
from anthology.backstory.backstory import load_backstory

log = logging.getLogger(__name__)

# Multiprocessing
lock = Lock()

# config file path
config_path = get_config_path("surveys")


def filter_demographic_variable(
    input: dict,
    trait_of_interest: list,
) -> dict:
    return {k: v for k, v in input.items() if input[k]["topic"] in trait_of_interest}


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
    output_result = run_ATP_survey(
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


def run_ATP_survey(
    case_id: int,
    backstory_df: pd.DataFrame,
    survey_cfg: Type[DictConfig],
    llm_cfg: Type[DictConfig],
    cfg: Type[DictConfig],
    model: str,
    debug: bool = False,
    prompt_style: str = None,
) -> dict:
    log.info(f"Running case_id: {case_id}")

    survey_questions = format_survey(
        questions=survey_cfg.questions,
        consistency_prompt=survey_cfg.consistency_prompt,
        randomize_option_order=survey_cfg.randomize_option_order,
        randomize_question_order=(
            survey_cfg.randomize_question_order and survey_cfg.in_series
        ),
        is_empty_prompt=(True if prompt_style == "nothing" else False),
        include_answer_forcing=survey_cfg.include_answer_forcing,
        in_series=survey_cfg.in_series,
    )

    temperature = llm_cfg.temperature
    max_tokens = llm_cfg.max_tokens
    top_p = llm_cfg.top_p

    backstory = backstory_df.full_passage.strip()
    trajectory = backstory
    prompt = backstory

    result_dict = {}
    result_dict["temperature"] = temperature
    result_dict["max_tokens"] = max_tokens
    result_dict["top_p"] = top_p
    result_dict["model"] = model
    result_dict["backstory"] = backstory.strip()

    for qid, question_info in survey_questions.items():
        question = question_info["question"]
        randomization_pattern = question_info["randomization_pattern"]
        is_compliance = 0
        compliance_retry_count = 0
        classification_result = ""

        if survey_cfg.in_series:
            prompt += f"\n\n{question}"
        else:
            prompt = f"{backstory}\n\n{question}"
        trajectory += f"\n\n{question}"

        while is_compliance == 0:
            answer = prompt_llm(
                model_name=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=None,
                stop=["\n", "Question:"],
            ).text

            parsed_response, response_value = response_parser(
                response=answer,
                randomization_pattern=randomization_pattern,
                qid=qid,
            )
            if response_value < 0:
                classification_result = "non-compliant"
                is_compliance = 0
            else:
                classification_result = parsed_response
                is_compliance = 1

            if (
                not cfg.include_compliance_forcing
                or compliance_retry_count >= cfg.number_compliance_forcing
            ):
                break
            compliance_retry_count += 1

        if debug:
            log.info(
                f"Answer {qid}: {answer}, Parsed response: {classification_result}, Response value: {response_value}"
            )

        question_and_answer = f"{question}{answer}".strip()

        if survey_cfg.in_series:
            prompt += f"{answer}"
            prompt = prompt.strip()

        trajectory += f"{answer}"
        trajectory = trajectory.strip()

        result_dict[f"{qid}_question"] = question
        result_dict[f"{qid}_answer"] = answer
        result_dict[f"{qid}_prompt"] = question_and_answer
        result_dict[f"{qid}_compliance"] = is_compliance
        result_dict[f"{qid}_classification"] = (
            question_and_answer + f"\nClassification: {classification_result}"
        )
        result_dict[f"{qid}"] = response_value
        result_dict[f"{qid}_randomization_pattern"] = randomization_pattern

    result_dict["in_series"] = survey_cfg.in_series
    result_dict["trajectory"] = trajectory

    for key in backstory_df.keys():
        if key not in result_dict.keys():
            result_dict[f"{key}"] = backstory_df[key]

    return result_dict


@hydra.main(config_path=str(config_path), config_name="ATP_config")
def my_app(cfg):
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

    set_random_seed(cfg.random_seed)
    survey_cfg = cfg.questionnaire.ATP
    llm_cfg = cfg.llm_parameters
    model = prepare_llm(
        api_provider=llm_cfg.api_provider,
        model_name=llm_cfg.model_name,
    )
    model_name = model.split("/")[-1]
    OmegaConf.set_struct(cfg, False)
    llm_cfg.model_name = model
    log.info(f"Model: {model}")

    if cfg.is_baseline:
        log.info("Running ATP baseline survey")
        cfg.save_dir = cfg.save_dir + "_baseline"
        cfg.output_data_name = f"ATP_W{cfg.wave}_baseline"
        atp_data_df, _ = pyreadstat.read_sav(cfg.survey_data_path)
        atp_demographics_df = pd.read_json(cfg.survey_demographics_metadata_path)
        atp_demographics_df = pd.DataFrame(
            filter_demographic_variable(
                input=atp_demographics_df,
                trait_of_interest=list(cfg.trait_of_interest),
            )
        )
        backstory_df = pd.DataFrame(
            generate_baseline_prompt_prefix(
                survey_data_df=atp_data_df,
                demographics_df=atp_demographics_df,
                prompt_style=cfg.prompt_style,
                include_answer_forcing=cfg.include_answer_forcing,
            )
        )
        log.info(
            f"Number of human respondents in {cfg.survey_data_path}: {len(backstory_df)}"
        )
        file_name = f"{cfg.output_data_name}_T_{llm_cfg.temperature}_{cfg.prompt_style}_{model_name}_{cfg.output_time_stamp}.pkl"
    else:
        log.info("Running ATP survey")
        backstory_df = load_backstory(
            backstory_path=pathlib.Path(__file__).parent.parent
            / cfg.backstories.backstory_path,
            backstory_type=cfg.backstories.backstory_type,
            num_backstory=cfg.backstories.num_backstories,
        )
        log.info(f"The number of backstories: {len(backstory_df)}")
        file_name = f"ATP_W{cfg.wave}_T_{llm_cfg.temperature}_{model_name}_{cfg.output_data_name}_{cfg.output_time_stamp}.pkl"

    cfg.output_data_path = pathlib.Path(cfg.run_dir) / file_name
    log.info(f"Output data path: {cfg.output_data_path}")
    log.info(f"Final output data path: {cfg.save_dir}")
    OmegaConf.set_struct(cfg, True)

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
