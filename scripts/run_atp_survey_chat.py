import logging
import pathlib
from threading import Lock
from typing import Type
from multiprocessing import pool

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
from anthology.lm_inference.llm import (
    prepare_llm,
    prompt_llm_chat_with_messages,
)
from anthology.survey.atp_survey import format_survey, response_parser
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
    output_result = run_ATP_study(
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


def run_ATP_study(
    case_id: int,
    backstory_df: pd.DataFrame,
    survey_cfg: Type[DictConfig],
    llm_cfg: Type[DictConfig],
    cfg: Type[DictConfig],
    model: str,
    debug: bool = False,
) -> dict:
    log.info(f"Running case_id: {case_id}")

    survey_questions, agree_first = format_survey(
        questions=survey_cfg.questions,
        consistency_prompt=survey_cfg.consistency_prompt,
        randomize_option_order=survey_cfg.randomize_option_order,
        randomize_question_order=(
            survey_cfg.randomize_question_order and survey_cfg.in_series
        ),
        is_empty_prompt=False,
        include_answer_forcing=survey_cfg.include_answer_forcing,
        in_series=survey_cfg.in_series,
    )

    temperature = llm_cfg.temperature
    max_tokens = llm_cfg.max_tokens
    top_p = llm_cfg.top_p

    # user_question = backstory_df.input_prompt.split("Question:")[-1].strip().split("Answer:")[0].strip()
    user_question = backstory_df.full_passage.strip()
    # assistant_backstory = backstory_df.backstory.strip()
    trajectory = [
        {
            "role": "system",
            "content": "Answer the following questions as if you are the person in the conversation.",
        },
        {"role": "user", "content": user_question},
        # {"role": "assistant", "content": assistant_backstory},
    ]

    result_dict = {}
    result_dict["temperature"] = temperature
    result_dict["max_tokens"] = max_tokens
    result_dict["top_p"] = top_p
    result_dict["model"] = model
    result_dict["backstory"] = user_question

    compliance_forcing_index = 0
    for qid, question in survey_questions.items():
        is_compliance = 0
        compliance_retry_count = 0
        classification_result = ""

        trajectory.append({"role": "user", "content": question})

        while is_compliance == 0:
            answer = prompt_llm_chat_with_messages(
                model_name=model,
                messages=trajectory,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=None,
                stop=["\n", "Question:"],
            ).message.content

            parsed_response, response_value = response_parser(
                response=answer,
                agree_first=agree_first,
                qid=qid,
            )
            if response_value < 0:
                classification_result = "non-compliant"
                is_compliance = 0
            else:
                classification_result = parsed_response
                is_compliance = 1
            is_compliance = 1

            if (
                compliance_forcing_index >= cfg.number_compliance_forcing
                or not cfg.include_compliance_forcing
                or compliance_retry_count >= cfg.number_compliance_forcing
            ):
                break
            compliance_retry_count += 1

        compliance_forcing_index += 1

        if debug:
            log.info(
                f"Answer {qid}: {answer}, Parsed response: {classification_result}, Response value: {response_value}"
            )

        question_and_answer = f"{question}{answer}".strip()

        trajectory.append({"role": "assistant", "content": answer})

        result_dict[f"{qid}_question"] = question
        result_dict[f"{qid}_answer"] = answer
        result_dict[f"{qid}_prompt"] = question_and_answer
        result_dict[f"{qid}_compliance"] = is_compliance
        result_dict[f"{qid}_classification"] = (
            question_and_answer + f"\nClassification: {classification_result}"
        )
        result_dict[f"{qid}"] = response_value

    result_dict["agree_first"] = agree_first
    result_dict["in_series"] = survey_cfg.in_series
    result_dict["trajectory"] = trajectory

    for key in backstory_df.keys():
        if key not in result_dict.keys():
            result_dict[f"{key}"] = backstory_df[key]

    return result_dict


@hydra.main(config_path=str(config_path), config_name="ATP_config")
def my_app(cfg):
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

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
    model_name = model.split("/")[-1]
    OmegaConf.set_struct(cfg, False)
    llm_cfg.model_name = model
    log.info(f"Model: {model}")

    output_data_path = (
        pathlib.Path(cfg.run_dir)
        / f"ATP_W{cfg.wave}_T_{llm_cfg.temperature}_{model_name}_{cfg.output_data_name}_{cfg.output_time_stamp}.pkl"
    )
    cfg.output_data_path = output_data_path
    log.info(f"Output data path: {output_data_path}")
    OmegaConf.set_struct(cfg, True)

    # run survey
    survey_cfg = cfg.questionnaire.ATP

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
            filename=f"ATP_W{cfg.wave}_T_{llm_cfg.temperature}_{model_name}_{cfg.output_data_name}_{cfg.output_time_stamp}.pkl",
        )


if __name__ == "__main__":
    my_app()
