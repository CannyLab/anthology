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
from anthology.lm_inference.llm import prepare_llm, prompt_llm
from anthology.backstory.backstory import load_backstory

log = logging.getLogger(__name__)

# Multiprocessing
lock = Lock()

# config file path
config_path = get_config_path("analysis")


def parallel_compute_perplexity(
    f,
    parallel_case_ids: list,
    backstory_df: pd.DataFrame,
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


def compute_perplexity_wrapper(
    case_id: int,
    backstory_df: pd.DataFrame,
    cfg: Type[DictConfig],
    output_dict: dict,
    model: str,
    debug: bool = False,
):
    output_result = compute_backstory_perplexity(
        case_id=case_id,
        backstory_df=backstory_df.iloc[case_id],
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


def compute_backstory_perplexity(
    case_id: int,
    backstory_df: pd.DataFrame,
    model: str,
) -> dict:
    log.info(f"Running case_id: {case_id}")

    max_tokens = 1

    backstory = backstory_df.backstory.strip()

    result_dict = {}
    result_dict["max_tokens"] = max_tokens
    result_dict["model"] = model
    result_dict["backstory"] = backstory

    answer = prompt_llm(
        model_name=model,
        prompt=backstory,
        max_tokens=max_tokens,
        logprobs=1,
        echo=True,
    )

    logprobs = answer.logprobs.token_logprobs[:-1]  # remove the generated token
    token_length = len(logprobs)

    result_dict["logprobs"] = logprobs
    result_dict["token_length"] = token_length

    for key in backstory_df.keys():
        if key not in result_dict.keys():
            result_dict[f"{key}"] = backstory_df[key]

    return result_dict


@hydra.main(config_path=str(config_path), config_name="backstory_perplexity")
def my_app(cfg):
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    set_random_seed(cfg.random_seed)

    # Load backstory data
    backstory_df = load_backstory(
        backstory_path=pathlib.Path(__file__).parent.parent
        / cfg.backstories.backstory_path,
        backstory_type=cfg.backstories.backstory_type,
        num_backstory=cfg.backstories.num_backstories,
    )
    log.info(f"The number of backstories: {len(backstory_df)}")

    if cfg.debug:
        # randomly sample 10 backstories
        cfg.num_processes = 1
        backstory_df = backstory_df.sample(10)
        log.info(f"Debug mode: The number of backstories: {len(backstory_df)}")

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

    filename = f"{model_name}_{cfg.output_data_name}_{cfg.output_time_stamp}.pkl"
    output_data_path = pathlib.Path(cfg.run_dir) / filename
    cfg.output_data_path = output_data_path
    log.info(f"Output data path: {output_data_path}")
    OmegaConf.set_struct(cfg, True)

    # config output data path
    output_dict = {}

    parallel_compute_perplexity(
        f=compute_perplexity_wrapper,
        parallel_case_ids=range(len(backstory_df)),
        backstory_df=backstory_df,
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
            filename=filename,
        )


if __name__ == "__main__":
    my_app()
