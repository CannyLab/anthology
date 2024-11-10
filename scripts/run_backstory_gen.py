import logging
import pathlib
import hydra
from threading import Lock

from omegaconf import OmegaConf

from anthology.lm_inference.llm import (
    prepare_llm,
    get_tokenizer,
    prompt_llm,
    parallel_generator,
)
from anthology.utils.data_utils import save_result_to_csv, publish_result, get_config_path

log = logging.getLogger(__name__)

# Multiprocessing
lock = Lock()

# config file path
config_path = get_config_path("generate_backstory")


def generator_wrapper(case_id: int, cfg: dict, output_dict: dict, tokenizer):
    generated_backstory = generate_backstory(case_id, cfg, tokenizer)
    if cfg.debug:
        log.info(f"Generation result:\n{generated_backstory}")
    with lock:
        output_dict[case_id] = generated_backstory

        # Save generated_backstory to csv every freq cases
        if (case_id + 1) % cfg.freq == 0:
            save_result_to_csv(output_dict, cfg.output_data_path)


def generate_backstory(backstory_idx: int, cfg: dict, tokenizer):
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

    preamble_question = cfg.self_generate.preamble_question
    backstory_question = cfg.self_generate.backstory_question
    surveyor = cfg.self_generate.surveyor
    respondent = cfg.self_generate.respondent
    connector = cfg.self_generate.connector

    temperature = cfg.llm_parameters.temperature
    max_tokens = cfg.llm_parameters.max_tokens
    top_p = cfg.llm_parameters.top_p
    model_name = cfg.llm_parameters.model_name

    # Generate the preamble answer maybe mostly asking if the respondent is a US citizen
    if preamble_question:
        preamble_prompt = (
            f"{surveyor}{connector} {preamble_question}\n\n{respondent}{connector}"
        )
        prompt += preamble_prompt

        if cfg.debug:
            log.info(f"Prompt: {prompt}")

        while True:
            preamble_answer_result = prompt_llm(
                prompt=preamble_prompt,
                max_tokens=100,
                temperature=temperature,
                stop=["\n", f"{surveyor}"],
                model_name=model_name,
                top_p=top_p,
                echo=False,
                logprobs=1,
            )

            if "yes" in preamble_answer_result.text.lower():
                preamble_answer = preamble_answer_result.text
                preamble_answer_logprobs = sum(
                    preamble_answer_result.logprobs.token_logprobs
                )
                preamble_answer_token_length = len(tokenizer.encode(preamble_answer))

                prompt += preamble_answer
                break
        backstory_question = (
            f"\n\n{surveyor}{connector} {backstory_question}\n\n{respondent}{connector}"
        )
        prompt += backstory_question
    else:
        prompt = (
            f"{surveyor}{connector} {backstory_question}\n\n{respondent}{connector}"
        )

    if cfg.debug:
        log.info(f"Prompt: {prompt}")

    # Generate the backstory
    generated_backstory_result = prompt_llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[f"{surveyor}", "Q:", "Interview"],
        model_name=model_name,
        top_p=top_p,
        echo=False,
        logprobs=1,
    )

    generated_backstory = generated_backstory_result.text
    backstory_likelihood = sum(generated_backstory_result.logprobs.token_logprobs)

    if cfg.debug:
        log.info(f"Likelihood of generated backstory: {backstory_likelihood}")
        log.info(f"Generated Backstory: {generated_backstory}")

    return {
        "full_passage": f"{surveyor}{connector} {backstory_question}\n\n{respondent}{connector}"
        + generated_backstory,
        "backstory": generated_backstory,
        "input_prompt": prompt,
        "preamble_prompt": preamble_prompt,
        "surveyor": surveyor,
        "respondent": respondent,
        "backstory_generated_token_length": len(tokenizer.encode(generated_backstory)),
        "backstory_logprob": backstory_likelihood,
        "total_generated_token_length": len(tokenizer.encode(generated_backstory)),
        "preamble_answer": preamble_answer,
        "preamble_answer_logprobs": preamble_answer_logprobs,
        "preamble_answer_token_length": preamble_answer_token_length,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "finish_reason": generated_backstory_result.finish_reason,
    }


@hydra.main(config_path=str(config_path), config_name="config")
def my_app(cfg):
    output_dict = {}
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Running LLM with config:\n{OmegaConf.to_yaml(cfg.llm_parameters)}")

    llm_cfg = cfg.llm_parameters
    model = prepare_llm(
        api_provider=llm_cfg.api_provider,
        model_name=llm_cfg.model_name,
    )
    model_name = model.split("/")[-1]

    OmegaConf.set_struct(cfg, False)
    cfg.llm_parameters.model_name = model
    tokenizer = get_tokenizer(cfg.llm_parameters.model_name)
    output_data_path = (
        pathlib.Path(cfg.run_dir)
        / f"{model_name}_generated_backstory_{cfg.output_data_name}.csv"
    )
    log.info(f"Output data path: {output_data_path}")
    cfg.output_data_path = output_data_path
    OmegaConf.set_struct(cfg, True)

    parallel_generator(
        f=generator_wrapper,
        parallel_case_ids=range(cfg.num_backstories),
        cfg=cfg,
        output_dict=output_dict,
        tokenizer=tokenizer,
    )
    save_result_to_csv(output_dict, output_data_path)

    if not cfg.debug:
        publish_result(
            output=output_dict,
            publish_dir=cfg.save_dir,
            filename=f"{model_name}_generated_backstory_{cfg.output_data_name}.csv",
        )


if __name__ == "__main__":
    my_app()
