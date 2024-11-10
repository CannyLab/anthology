import logging
import pathlib
import hydra
import random
from threading import Lock

from omegaconf import OmegaConf


from anthology.utils.random_utils import set_random_seed
from anthology.backstory.backstory import load_backstory
from anthology.demographic_survey.demography_survey import generate_mcq
from anthology.lm_inference.llm import (
    prepare_llm,
    prompt_llm,
    parallel_generator,
    prompt_llm_chat,
)
from anthology.utils.data_utils import (
    save_result_to_pkl,
    save_result_to_csv,
    publish_result,
    get_config_path,
)
from anthology.demographic_survey.response_parser import (
    demographics_survey_response_parser,
    llm_parsing_response_parse,
    choices_to_distribution,
)

log = logging.getLogger(__name__)

# for debugging...

# Multiprocessing
lock = Lock()

# config file path
config_path = get_config_path("surveys")


def prepare_questions(cfg: OmegaConf):
    demographic_questions = {}
    llm_as_parser_questions = {}

    demographic_traits = cfg.questionnaire.demographics
    mcq_symbol = cfg.format.mcq_symbol
    choice_format = cfg.format.choice_format
    surveyor_respondent = cfg.format.surveyor_respondent

    for key, value in demographic_traits.items():
        log.info("Formatting question: {}<\end>".format(key))
        if value.type == "multiple_choice":
            assert hasattr(
                value, "choices"
            ), "Multiple choice question must have choices"
            formatted_demographic_question = generate_mcq(
                question=value.question,
                choices=value.choices,
                choice_format=choice_format,
                mcq_symbol=mcq_symbol.symbol,
                surveyor_respondent=surveyor_respondent,
            )
            # change the final answer choice to "Was not mentioned"
            parsing_question_choices = value.choices[:-1]
            parsing_question_choices.append("Was not mentioned")
            formatted_llm_parsing_question = generate_mcq(
                question=value.llm_parser_question,
                choices=parsing_question_choices,
                choice_format=choice_format,
                mcq_symbol=mcq_symbol.symbol,
                surveyor_respondent=surveyor_respondent,
            )

            demographic_questions[key] = formatted_demographic_question
            llm_as_parser_questions[key] = formatted_llm_parsing_question

            if cfg.debug:
                log.info(
                    "Formatted question:\n{}<\end>".format(
                        formatted_demographic_question
                    )
                )
    return demographic_questions, llm_as_parser_questions


def generator_wrapper(case_id: int, cfg: dict, output_dict: OmegaConf):
    result_dict = perform_demography_survey(case_id, cfg=cfg)
    with lock:
        output_dict[case_id] = result_dict

        # Save generated_backstory to csv every freq cases
        if (case_id + 1) % cfg.freq == 0:
            csv_output_path = str(cfg.output_data_path).replace(".pkl", ".csv")
            csv_output_path = pathlib.Path(csv_output_path)

            save_result_to_csv(output_dict, csv_output_path)
            save_result_to_pkl(output_dict, cfg.output_data_path)


def survey_demographics(
    backstory: str,
    cfg: OmegaConf,
) -> dict:
    result_dict = {}
    # Preprocessed {demographic trait: question_string} dictionaries
    demographic_questions = cfg.demographic_questions
    llm_as_parser_questions = cfg.llm_as_parser_questions
    # llm parameters
    temperature = cfg.llm_parameters.temperature
    max_tokens = cfg.llm_parameters.max_tokens
    top_p = cfg.llm_parameters.top_p
    model_name = cfg.llm_parameters.model_name
    special_prompt = cfg.special_prompt
    model_name = cfg.llm_parameters.model_name

    if cfg.debug:
        log.info(f"Backstory: {backstory}")
    # perform survey for each demographic trait
    for trait, question in demographic_questions.items():
        # add the backstory to the question
        survey_formatted_question = f"{backstory.strip()}\n\n{question}"
        if cfg.debug:
            log.info(f"Survey Trait: {trait}")
            log.info(f"Survey Question: {question}")
        num_choices = len(cfg.questionnaire.demographics[trait].choices)
        if cfg.use_llm_as_parser and trait not in ["race", "gender", "religion"]:
            # Case (1) Using LLM as parser
            # use llm to parse the responses
            llm_parser_prompt = (
                (
                    backstory.replace(cfg.format.surveyor_respondent.surveyor, "Essay")
                ).strip()
                + "\n\n"
                + llm_as_parser_questions[trait]
            )
            if cfg.debug:
                log.info(f"llm_parser_prompt: {llm_parser_prompt}")

            llm_parser_response_text = prompt_llm_chat(
                system_prompt="Please select the best answer from the following choices:",
                user_prompt=llm_parser_prompt,
                max_tokens=cfg.llm_parsing_parameters.max_tokens,
                temperature=cfg.llm_parsing_parameters.temperature,
                stop=[f"{cfg.format.surveyor_respondent.surveyor}"],
                model_name=cfg.llm_parsing_parameters.model_name,
                top_p=cfg.llm_parsing_parameters.top_p,
                logprobs=0,
            )
            if cfg.debug:
                log.info(f"llm_parser_response_text: {llm_parser_response_text}")
            # parse the llm-as-parser response
            # parsed_choice: int - the choice index
            llm_parser_parsed_choice = llm_parsing_response_parse(
                response=llm_parser_response_text,
                question_cfg=cfg.questionnaire.demographics[trait],
            )
            if cfg.debug:
                log.info(f"llm_parser_parsed_choice: {llm_parser_parsed_choice}")
            # if reponse is "Was not mentioned"
            # then we need to adminster actual demographics survey to base model
            if llm_parser_parsed_choice >= num_choices - 1:
                # Case (1-1) Using LLM as parser but the parsing result is "Was not mentioned" or noncompliant answer
                # need to swap back to survey model (not the chat model for parsing)
                if cfg.debug:
                    log.info(f"Need to administer survey for this trait: {trait}")

                if "together" in cfg.llm_parameters.api_provider:
                    survey_responses = []
                    for _ in range(cfg.num_sample_response // 5):
                        survey_responses.extend(
                            prompt_llm(
                                prompt=survey_formatted_question,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                stop=[
                                    "\n",
                                    f"{cfg.format.surveyor_respondent.surveyor}",
                                ],
                                model_name=model_name,
                                top_p=top_p,
                                echo=False,
                                logprobs=0,
                                n=5,
                            )
                        )
                else:
                    survey_responses = prompt_llm(
                        prompt=survey_formatted_question,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["\n", f"{cfg.format.surveyor_respondent.surveyor}"],
                        model_name=model_name,
                        top_p=top_p,
                        echo=False,
                        logprobs=0,
                        n=cfg.num_sample_response,
                    )
                survey_response_texts = [response.text for response in survey_responses]
                if cfg.debug:
                    log.info(f"survey_response_texts: {survey_response_texts}")
                # parse the survey responses
                # survey_responses: list of strings

                parsed_survey_choices = []
                for response in survey_response_texts:
                    parsed_choice = demographics_survey_response_parser(
                        response=response,
                        question_name=trait,
                        question_cfg=cfg.questionnaire.demographics[trait],
                    )
                    parsed_survey_choices.append(parsed_choice["choice"])
                if cfg.debug:
                    log.info(f"parsed_survey_choices: {parsed_survey_choices}")
                count_noncompliant = parsed_survey_choices.count(num_choices)
                # convert the parsed choices to distribution
                response_distribution = choices_to_distribution(
                    choices_list=parsed_survey_choices,
                    num_options=len(cfg.questionnaire.demographics[trait].choices),
                )
                response_texts = ";".join(
                    [response.text for response in survey_responses]
                )
                # add the result to the result dict
                result_dict[f"{trait}_is_llm_parsing_result"] = False
                result_dict[f"{trait}_is_survey_result"] = True
            else:
                # Case (1-2) Using LLM as parser and the parsing result was not "Was not mentioned"
                # no need to administer the survey for this trait
                count_noncompliant = 0
                parsed_survey_choices = [llm_parser_parsed_choice]
                response_distribution = choices_to_distribution(
                    choices_list=[llm_parser_parsed_choice],
                    num_options=len(cfg.questionnaire.demographics[trait].choices),
                )
                result_dict[f"{trait}_is_llm_parsing_result"] = True
                result_dict[f"{trait}_is_survey_result"] = False
                response_texts = llm_parser_response_text
            result_dict[f"{trait}_llm_parsing_question"] = llm_as_parser_questions[
                trait
            ]
            result_dict[f"{trait}_llm_parsing_prompt"] = llm_parser_prompt
            result_dict[f"{trait}_llm_parsing_response"] = llm_parser_response_text
            result_dict[f"{trait}_llm_parsing_parsed_choice"] = llm_parser_parsed_choice
        else:
            # Case (2) Not using LLM as parser
            result_dict[f"{trait}_is_llm_parsing_result"] = False
            result_dict[f"{trait}_is_survey_result"] = True
            result_dict[f"{trait}_llm_parsing_question"] = ""
            result_dict[f"{trait}_llm_parsing_prompt"] = ""
            result_dict[f"{trait}_llm_parsing_response"] = ""
            result_dict[f"{trait}_llm_parsing_parsed_choice"] = -1

            # survey_responses: list of OpenAI completion objects
            if "together" in cfg.llm_parameters.api_provider:
                survey_responses = []
                for _ in range(cfg.num_sample_response // 5):
                    survey_responses.extend(
                        prompt_llm(
                            prompt=survey_formatted_question,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=["\n", f"{cfg.format.surveyor_respondent.surveyor}"],
                            model_name=model_name,
                            top_p=top_p,
                            echo=False,
                            logprobs=0,
                            n=5,
                        )
                    )
            else:
                survey_responses = prompt_llm(
                    prompt=survey_formatted_question,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\n", f"{cfg.format.surveyor_respondent.surveyor}"],
                    model_name=model_name,
                    top_p=top_p,
                    echo=False,
                    logprobs=0,
                    n=cfg.num_sample_response,
                )
            # parse the survey responses
            # survey_responses: list of strings
            survey_response_texts = [response.text for response in survey_responses]
            if cfg.debug:
                log.info(f"survey_response_texts: {survey_response_texts}")
            parsed_survey_choices = []
            for response in survey_response_texts:
                parsed_choice = demographics_survey_response_parser(
                    response=response,
                    question_name=trait,
                    question_cfg=cfg.questionnaire.demographics[trait],
                )
                parsed_survey_choices.append(parsed_choice["choice"])
            if cfg.debug:
                log.info(f"parsed_survey_choices: {parsed_survey_choices}")
            # count the number of noncompliant responses
            # that are indicated as 'num_choices' in the parsed_survey_choices
            count_noncompliant = parsed_survey_choices.count(num_choices)
            response_distribution = choices_to_distribution(
                choices_list=parsed_survey_choices,
                num_options=len(cfg.questionnaire.demographics[trait].choices),
            )
            response_texts = ";".join([response.text for response in survey_responses])
            # add the result to the result dict

        result_dict[f"{trait}_survey_question"] = question
        result_dict[f"{trait}_survey_prompt"] = survey_formatted_question
        result_dict[f"{trait}_survey_choices"] = parsed_survey_choices
        result_dict[f"{trait}_response_distribution"] = response_distribution
        result_dict[f"{trait}_response_texts"] = response_texts
        result_dict[f"{trait}_count_noncompliant"] = count_noncompliant

    # add metadata to the result dict
    result_dict["backstory"] = backstory
    result_dict["special_prompt"] = [x for x in special_prompt]
    result_dict["temperature"] = temperature
    result_dict["max_tokens"] = max_tokens
    result_dict["top_p"] = top_p
    result_dict["model_name"] = model_name
    return result_dict


def perform_demography_survey(case_id: int, cfg: OmegaConf):
    demographic_questions = cfg.demographic_questions

    log.info("Performing demography survey for user id: {}".format(case_id))
    log.info("Demographic questions: {}".format(demographic_questions.keys()))

    backstory = cfg.backstory_dict[case_id]
    assert isinstance(backstory, str), "Backstory must be a string"
    result_dict = {}
    result_dict["uid"] = case_id
    result_dict.update(
        survey_demographics(
            backstory=backstory,
            cfg=cfg,
        )
    )
    return result_dict


@hydra.main(config_path=str(config_path), config_name="demographics_config")
def my_app(cfg):
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Running LLM with config:\n{OmegaConf.to_yaml(cfg.llm_parameters)}")
    assert (
        cfg.num_sample_response % 5 == 0
    ), "num_sample_response must be a multiple of 5"
    # Set random seed
    set_random_seed(cfg.random_seed)

    # set values for global variables
    backstory_df = load_backstory(
        backstory_path=pathlib.Path(__file__).parent.parent
        / cfg.backstories.backstory_path,
        backstory_type=cfg.backstories.backstory_type,
        num_backstory=cfg.backstories.num_backstories,
    )
    log.info(f"The number of backstories: {len(backstory_df)}")

    # rename the columns for surveyor and respondent
    # This is needed because the column names are different in the different backstories files
    if "asker" in backstory_df.columns:
        # rename asker column to surveyor
        backstory_df.rename(columns={"asker": "surveyor"}, inplace=True)
    if "responder" in backstory_df.columns:
        # rename responder column to respondent
        backstory_df.rename(columns={"responder": "respondent"}, inplace=True)

    # check backstory_df has the columns of surveyor, respondent and full_passage
    # if so, overrides the surveyor and respondent in the config file
    # We must match the prompt format that generated the backstories
    # and the prompt format that we are using to generate the survey questions
    if "surveyor" in backstory_df.columns:
        cfg.format.surveyor_respondent.surveyor = backstory_df["surveyor"][0]
    if "respondent" in backstory_df.columns:
        cfg.format.surveyor_respondent.respondent = backstory_df["respondent"][0]
    if not "full_passage" in backstory_df.columns:
        raise ValueError("Backstory dataframe must have a column named 'full_passage'")

    # format the questions
    demographic_questions, llm_as_parser_questions = prepare_questions(cfg)

    # prepare the LLM model based on config
    llm_cfg = cfg.llm_parameters
    model = prepare_llm(
        api_provider=llm_cfg.api_provider,
        model_name=llm_cfg.model_name,
    )

    OmegaConf.set_struct(cfg, False)
    cfg.llm_parameters.model_name = model
    model_name = model.split("/")[-1]

    backstory_uids = [int(uid) for uid in backstory_df["uid"].values]
    if cfg.debug:
        backstory_uids = random.sample(backstory_uids, 10)
    # create dictionary of "uid" to "full_passage"
    backstory_dict = {}
    for uid in backstory_uids:
        backstory_dict[uid] = backstory_df[backstory_df["uid"] == uid][
            "full_passage"
        ].values[0]
    if cfg.debug:
        output_data_path = (
            pathlib.Path(cfg.run_dir)
            / f"{model_name}_demographics_survey_{cfg.output_data_name}_debug.pkl"
        )
    else:
        output_data_path = (
            pathlib.Path(cfg.run_dir)
            / f"{model_name}_demographics_survey_{cfg.output_data_name}.pkl"
        )
    log.info(f"Output data path: {output_data_path}")
    cfg.output_data_path = output_data_path
    cfg.backstory_dict = backstory_dict
    cfg.demographic_questions = demographic_questions
    cfg.llm_as_parser_questions = llm_as_parser_questions
    OmegaConf.set_struct(cfg, True)

    # pass backstory uids as case ids
    output_dict = {}
    parallel_generator(
        f=generator_wrapper,
        parallel_case_ids=backstory_uids,
        cfg=cfg,
        output_dict=output_dict,
    )
    save_result_to_pkl(output_dict, output_data_path)

    if not cfg.debug:
        publish_result(
            output=output_dict,
            publish_dir=cfg.save_dir,
            filename=f"{model_name}_demographics_survey_{cfg.output_data_name}.pkl",
        )


if __name__ == "__main__":
    my_app()
