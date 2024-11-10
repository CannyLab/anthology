import re
import numpy as np

from math import exp

from collections import defaultdict


from .demographics_regex_patterns import (
    PATTERN_LIST_DICT,
    PATTERN_CASE_SENSITIVITY_DICT,
    NUMBER_RANGE_DICT,
)
from .response_parse_utils import *


def preprocess_numerical_response(
    response: str,
    question_name: str,
    question_cfg: dict,
) -> str:
    if not question_cfg.is_numerical:
        return response
    if question_name == "income":
        response = re.sub(
            PATTERN_MILLION,
            multiply_by_million_and_remove_commas,
            response,
        )
        response = re.sub(
            PATTERN_THOUSAND,
            multiply_by_thousand_and_remove_commas,
            response,
        )
    return response


####################################################################
# response classification with the regular expression parsing method
def numerical_response_parser(
    response: str,
    question_name: str,
    question_cfg: dict,
) -> dict:
    # preprocess the response
    response = preprocess_numerical_response(
        response=response,
        question_name=question_name,
        question_cfg=question_cfg,
    )
    num_choices = len(question_cfg.choices)
    pattern_list = PATTERN_LIST_DICT[question_name]
    case_sensitivity_list = PATTERN_CASE_SENSITIVITY_DICT[question_name]
    num_pattern = len(pattern_list)
    if question_name == "income":
        try:
            response = re.sub(
                PATTERN_MILLION,
                multiply_by_million_and_remove_commas,
                response,
            )
            response = re.sub(
                PATTERN_THOUSAND,
                multiply_by_thousand_and_remove_commas,
                response,
            )
        except:
            pass
    # parse all regex patterns in the response completion.
    pattern_parsed = [[] for i in range(num_pattern)]
    for i in range(num_pattern):
        pattern_parsed[i] = re.findall(
            pattern_list[i],
            response,
            re.IGNORECASE if not case_sensitivity_list[i] else 0,
        )
    pattern_appear_count = [len(pattern_parsed[j]) for j in range(num_pattern)]
    # Let's classify based on the parse result.
    # Rule: if the label pattern [ex."A)"] exists
    #           if multiple label patterns exist, classify as ambiguous
    #           if only one label pattern exists, classify based on the label
    #       elif prefer-not exists
    #           if no number pattern exists, classify as prefer-not
    #           if number pattern also exists, classify as ambiguous
    #       elif number pattern [ex."25"] exists
    #           if multiple number patterns exist [ex."25, 50"], classify as ambiguous
    #           if only one number pattern exists, classify based on the number
    #       else
    #           classify as noncompliant
    #       if ambiguous and allow_llm, call llm for classification
    complete = False
    is_ambiguous = False
    choice = -1
    if pattern_appear_count[-1] > 0:
        if pattern_appear_count[-1] == 1:
            choice = label_to_index_mapper(
                letter=pattern_parsed[-1][0][0],
                num_choices=num_choices,
            )
            if choice > -1:
                complete = True
        else:
            is_ambiguous = True
            complete = True
    if complete == False and pattern_appear_count[-2] > 0:
        if max(pattern_appear_count[:-2]) > 0:
            is_ambiguous = True
        else:
            choice = num_choices - 1
        complete = True
    if complete == False and max(pattern_appear_count[:-2]) > 0:
        collect_all_pattern = []
        for j in range(num_pattern - 2):
            for k in range(pattern_appear_count[j]):
                get_number = re.findall(
                    PATTERN_NUMBER,
                    pattern_parsed[j][k],
                )
                try:
                    temp = float(get_number[0].replace(",", ""))
                except:
                    return {"choice": num_choices, "is_ambiguous": True}
                over_or_under = 1 if j == 3 else -1 if j == 2 else 0
                collect_all_pattern.append(
                    number_to_index_mapper(
                        number=temp + over_or_under,
                        question_cfg=question_cfg,
                        number_range_list=NUMBER_RANGE_DICT[question_name],
                    )
                )
        collect_all_pattern = list(set(collect_all_pattern))
        if len(collect_all_pattern) == 1:
            choice = collect_all_pattern[0]
        else:
            is_ambiguous = True
        complete = True
    if complete == False or is_ambiguous:
        choice = num_choices
    return_dict = {
        "choice": choice,
        "is_ambiguous": is_ambiguous,
    }
    return return_dict


def string_response_parser(
    response: str,
    question_name: str,
    question_cfg: dict,
) -> dict:
    assert not question_cfg.is_numerical
    num_choices = len(question_cfg.choices)
    pattern_list = PATTERN_LIST_DICT[question_name]
    case_sensitivity_list = PATTERN_CASE_SENSITIVITY_DICT[question_name]
    num_pattern = len(pattern_list)
    # parse all regex patterns in the response completion.
    pattern_parsed = [[] for i in range(num_pattern)]
    for i in range(num_pattern):
        pattern_parsed[i] = re.findall(
            pattern_list[i],
            response,
            re.IGNORECASE if not case_sensitivity_list[i] else 0,
        )
    pattern_appear_count = [len(pattern_parsed[j]) for j in range(num_pattern)]
    # Let's classify based on the parse result.
    # Rule: if the label pattern [ex."A)"] exists
    #           if multiple label patterns exist, classify as ambiguous
    #           if only one label pattern exists, classify based on the label
    #       elif string pattern [ex."Asian"] exists
    #           if multiple string patterns exist [ex."High school graduate, but also doctor"], classify as ambiguous
    #           if only one string pattern exists, classify based on the string
    #       else
    #           classify as noncompliant
    #       if ambiguous and allow_llm, call llm for classification
    complete = False
    is_ambiguous = False
    choice = -1
    if pattern_appear_count[-1] > 0:
        if pattern_appear_count[-1] == 1:
            choice = label_to_index_mapper(
                letter=pattern_parsed[-1][0][0],
                num_choices=num_choices,
            )
            if choice > -1:
                complete = True
        else:
            is_ambiguous = True
            complete = True
    if complete == False and max(pattern_appear_count[:num_choices]) > 0:
        if sum(pattern_appear_count[:num_choices]) == max(
            pattern_appear_count[:num_choices]
        ):
            choice = np.argmax(pattern_appear_count[:num_choices])
        else:
            is_ambiguous = True
        complete = True
    if complete == False or is_ambiguous:
        choice = num_choices

    return_dict = {
        "choice": choice,
        "is_ambiguous": is_ambiguous,
    }
    return return_dict


def choices_to_distribution(
    choices_list: list,
    num_options: int,
) -> list:
    bucket = [0 for x in range(num_options + 1)]
    for x in choices_list:
        if x < 0 or x >= num_options:
            bucket[num_options] += 1
        else:
            bucket[x] += 1
    return bucket


"""
Top-level regex-based LLM response paraser
this parser applies to  
    - If the question is numerical, call numerical_response_parser
    - If the question is not numerical, call string_response_parser

    Args:
        response: str: the response to be parsed
        question_cfg: dict: the question_cfg dictionary
    Return:
        dict: the classification result
"""


def demographics_survey_response_parser(
    response: str,
    question_name: str,
    question_cfg: dict,
) -> dict:
    if question_cfg.is_numerical:
        return numerical_response_parser(
            response=response,
            question_name=question_name,
            question_cfg=question_cfg,
        )
    else:
        return string_response_parser(
            response=response,
            question_name=question_name,
            question_cfg=question_cfg,
        )


"""
Top-level function to parse the string output from the LLM-as-a-parser: 
    Args:
        response: str: the response to be parsed
        question_cfg: dict: the question_cfg dictionary
    Return:
        choice: int, the choice index
"""


def llm_parsing_response_parse(response: str, question_cfg: dict) -> int:
    pattern = re.findall(
        r"[A-Z]\)",
        response,
    )
    num_choices = len(question_cfg.choices)
    if len(pattern) > 0:
        pattern = pattern[-1][0]
        choice = label_to_index_mapper(
            letter=pattern,
            num_choices=num_choices,
        )
    else:
        choice = num_choices
    return choice


def demographic_logprob_parser(
    logprobs: dict,
    num_choices: int,
) -> dict:
    result_dict = {}

    for i in range(num_choices):
        key = chr(i + 65)
        result_dict[key] = 0.0

    for logprob in logprobs:
        token = logprob.token

        # get logprobs of capital alphabet letters A, B, C, ...
        # or (A, (B, (C, ...
        if token.startswith("("):
            token = token[-1]

        if token in result_dict:
            result_dict[token] += exp(logprob.logprob)

    return result_dict
