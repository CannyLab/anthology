import random
import re

from anthology.utils.survey_utils import generate_answer_forcing_prompt
import numpy as np
from copy import deepcopy
from typing import Type
from omegaconf import DictConfig

QUESTION_FORMAT = """{consistency_prompt}
Question: {question_body}
{choices}
{answer_forcing_prompt}
Answer:"""

QID_TO_OPTIONS_MAP = {
    # Wave 34 questions
    "FUD37A_W34": "map_dict_very_likely",
    "FUD37B_W34": "map_dict_very_likely",
    "FUD37C_W34": "map_dict_very_likely",
    "FUD22_W34": "map_dict_most_of_it",
    "FUD35_W34": "map_dict_a_great_deal",
    "EAT5B_W34": "map_dict_a_great_deal_of_health_risk",
    "EAT5C_W34": "map_dict_a_great_deal_of_health_risk",
    "EAT5D_W34": "map_dict_a_great_deal_of_health_risk",
    # Wave 92 questions
    "SOCIETY_TRANS_W92": "map_dict_very_good_for_society",
    "SOCIETY_RHIST_W92": "map_dict_very_good_for_society",
    "SOCIETY_JBCLL_W92": "map_dict_very_good_for_society",
    "SOCIETY_RELG_W92": "map_dict_very_good_for_society",
    "SOCIETY_WHT_W92": "map_dict_very_good_for_society",
    "SOCIETY_GUNS_W92": "map_dict_very_good_for_society",
    "SOCIETY_SSM_W92": "map_dict_very_good_for_society",
    "ELITEUNDMOD_W92": "map_dict_very_well",
    "PROBOFF_a_W92": "map_dict_major_problem",
    "RACESURV52MOD_W92": "map_dict_a_lot",
    "ELECT_IMPT3_PRVFR_W92": "map_dict_very_important",
    "CANDEXP_W92": "map_dict_very_important",
    "LEGALIMMIGAMT_W92": "map_dict_increase_a_lot",
    "UNIMMIGCOMM_W92": "map_dict_a_lot_better",
    "GODMORALIMP_W92": "map_dict_essential",
    "REPRSNTREP_W92": "map_dict_very_well",
    "REPRSNTDEM_W92": "map_dict_very_well",
    # Wave 99 questions
    "POSNEGAI_a_W99": "map_dict_very_excited",
    "POSNEGAI_b_W99": "map_dict_very_excited",
    "POSNEGAI_c_W99": "map_dict_very_excited",
    "POSNEGAI_d_W99": "map_dict_very_excited",
    "POSNEGAI_e_W99": "map_dict_very_excited",
    "POSNEGAI_f_W99": "map_dict_very_excited",
    # Wave 111 questions
    "ONCOMMIT_W111": "map_dict_a_lot_harder",
    "ONSAFE_W111": "map_dict_very_safe",
    "ONIMPACT_W111": "map_dict_mostly_positive_effect",
    "GAMB2_W111": "map_dict_a_good_thing_for_society",
    "GAMB3_W111": "map_dict_a_good_thing_for_sports",
    "HARASSWRK2_W111": "map_dict_extremely_common",
    "HARASSWRK3_W111": "map_dict_extremely_common",
    "HARASSWRK4_W111": "map_dict_more_likely_to_be_held_responsible",
    "HARASSWRK5_W111": "map_dict_more_likely_to_be_believed",
    "HARASSWRK1M_W111": "map_dict_easier_for_men_to_know_how_to_interact_with_women_in_the_workplace",
    "HARASSWRK1W_W111": "map_dict_easier_for_women_to_know_how_to_interact_with_men_in_the_workplace",
}

OPTIONS_TO_LABEL_MAP = {
    "map_dict_very_likely": {
        1: "Very likely",
        2: "Fairly likely",
        3: "Not too likely",
        4: "Not at all likely",
    },
    "map_dict_a_great_deal": {
        1: "A great deal",
        2: "Some",
        3: "Not too much",
        4: "Not at all",
    },
    "map_dict_most_of_it": {
        1: "Most of it",
        2: "Some of it",
        3: "Not too much",
        4: "None at all",
    },
    "map_dict_a_great_deal_of_health_risk": {
        1: "A great deal of health risk",
        2: "Some health risk",
        3: "Not too much health risk",
        4: "No health risk at all",
    },
    "map_dict_very_good_for_society": {
        1: "Very good for society",
        2: "Somewhat good for society",
        3: "Neither good nor bad for society",
        4: "Somewhat bad for society",
        5: "Very bad for society",
    },
    "map_dict_a_lot_harder": {
        1: "A lot harder",
        2: "A little harder",
        3: "Made no difference",
        4: "A little easier",
        5: "A lot easier",
    },
    "map_dict_very_safe": {
        1: "Very safe",
        2: "Somewhat safe",
        3: "Not too safe",
        4: "Not at all safe",
    },
    "map_dict_extremely_common": {
        1: "Extremely common",
        2: "Very common",
        3: "Somewhat common",
        4: "Not too common",
        5: "Not at all common",
    },
    "map_dict_a_good_thing_for_society": {
        1: "A good thing for society",
        2: "A bad thing for society",
        3: "Neither a good nor bad thing for society",
    },
    "map_dict_a_good_thing_for_sports": {
        1: "A good thing for sports",
        2: "A bad thing for sports",
        3: "Neither a good nor bad thing for sports",
    },
    "map_dict_very_excited": {
        1: "Very excited",
        2: "Somewhat excited",
        3: "Equal excitement and concern",
        4: "Somewhat concerned",
        5: "Very concerned",
    },
    "map_dict_more_likely_to_be_held_responsible": {
        1: "More likely to be held responsible",
        2: "Less likely to be held responsible",
        3: "Has not made much difference",
    },
    "map_dict_more_likely_to_be_believed": {
        1: "More likely to be believed",
        2: "Less likely to be believed",
        3: "Has not made much difference",
    },
    "map_dict_easier_for_men_to_know_how_to_interact_with_women_in_the_workplace": {
        1: "Easier for men to know how to interact with women in the workplace",
        2: "Harder for men to know how to interact with women in the workplace",
        3: "Has not made much difference",
    },
    "map_dict_easier_for_women_to_know_how_to_interact_with_men_in_the_workplace": {
        1: "Easier for women to know how to interact with men in the workplace",
        2: "Harder for women to know how to interact with men in the workplace",
        3: "Has not made much difference",
    },
    "map_dict_mostly_positive_effect": {
        1: "Mostly positive effect",
        2: "Mostly negative effect",
        3: "Neither positive nor negative effect",
    },
    "map_dict_very_well": {
        1: "Very well",
        2: "Somewhat well",
        3: "Not too well",
        4: "Not at all well",
    },
    "map_dict_major_problem": {
        1: "Major problem",
        2: "Minor problem",
        3: "Not a problem",
    },
    "map_dict_a_lot": {
        1: "A lot",
        2: "Some",
        3: "Not much",
        4: "Not at all",
    },
    "map_dict_very_important": {
        1: "Very important",
        2: "Somewhat important",
        3: "Not too important",
        4: "Not at all important",
    },
    "map_dict_increase_a_lot": {
        1: "Increase a lot",
        2: "Increase a little",
        3: "Stay about the same",
        4: "Decrease a little",
        5: "Decrease a lot",
    },
    "map_dict_a_lot_better": {
        1: "A lot better",
        2: "A little better",
        3: "A little worse",
        4: "A lot worse",
    },
    "map_dict_essential": {
        1: "Essential",
        2: "Important, but not essential",
        3: "Not too important",
        4: "Not at all important",
    },
}


def format_options(
    options: list[str],
) -> str:
    """
    Takes a list of option strings, returns a formatted string
    """
    formatted_options = ""
    for idx, option in enumerate(options):
        formatted_options += f"({chr(idx + 65)}) {option}\n"
    return formatted_options.strip()


def format_survey(
    questions: Type[DictConfig],
    consistency_prompt: str = "",
    randomize_option_order: bool = False,
    randomize_question_order: bool = False,
    is_empty_prompt: bool = False,
    include_answer_forcing: bool = False,
    in_series: bool = False,
) -> tuple[dict[str, str], bool]:
    """
    questions: configuration of questions
    randomize_option_order: global flag to determine whether to randomize option order
    * format_survey() takes questions and
    formatted_questions: a dictionary of questions
        key: qid (qid is taken from the human survey)
        value: formatted mcq string
        * randomize (option order, question order), consistency prompt, all considered
    """
    formatted_questions = {}

    if randomize_question_order:
        questions = dict(random.sample(questions.items(), len(questions)))

    for qid, question_info in questions.items():
        available_choices = deepcopy(question_info.choices)
        if (
            randomize_option_order and question_info.allow_randomization
        ):  # choose one from predetermined permutation
            randomization_pattern = random.choice(question_info.randomization_pattern)
            available_choices = [available_choices[i] for i in randomization_pattern]
        formatted_options = format_options(options=available_choices)
        answer_forcing_prompt = (
            generate_answer_forcing_prompt(num_options=len(available_choices))
            if include_answer_forcing
            else ""
        )
        formatted_question = QUESTION_FORMAT.format(
            consistency_prompt=(  # consistency prompt is not added when:
                # empty prompt (not given any backstories nor baseline prompts)
                # and it is the first question in series, or not asked in series
                ""
                if (
                    is_empty_prompt
                    and (qid == list(questions.keys())[0] or in_series == False)
                )
                else consistency_prompt
            ),
            question_body=question_info.question_body,
            choices=formatted_options,
            answer_forcing_prompt=answer_forcing_prompt,
        ).strip()
        formatted_questions[qid] = {
            "question": formatted_question,
            "randomization_pattern": randomization_pattern
            if (randomize_option_order and question_info.allow_randomization)
            else np.arange(len(available_choices)),
        }

    return formatted_questions


def label_to_value_mapper(
    label: chr,
    randomization_pattern: list[int],
) -> int:
    """
    mapping example:
    label = "A", randomization_pattern = [0,1,2,3]: return 1
    label = "E", randomization_pattern = [0,1,2,3,4]: return 5
    label = "A", randomization_pattern = [1,0,2]: return 2
    """
    try:
        return randomization_pattern.index(ord(label) - 65) + 1
    except:
        return -1000


def response_parser(
    response: str,
    randomization_pattern: list[int],
    qid: str,
) -> tuple[str, int]:
    options_to_label_map = OPTIONS_TO_LABEL_MAP[QID_TO_OPTIONS_MAP[qid]]

    label = regex_letter_classifier(response)
    text = regex_response_classifier(response, qid)  # lowercase text

    if label and not text:  # only the label appears in the response
        try:
            label_value = label_to_value_mapper(label, randomization_pattern)
            return options_to_label_map[label_value], label_value
        except:
            return "non_compliant", -1000

    elif text and not label:  # only the text appears in the response
        try:
            for key, value in options_to_label_map.items():
                if text == value.lower():
                    return value, key
            return "non_compliant", -1000
        except:
            return "non_compliant", -1000

    elif text and label:  # both the label and text appear in the response
        try:
            label_value = label_to_value_mapper(label, randomization_pattern)
            for key, value in options_to_label_map.items():
                if text == value.lower():
                    text_value = key
                    break
            if text_value == label_value:
                return options_to_label_map[label_value], label_value
            else:
                return "non_compliant", -1000
        except:
            return "non_compliant", -1000

    else:  # neither the label nor the text appear in the response
        return "non_compliant", -1000


def regex_letter_classifier(response):
    pattern = r"(?:^|[\[\(\"\' ])([A-Z])(?=$|[\]\)\"\'., ])(?! great)(?! lot)(?! little)(?! good)(?! bad)"
    match = list(set(re.findall(pattern, response)))
    return (
        match[0] if len(match) == 1 else None
    )  # if there is only one match, return the letter


def regex_response_classifier(response, qid):
    pattern_list = list(OPTIONS_TO_LABEL_MAP[QID_TO_OPTIONS_MAP[qid]].values())
    match_list = [[] for idx in range(len(pattern_list))]
    for idx, pattern in enumerate(pattern_list):
        match_list[idx] = list(set(re.findall(pattern, response, re.IGNORECASE)))

    matched_number = 0
    for match in match_list:
        if len(match) > 0:  # if there is a match
            matched_number += 1
            text = match[0].lower()
    return (
        text if matched_number == 1 else None
    )  # if there is only one match, return the text
