import random
import re

from anthology.utils.survey_utils import generate_answer_forcing_prompt
from copy import deepcopy
from typing import Type
from omegaconf import DictConfig, ListConfig

QUESTION_FORMAT = """{consistency_prompt}
Question: {question}
{choices}
{answer_forcing_prompt}
Answer:"""

DEMO_FIRST_LABEL_TEXT_PAIRS = {
    "A": "democrat",
    "a": "democrat",
    "B": "republican",
    "b": "republican",
    "C": "independent",
    "c": "independent",
    "D": "other",
    "d": "other",
    "E": "prefer_not_to_answer",
    "e": "prefer_not_to_answer",
    "F": "no_answer",
    "f": "no_answer",
}

REP_FIRST_LABEL_TEXT_PAIRS = {
    "B": "democrat",
    "b": "democrat",
    "A": "republican",
    "a": "republican",
    "C": "independent",
    "c": "independent",
    "D": "other",
    "d": "other",
    "E": "prefer_not_to_answer",
    "e": "prefer_not_to_answer",
    "F": "no_answer",
    "f": "no_answer",
}

LIBERAL_FIRST_LABEL_TEXT_PAIRS = {

}

CONSERVATIVE_FIRST_LABEL_TEXT_PAIRS = {
    
}

STRENGTH_LABEL_TEXT = {
    "A": "strong",
    "a": "strong",
    "B": "not_very_strong",
    "b": "not_very_strong",
}

IND_LEANING_LABEL_TEXT = {
    "A": "closer_to_republican",
    "a": "closer_to_republican",
    "B": "neither",
    "b": "neither",
    "C": "closer_to_democrat",
    "c": "closer_to_democrat",
}


def regex_strength_letter_classifier(response):
    pattern = r"^\s*[\(\[]?([A-B])[\)\]]?\.?\s*(?:$|\s+|\))"

    # Search for the pattern in the input text
    match = re.match(pattern, response, re.IGNORECASE)

    # If a match is found, return the letter
    if match:
        return match.group(1)
    else:
        return None


def regex_strength_response_classifier(response):
    pattern = r".*?(strong|not very strong|neither one or equally close to both).*?"

    # Search for the pattern in the input text
    match = re.match(pattern, response, re.IGNORECASE)

    # If a match is found, return the letter and the response
    if match:
        text = match.group(1)
        text = text.lower().replace(" ", "_")

    else:
        text = None

    return text


def regex_leaning_letter_classifier(response):
    pattern = r"^\s*[\(\[]?([A-C])[\)\]]?\.?\s*(?:$|\s+|\))"

    # Search for the pattern in the input text
    match = re.match(pattern, response, re.IGNORECASE)

    # If a match is found, return the letter
    if match:
        return match.group(1)
    else:
        return None


def regex_leaning_response_classifier(response):
    pattern = r".*?(closer to republican|neither|closer to democrat).*?"

    # Search for the pattern in the input text
    match = re.match(pattern, response, re.IGNORECASE)

    # If a match is found, return the letter and the response
    if match:
        text = match.group(1)
        text = text.lower().replace(" ", "_")

    else:
        text = None

    return text

def regex_leaning_letter_classifier_v2(response):
    pattern = r"(?:^|[\[\( ])([A-C])(?=$|[\]\). ])"

    # Search for the pattern in the input text
    match = re.findall(pattern, response,)

    match = list(set(match))

    # If a match is found, return the letter
    if match and len(match) == 1:
        return match[0]
    else:
        return None

def regex_letter_classifier_v2(response):
    pattern = r"(?:^|[\[\( ])([A-E])(?=$|[\]\). ])"
    # Search for the pattern in the input text
    match = re.findall(pattern, response,)

    match = list(set(match))

    # If a match is found, return the letter
    if match and len(match) == 1:
        return match[0]
    else:
        return None

def regex_letter_classifier(response):
    pattern = r"^\s*[\(\[]?([A-E])[\)\]]?\.?\s*(?:$|\s+|\))"

    # Search for the pattern in the input text
    match = re.match(pattern, response, re.IGNORECASE)

    # If a match is found, return the letter
    if match:
        return match.group(1)
    else:
        return None


def regex_response_classifier(response):
    pattern = r".*?(democrat|democrats|republican|republicans|other|prefer not to answer|independent).*?"

    # Search for the pattern in the input text
    match = re.match(pattern, response, re.IGNORECASE)

    # If a match is found, return the letter and the response
    if match:
        text = match.group(1)
        text = text.lower().replace(" ", "_")

    else:
        text = None

    return text

def regex_response_classifier_v2(response):
    pattern = r"democrat|republican|independent|other|prefer not to"

    # Search for the pattern in the input text
    match = re.findall(pattern, response, re.IGNORECASE)
    match = list(set(match))

    # If a match is found, return the letter and the response
    if len(match) == 1:
        text = match[0]
        text = text.lower().replace(" ", "_")

    else:
        text = None

    return text

def strength_parser(response):
    strength_letter = regex_strength_letter_classifier(response)
    strength_response = regex_strength_response_classifier(response)

    if strength_letter and strength_response:
        if STRENGTH_LABEL_TEXT[strength_letter] == strength_response:
            return STRENGTH_LABEL_TEXT[strength_letter]
        else:
            return "non_compliant"
    elif strength_letter:
        return STRENGTH_LABEL_TEXT[strength_letter]
    elif strength_response:
        return strength_response
    else:
        return "non_compliant"


def ind_leaning_parser(response):
    leaning_letter = regex_leaning_letter_classifier_v2(response)
    leaning_response = regex_leaning_response_classifier(response)

    if leaning_letter and leaning_response:
        if IND_LEANING_LABEL_TEXT[leaning_letter] == leaning_response:
            return IND_LEANING_LABEL_TEXT[leaning_letter]
        else:
            return "non_compliant"
    elif leaning_letter:
        return IND_LEANING_LABEL_TEXT[leaning_letter]
    elif leaning_response:
        return leaning_response
    else:
        return "non_compliant"


def affiliation_parser(response, democrat_first):
    parsed_letter = regex_letter_classifier_v2(response)
    parsed_response = regex_response_classifier_v2(response)

    if democrat_first:
        label_text_pairs = DEMO_FIRST_LABEL_TEXT_PAIRS
    else:
        label_text_pairs = REP_FIRST_LABEL_TEXT_PAIRS

    if parsed_letter and parsed_response:
        if label_text_pairs[parsed_letter] == parsed_response:
            return label_text_pairs[parsed_letter]
        else:
            return "non_compliant"
    elif parsed_letter:
        return label_text_pairs[parsed_letter]
    elif parsed_response:
        return parsed_response
    else:
        return "non_compliant"


def format_options(
    options: list[str],
) -> str:
    """
    Format the options

    Args:
        options: list of options

    Returns:
        formatted options
    """

    formatted_options = ""

    for idx, option in enumerate(options):
        formatted_options += f"({chr(idx + 65)}) {option}\n"

    return formatted_options.strip()


def format_affiliation_surveys(
    question: str,
    available_options: Type[ListConfig],
    consistency_prompt: str,
    randomize_choice: bool = True,
) -> str:
    democrat_first = True

    available_options = deepcopy(available_options)

    if randomize_choice and random.choice([True, False]):
        # Swap the first and second options
        available_options[0], available_options[1] = (
            available_options[1],
            available_options[0],
        )
        democrat_first = False

    serialized_choices = f"a {available_options[0]}, a {available_options[1]},"
    question = question.format(RANDOM_DEMO_REPUBLICAN=serialized_choices)

    formatted_options = format_options(available_options)
    answer_forcing_prompt = generate_answer_forcing_prompt(len(available_options))

    formatted_question = QUESTION_FORMAT.format(
        consistency_prompt=consistency_prompt,
        question=question,
        choices=formatted_options,
        answer_forcing_prompt=answer_forcing_prompt,
    )

    return formatted_question, democrat_first


def format_ideology_surveys(
    question: str,
    available_options: Type[ListConfig],
    consistency_prompt: str,
    randomize_choice: bool = True,
) -> str:

    available_options = deepcopy(available_options)
    liberal_first = True
    if randomize_choice and random.choice([True, False]):
        # Swap the first and second options
        available_options[0], available_options[4] = (
            available_options[4],
            available_options[0],
        )
        available_options[1], available_options[3] = (
            available_options[3],
            available_options[1],
        )
        liberal_first = False

    serialized_choices = f"{available_options[1].lower()} to {available_options[3].lower()}"
    question = question.format(RANDOM_SPECTRUM=serialized_choices)

    formatted_options = format_options(available_options)
    answer_forcing_prompt = generate_answer_forcing_prompt(len(available_options))

    formatted_question = QUESTION_FORMAT.format(
        consistency_prompt=consistency_prompt,
        question=question,
        choices=formatted_options,
        answer_forcing_prompt=answer_forcing_prompt,
    )

    return formatted_question, liberal_first


def format_strength_surveys(
    question: str,
    available_options: Type[ListConfig],
    consistency_prompt: str,
) -> str:
    formatted_options = format_options(available_options)
    answer_forcing_prompt = generate_answer_forcing_prompt(len(available_options))

    formatted_question = QUESTION_FORMAT.format(
        consistency_prompt=consistency_prompt,
        question=question,
        choices=formatted_options,
        answer_forcing_prompt=answer_forcing_prompt,
    )

    return formatted_question
