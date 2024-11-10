import pathlib
import sys
from typing import Type

from omegaconf import DictConfig, ListConfig

sys.path.append(str(pathlib.Path(__file__).parent.parent))


def generate_mcq(
    question: str,
    choices: Type[ListConfig],
    choice_format: Type[DictConfig],
    mcq_symbol: str,
    surveyor_respondent: Type[DictConfig],
) -> str:
    """
    Generate multiple choice question

    Args:
        question: question text
        choices: list of choices
        choice_format: question format
        mcq_symbols: multiple choice symbols
        surveyor_respondent: surveyor and respondent format

    Returns:
        formatted question and choices
    """
    surveyor = surveyor_respondent.surveyor
    # Role of asking the question
    # e.g. Question, Surveyor, Interviewer, etc.
    respondent = surveyor_respondent.respondent
    # Role of answering the question
    # e.g. Answer, Respondent, Interviewee, etc.
    connector = surveyor_respondent.connector
    # Connector between each role and the question or answer
    # e.g. (Q:, A:), (Q., A.), (Q, A), etc.

    left_symbol = choice_format.left_symbol
    # Left bracket of the choice symbol
    # e.g. (, [, etc.
    right_symbol = choice_format.right_symbol
    # Right bracket of the choice symbol
    # e.g. ), ], empty string, etc.

    mcq_choice_str = ""

    # generate choices
    for idx, choice in enumerate(choices):
        if mcq_symbol == "number":
            choice_symbol = str(idx + 1)
        elif mcq_symbol == "uppercase":
            choice_symbol = chr(idx + 65)
        elif mcq_symbol == "lowercase":
            choice_symbol = chr(idx + 97)
        else:
            raise ValueError("mcq_symbol must be one of number, uppercase, lowercase")

        mcq_choice_str += f"{left_symbol}{choice_symbol}{right_symbol} {choice}\n"

    # generate question
    formatted_question = f"{surveyor}{connector} {question}\n{mcq_choice_str}\n"

    # Check if connector is not empty string
    # We need to be super careful with the connector
    # as LLM is sensitive to trailing spaces.
    if connector:
        formatted_question += f"{respondent}{connector}"
    else:
        formatted_question += f"{respondent}"

    return formatted_question
