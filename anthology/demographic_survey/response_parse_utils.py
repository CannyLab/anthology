import re

from .demographics_regex_patterns import *


def nonzero_index_finder(
    arr: list,
) -> list:
    index_arr = []
    for i in range(len(arr)):
        if arr[i] > 0:
            index_arr.append(i)
    return index_arr


def label_to_index_mapper(
    letter: str,
    num_choices: int,
) -> int:
    ordinal = ord(letter) - ord("A")
    return ordinal if ordinal < num_choices else -1


def number_to_index_mapper(
    number: int,
    question_cfg: dict,
    number_range_list: list,
) -> int:
    assert question_cfg["is_numerical"]
    index = 0
    while index < len(number_range_list):
        if number < number_range_list[index]:
            break
        index += 1
    if "age" in question_cfg.question:
        return index - 1
    else:
        return index


def list_to_bucket(
    arr: list,
    num_choices: int,
) -> list:
    bucket = [0 for x in range(num_choices + 1)]
    for x in arr:
        bucket[x] += 1
    return bucket


# replace the number with the number * 1 million
def multiply_by_million_and_remove_commas(match):
    number_str = match.group(1).replace(",", "")
    number = float(number_str)
    new_number = number * 1000000
    return f"{new_number:,.0f}"


# replace the number with the number * 1 thousand
def multiply_by_thousand_and_remove_commas(match):
    number_str = match.group(1).replace(",", "")
    number = float(number_str)
    new_number = number * 1000
    return f"{new_number:,.0f}"
