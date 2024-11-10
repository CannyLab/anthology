from copy import deepcopy

import pandas as pd


def conditioned_demographic_survey_info(
    original_file: pd.DataFrame,
    demographic_group_to_include: dict,
    demographic_group_to_exclude: dict,
) -> pd.DataFrame:
    """
    input:
        - raw human survey information
        - demographic group to include (exclude)
            - key: demographic variable
            - value: list of groups to include
    """

    ret_file = deepcopy(original_file)

    for variable, include_group in demographic_group_to_include.items():
        if include_group == None:
            continue
        ret_file = ret_file[ret_file[variable].isin(include_group)]
    for variable, exclude_group in demographic_group_to_exclude.items():
        if exclude_group == None:
            continue
        ret_file = ret_file[~ret_file[variable].isin(exclude_group)]

    return ret_file
