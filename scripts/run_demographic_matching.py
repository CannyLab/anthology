import sys
from copy import deepcopy

import pathlib
import hydra
import random
import pandas as pd
import pyreadstat
import numpy as np
from scipy.optimize import linear_sum_assignment
from omegaconf import OmegaConf, DictConfig

sys.path.append(str(pathlib.Path(__file__).parent.parent))

config_path = pathlib.Path(__file__).parent.parent / "configs" / "demographic_matching"

def edge_weight_calculation(
    human_data: pd.DataFrame,
    demographics_survey: pd.DataFrame,
    trait_of_interest: list[str],
) -> np.ndarray:
    """
    Calculate the edge weight matrix (N by M) between a virtual user and a human user.
    N = number of human users
    M = number of virtual users
    Edge weight is calcualted by the similarity of a virtual user's response distribution and a human user trait.
    """
    edge_weight = np.ones((len(human_data), len(demographics_survey)))
    for human_trait in trait_of_interest:
        human_matrix = np.array(human_data[f"{human_trait}_list"].tolist())
        virtual_matrix = np.array(demographics_survey[f"{human_trait}_list"].tolist()).T
        edge_weight = edge_weight * (human_matrix @ virtual_matrix)
    return edge_weight


def maximum_weight_sum_matching(
    human_data: pd.DataFrame,
    demographics_survey: pd.DataFrame,
    trait_of_interest: list[str],
) -> dict:
    """
    Perform a maximum weight sum matching.
    return value contains:
        - a tuple containing the matched human user (QKEY) and a virtual user (uid)
        - a list containing the edge weight of the matched pairs
    """
    edge_weight = edge_weight_calculation(
        human_data,
        demographics_survey,
        trait_of_interest,
    )
    row_ind, col_ind = linear_sum_assignment(-edge_weight)
    tuple_list = []
    for i in range(len(row_ind)):
        tuple_list.append(
            tuple([human_data.at[row_ind[i], "QKEY"],demographics_survey.at[col_ind[i], "uid"]])
        )
    return {
        "qkey_uid_tuple": tuple_list,
        "selected_weight_list": edge_weight[row_ind, col_ind],
    }


def greedy_matching(
    human_data: pd.DataFrame,
    demographics_survey: pd.DataFrame,
    trait_of_interest: list[str],
) -> dict:
    """
    Perform a greedy matching.
    return value contains:
        - a tuple containing the matched human user (QKEY) and a virtual user (uid)
        - a list containing the edge weight of the matched pairs
    """
    edge_weight = edge_weight_calculation(
        human_data,
        demographics_survey,
        trait_of_interest,
    )
    tuple_list = []
    edge_weight_list = []
    for human_idx in range(len(edge_weight)):
        tuple_list.append(
            tuple([human_data.at[human_idx, "QKEY"],demographics_survey.at[np.argmax(edge_weight[human_idx]), "uid"]])
        )
        edge_weight_list.append(np.max(edge_weight[human_idx]))
    return {
        "qkey_uid_tuple": tuple_list,
        "selected_weight_list": edge_weight_list,
    }


TRAIT_DICT = { # contains the column name of the human survey datafile and LLM demographic survey keys
    "ATP_W34": [
        "F_INCOME_FINAL",
        "F_RACETHN_RECRUITMENT",
        "F_AGECAT_FINAL",
        "F_SEX_FINAL",
        "F_EDUCCAT2_FINAL",
        'F_CREGION_FINAL',
        'F_RELIG_FINAL',
        'F_PARTY_FINAL',
    ],
    "ATP_W92": [
        "F_INC_SDT1",
        "F_RACETHNMOD",
        "F_AGECAT",
        "F_GENDER",
        "F_EDUCCAT2",
        'F_CREGION',
        'F_RELIG',
        'F_PARTY_FINAL',
    ],
    "ATP_W111": [
        "F_INC_SDT1",
        "F_RACETHNMOD",
        "F_AGECAT",
        "F_GENDER",
        "F_EDUCCAT2",
        'F_CREGION',
        'F_RELIG',
        'F_PARTY_FINAL',
    ],
    "ATP_W99": [
        "F_INC_SDT1",
        "F_RACETHNMOD",
        "F_AGECAT",
        "F_GENDER",
        "F_EDUCCAT2",
        'F_CREGION',
        'F_RELIG',
        'F_PARTY_FINAL',
    ],
    "llm": [
        "income_level_category_13",
        "race",
        "age_category_4",
        "gender",
        "education_level",
        'region',
        'religion',
        'political_affiliation',
    ],
}

# intermediate representation is required since LLM demographic survey is performed usually with a finer grained categories.
# For example, we have finer categories for race and ethnicity question (taken from: https://idealdeisurvey.stanford.edu/frequently-asked-questions/survey-definitions)
# compared to the American Trends Panel Wave 34 survey.

INTERMEDIATE_REP_LLM = {
    "ATP_W92": {
        "income_level_category_13": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8],
        "race": [4, 3, 1, 2, 4, 4, 0, 4],
        "age_category_4": [0, 1, 2, 3],
        "gender": [0, 1, 2],
        "education_level": [0, 1, 2, 3, 4, 5, 5, 5],
        "region": [0, 1, 2, 3],
        "religion": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "political_affiliation": [0, 1, 2, 3],
    },
    "ATP_W111": {
        "income_level_category_13": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8],
        "race": [4, 3, 1, 2, 4, 4, 0, 4],
        "age_category_4": [0, 1, 2, 3],
        "gender": [0, 1, 2],
        "education_level": [0, 1, 2, 3, 4, 5, 5, 5],
        "region": [0, 1, 2, 3],
        "religion": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "political_affiliation": [0, 1, 2, 3],
    },
    "ATP_W99": {
        "income_level_category_13": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8],
        "race": [4, 3, 1, 2, 4, 4, 0, 4],
        "age_category_4": [0, 1, 2, 3],
        "gender": [0, 1, 2],
        "education_level": [0, 1, 2, 3, 4, 5, 5, 5],
        "region": [0, 1, 2, 3],
        "religion": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "political_affiliation": [0, 1, 2, 3],
    },
    "ATP_W34": {
        "income_level_category_13": [0, 1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 8, 8],
        "race": [3, 3, 1, 2, 3, 3, 0, 3],
        "age_category_4": [0, 1, 2, 3],
        "gender": [0, 1, 2],
        "education_level": [0, 1, 2, 3, 4, 5, 5, 5],
        "region": [0, 1, 2, 3],
        "religion": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "political_affiliation": [0, 1, 2, 3],
    },
}

INTERMEDIATE_REP_SURVEY = {
    "ATP_W92": {
        "F_INC_SDT1": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "F_RACETHNMOD": [None, 0, 1, 2, 3, 4],
        "F_AGECAT": [None, 0, 1, 2, 3],
        "F_GENDER": [None, 0, 1, 2],
        "F_EDUCCAT2": [None, 0, 1, 2, 3, 4, 5],
        "F_CREGION": [None, 0, 1, 2, 3],
        "F_RELIG": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "F_PARTY_FINAL": [None, 1, 0, 2, 3],
    },
    "ATP_W111": {
        "F_INC_SDT1": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "F_RACETHNMOD": [None, 0, 1, 2, 3, 4],
        "F_AGECAT": [None, 0, 1, 2, 3],
        "F_GENDER": [None, 0, 1, 2],
        "F_EDUCCAT2": [None, 0, 1, 2, 3, 4, 5],
        "F_CREGION": [None, 0, 1, 2, 3],
        "F_RELIG": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "F_PARTY_FINAL": [None, 1, 0, 2, 3],
    },
    "ATP_W99": {
        "F_INC_SDT1": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "F_RACETHNMOD": [None, 0, 1, 2, 3, 4],
        "F_AGECAT": [None, 0, 1, 2, 3],
        "F_GENDER": [None, 0, 1, 2],
        "F_EDUCCAT2": [None, 0, 1, 2, 3, 4, 5],
        "F_CREGION": [None, 0, 1, 2, 3],
        "F_RELIG": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "F_PARTY_FINAL": [None, 1, 0, 2, 3],
    },
    "ATP_W34": {
        "F_INCOME_FINAL": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "F_RACETHN_RECRUITMENT": [None, 0, 1, 2, 3],
        "F_AGECAT_FINAL": [None, 0, 1, 2, 3],
        "F_SEX_FINAL": [None, 0, 1, 2],
        "F_EDUCCAT2_FINAL": [None, 0, 1, 2, 3, 4, 5],
        "F_CREGION_FINAL": [None, 0, 1, 2, 3],
        "F_RELIG_FINAL": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "F_PARTY_FINAL": [None, 1, 0, 2, 3],
    },
}

QA_MCQ_PROMPT = {
    "ATP_W92": {
        "F_INC_SDT1": """Question: What is your annual household income?
(A) Less than $30,000
(B) $30,000 to less than $40,000
(C) $40,000 to less than $50,000
(D) $50,000 to less than $60,000
(E) $60,000 to less than $70,000
(F) $70,000 to less than $80,000
(G) $80,000 to less than $90,000
(H) $90,000 to less than $100,000
(I) $100,000 or more
Answer with (A), (B), (C), (D), (E), (F), (G), (H), or (I).
Answer:""",
        "F_RACETHNMOD": """Question: Which of the following racial or ethnic groups do you identify with?
(A) White non-Hispanic
(B) Black non-Hispanic
(C) Hispanic
(D) Other
(E) Asian non-Hispanic
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_AGECAT": """Question: What is your age?
(A) 18-29
(B) 30-49
(C) 50-64
(D) 65+
Answer with (A), (B), (C), or (D).
Answer:""",
        "F_GENDER": """Question: What is your gender?
(A) A man
(B) A woman
(C) In some other way
Answer with (A), (B), or (C).
Answer:""",
        "F_EDUCCAT2": """Question: What is the highest level of education you have completed?
(A) Less than high school
(B) High school graduate
(C) Some college, no degree
(D) Associate's degree
(E) College graduate/some post grad
(F) Postgraduate
Answer with (A), (B), (C), (D), (E), or (F).
Answer:""",
        "F_CREGION": """Question: In which region of the United States do you live?
(A) Northeast
(B) Midwest
(C) South
(D) West
(E) Prefer not to answer
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_RELIG": """Question: What is your religion?
(A) Protestant
(B) Roman Catholic
(C) Mormon (Church of Jesus Christ of Latter-day Saints or LDS)
(D) Orthodox (such as Greek, Russian, or some other Orthodox church)
(E) Jewish
(F) Muslim
(G) Buddhist
(H) Hindu
(I) Atheist
(J) Agnostic
(K) Other
(L) Nothing in particular
(M) Prefer not to answer
Answer with (A), (B), (C), (D), (E), (F), (G), (H), (I), (J), (K), (L), or (M).
Answer:""",
        "F_PARTY_FINAL": """Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?
(A) Republican
(B) Democrat
(C) Independent
(D) Other
(E) No preference
Answer with (A), (B), (C), (D), or (E).
Answer:""",
    },
    "ATP_W111": {
        "F_INC_SDT1": """Question: What is your annual household income?
(A) Less than $30,000
(B) $30,000 to less than $40,000
(C) $40,000 to less than $50,000
(D) $50,000 to less than $60,000
(E) $60,000 to less than $70,000
(F) $70,000 to less than $80,000
(G) $80,000 to less than $90,000
(H) $90,000 to less than $100,000
(I) $100,000 or more
Answer with (A), (B), (C), (D), (E), (F), (G), (H), or (I).
Answer:""",
        "F_RACETHNMOD": """Question: Which of the following racial or ethnic groups do you identify with?
(A) White non-Hispanic
(B) Black non-Hispanic
(C) Hispanic
(D) Other
(E) Asian non-Hispanic
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_AGECAT": """Question: What is your age?
(A) 18-29
(B) 30-49
(C) 50-64
(D) 65+
Answer with (A), (B), (C), or (D).
Answer:""",
        "F_GENDER": """Question: What is your gender?
(A) A man
(B) A woman
(C) In some other way
Answer with (A), (B), or (C).
Answer:""",
        "F_EDUCCAT2": """Question: What is the highest level of education you have completed?
(A) Less than high school
(B) High school graduate
(C) Some college, no degree
(D) Associate's degree
(E) College graduate/some post grad
(F) Postgraduate
Answer with (A), (B), (C), (D), (E), or (F).
Answer:""",
        "F_CREGION": """Question: In which region of the United States do you live?
(A) Northeast
(B) Midwest
(C) South
(D) West
(E) Prefer not to answer
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_RELIG": """Question: What is your religion?
(A) Protestant
(B) Roman Catholic
(C) Mormon (Church of Jesus Christ of Latter-day Saints or LDS)
(D) Orthodox (such as Greek, Russian, or some other Orthodox church)
(E) Jewish
(F) Muslim
(G) Buddhist
(H) Hindu
(I) Atheist
(J) Agnostic
(K) Other
(L) Nothing in particular
(M) Prefer not to answer
Answer with (A), (B), (C), (D), (E), (F), (G), (H), (I), (J), (K), (L), or (M).
Answer:""",
        "F_PARTY_FINAL": """Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?
(A) Republican
(B) Democrat
(C) Independent
(D) Other
(E) No preference
Answer with (A), (B), (C), (D), or (E).
Answer:""",
    },
    "ATP_W99": {
        "F_INC_SDT1": """Question: What is your annual household income?
(A) Less than $30,000
(B) $30,000 to less than $40,000
(C) $40,000 to less than $50,000
(D) $50,000 to less than $60,000
(E) $60,000 to less than $70,000
(F) $70,000 to less than $80,000
(G) $80,000 to less than $90,000
(H) $90,000 to less than $100,000
(I) $100,000 or more
Answer with (A), (B), (C), (D), (E), (F), (G), (H), or (I).
Answer:""",
        "F_RACETHNMOD": """Question: Which of the following racial or ethnic groups do you identify with?
(A) White non-Hispanic
(B) Black non-Hispanic
(C) Hispanic
(D) Other
(E) Asian non-Hispanic
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_AGECAT": """Question: What is your age?
(A) 18-29
(B) 30-49
(C) 50-64
(D) 65+
Answer with (A), (B), (C), or (D).
Answer:""",
        "F_GENDER": """Question: What is your gender?
(A) A man
(B) A woman
(C) In some other way
Answer with (A), (B), or (C).
Answer:""",
        "F_EDUCCAT2": """Question: What is the highest level of education you have completed?
(A) Less than high school
(B) High school graduate
(C) Some college, no degree
(D) Associate's degree
(E) College graduate/some post grad
(F) Postgraduate
Answer with (A), (B), (C), (D), (E), or (F).
Answer:""",
        "F_CREGION": """Question: In which region of the United States do you live?
(A) Northeast
(B) Midwest
(C) South
(D) West
(E) Prefer not to answer
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_RELIG": """Question: What is your religion?
(A) Protestant
(B) Roman Catholic
(C) Mormon (Church of Jesus Christ of Latter-day Saints or LDS)
(D) Orthodox (such as Greek, Russian, or some other Orthodox church)
(E) Jewish
(F) Muslim
(G) Buddhist
(H) Hindu
(I) Atheist
(J) Agnostic
(K) Other
(L) Nothing in particular
(M) Prefer not to answer
Answer with (A), (B), (C), (D), (E), (F), (G), (H), (I), (J), (K), (L), or (M).
Answer:""",
        "F_PARTY_FINAL": """Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?
(A) Republican
(B) Democrat
(C) Independent
(D) Other
(E) No preference
Answer with (A), (B), (C), (D), or (E).
Answer:""",
    },
    "ATP_W34": {
        "F_INCOME_FINAL": """Question: What is your annual household income?
(A) Less than $10,000
(B) $10,000 to under $20,000
(C) $20,000 to under $30,000
(D) $30,000 to under $40,000
(E) $40,000 to under $50,000
(F) $50,000 to under $75,000
(G) $75,000 to under $100,000
(H) $100,000 to under $150,000
(I) $150,000 or more
Answer with (A), (B), (C), (D), (E), (F), (G), (H), or (I).
Answer:""",
        "F_RACETHN_RECRUITMENT": """Question: Which of the following racial or ethnic groups do you identify with?
(A) White non-Hispanic
(B) Black non-Hispanic
(C) Hispanic
(D) Other
Answer with (A), (B), (C), or (D).
Answer:""",
        "F_AGECAT_FINAL": """Question: What is your age?
(A) 18-29
(B) 30-49
(C) 50-64
(D) 65+
Answer with (A), (B), (C), or (D).
Answer:""",
        "F_SEX_FINAL": """Question: What is your gender?
(A) Male
(B) Female
Answer with (A), or (B).
Answer:""",
        "F_EDUCCAT2_FINAL": """Question: What is the highest level of education you have completed?
(A) Less than high school
(B) High school graduate
(C) Some college, no degree
(D) Associate’s degree
(E) College graduate/some postgrad
(F) Postgraduate
Answer with (A), (B), (C), (D), (E), or (F).
Answer:""",
        "F_CREGION_FINAL": """Question: In which region of the United States do you live?
(A) Northeast
(B) Midwest
(C) South
(D) West
(E) Prefer not to answer
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_RELIG_FINAL": """Question: What is your religion?
(A) Protestant
(B) Roman Catholic
(C) Mormon (Church of Jesus Christ of Latter-day Saints or LDS)
(D) Orthodox (such as Greek, Russian, or some other Orthodox church)
(E) Jewish
(F) Muslim
(G) Buddhist
(H) Hindu
(I) Atheist
(J) Agnostic
(K) Other
(L) Nothing in particular
(M) Prefer not to answer
Answer with (A), (B), (C), (D), (E), (F), (G), (H), (I), (J), (K), (L), or (M).
Answer:""",
        "F_PARTY_FINAL": """Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?
(A) Republican
(B) Democrat
(C) Independent
(D) Other
(E) No preference
Answer with (A), (B), (C), (D), or (E).
Answer:""",
        "F_CITIZEN_RECODE_FINAL": """Question: Are you a US Citizen?
(A) US Citizen
(B) Not US Citizen
Answer with (A), or (B).
Answer:""",
    },
}

QA_FREEFORM_PROMPT = {
    "ATP_W92": {
        "F_INC_SDT1": "Question: What is your annual household income?",
        "F_RACETHNMOD": "Question: Which of the following racial or ethnic groups do you identify with?",
        "F_AGECAT": "Question: What is your age?",
        "F_GENDER": "Question: What is your gender?",
        "F_EDUCCAT2": "Question: What is the highest level of education you have completed?",
        "F_CREGION": "Question: In which region of the United States do you live?",
        "F_RELIG": "Question: What is your religion?",
        "F_PARTY_FINAL": "Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?",
    },
    "ATP_W111": {
        "F_INC_SDT1": "Question: What is your annual household income?",
        "F_RACETHNMOD": "Question: Which of the following racial or ethnic groups do you identify with?",
        "F_AGECAT": "Question: What is your age?",
        "F_GENDER": "Question: What is your gender?",
        "F_EDUCCAT2": "Question: What is the highest level of education you have completed?",
        "F_CREGION": "Question: In which region of the United States do you live?",
        "F_RELIG": "Question: What is your religion?",
        "F_PARTY_FINAL": "Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?",
    },
    "ATP_W99": {
        "F_INC_SDT1": "Question: What is your annual household income?",
        "F_RACETHNMOD": "Question: Which of the following racial or ethnic groups do you identify with?",
        "F_AGECAT": "Question: What is your age?",
        "F_GENDER": "Question: What is your gender?",
        "F_EDUCCAT2": "Question: What is the highest level of education you have completed?",
        "F_CREGION": "Question: In which region of the United States do you live?",
        "F_RELIG": "Question: What is your religion?",
        "F_PARTY_FINAL": "Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?",
    },
    "ATP_W34": {
        "F_INCOME_FINAL": "Question: What is your annual household income?",
        "F_RACETHN_RECRUITMENT": "Question: Which of the following racial or ethnic groups do you identify with?",
        "F_AGECAT_FINAL": "Question: What is your age?",
        "F_SEX_FINAL": "Question: What is your gender?",
        "F_EDUCCAT2_FINAL": "Question: What is the highest level of education you have completed?",
        "F_CREGION_FINAL": "Question: In which region of the United States do you live?",
        "F_RELIG_FINAL": "Question: What is your religion?",
        "F_PARTY_FINAL": "Question: Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent or what?",
        "F_CITIZEN_RECODE_FINAL": "Question: Are you a US Citizen?",
    },
}

@hydra.main(config_path=str(config_path), config_name="config")
def my_app(cfg):

    # global TRAIT_DICT, INTERMEDIATE_REP_LLM, INTERMEDIATE_REP_SURVEY

    demographics_survey = pd.read_pickle(cfg.demographics_survey_path)
    demographics_survey_original = deepcopy(demographics_survey)
    human_data, meta = pyreadstat.read_sav(cfg.human_survey_path)
    human_data_original = deepcopy(human_data)
    demographic_questions = pd.read_json(cfg.demographics_info_path)
    human_survey = f"ATP_W{cfg.wave}"
    print(f"human respondents: {len(human_data)}")
    print(f"virtual users: {len(demographics_survey)}")

    trait_of_interest = list(cfg.trait_of_interest)
    llm_column_to_atp_column = {
        k: v for k, v in dict(
            zip(TRAIT_DICT["llm"], TRAIT_DICT[human_survey])
        ).items() if k in trait_of_interest
    }
    trait_of_interest = {
        "llm": list(llm_column_to_atp_column.keys()),
        human_survey: list(llm_column_to_atp_column.values()),
    }
    intermediate_rep_llm = INTERMEDIATE_REP_LLM[human_survey]
    intermediate_rep_survey = INTERMEDIATE_REP_SURVEY[human_survey]
    value_to_string_map = demographic_questions

    # change human survey data entries to the intermediate representation
    for human_trait in trait_of_interest[human_survey]:
        n_cat = max(intermediate_rep_survey[human_trait][1:]) + 1
        human_data[f"{human_trait}_list"] = ""
        for idx in human_data.index:
            raw_value = human_data.at[idx, human_trait]
            if (
                np.isnan(raw_value)
                or "Refused" in meta.variable_value_labels[human_trait][raw_value]
            ):
                human_data.at[idx, human_trait] = np.nan
                human_data.at[idx, f"{human_trait}_list"] = np.array(
                    [1 / n_cat for _ in range(n_cat)]
                )
            else:
                intermediate_value = intermediate_rep_survey[human_trait][int(raw_value)]
                human_data.at[idx, human_trait] = intermediate_value
                human_data.at[idx, f"{human_trait}_list"] = np.zeros(n_cat)
                human_data.at[idx, f"{human_trait}_list"][intermediate_value] = 1
    # change llm demographic survey data entries to the intermediate representation
    for trait in trait_of_interest["llm"]:
        human_trait = llm_column_to_atp_column[trait]
        demographics_survey[f"{human_trait}_list"] = ""
        for idx in demographics_survey.index:
            resp_dist = demographics_survey.at[idx, f"{trait}_response_distribution"][:-2]
            try:
                resp_dist = [x / sum(resp_dist) for x in resp_dist]
            except:  # division by zero. All responses noncompliant or prefer-not
                resp_dist = [1 / len(resp_dist) for x in resp_dist]

            demographics_survey.at[idx, f"{human_trait}_list"] = np.zeros(
                int(max(intermediate_rep_llm[trait])) + 1
            )
            for cat, value in enumerate(resp_dist):
                demographics_survey.at[idx, f"{human_trait}_list"][
                    intermediate_rep_llm[trait][cat]
                ] += value

    if cfg.matching_method == "greedy": # greedy matching
        outcome = greedy_matching(
            human_data,
            demographics_survey,
            trait_of_interest[human_survey],
        )
    elif cfg.matching_method == "hungarian": # maximum weight sum matching
        outcome = maximum_weight_sum_matching(
            human_data,
            demographics_survey,
            trait_of_interest[human_survey],
        )
    elif cfg.matching_method == "random":
        random.seed(42)
        qkey_list = human_data["QKEY"].tolist()
        uid_list = demographics_survey["uid"].tolist()
        uid_list = random.sample(uid_list, len(qkey_list))
        outcome = {
            "qkey_uid_tuple": list(zip(qkey_list, uid_list)),
            "selected_weight_list": [0 for _ in range(len(qkey_list))],
        }
    else:
        raise ValueError(f"Invalid matching method: {cfg.matching_method}")
    qkey_uid_list = outcome["qkey_uid_tuple"]
    edge_weight_list = outcome["selected_weight_list"]
    qkey_uid_list = sorted(
        qkey_uid_list, key=lambda x: edge_weight_list[qkey_uid_list.index(x)]
    )
    edge_weight_list = sorted(edge_weight_list)
    print(f"number of matching pairs: {len(qkey_uid_list)}")
    print(f"Remind: total human respondents: {len(human_data)}")
    print(f"Average edge weight: {np.mean(edge_weight_list)}")

    matched_qkey = [x[0] for x in qkey_uid_list]
    matched_uid = [x[1] for x in qkey_uid_list]

    human_data_selected = human_data_original[
        human_data_original["QKEY"].isin(matched_qkey)
    ]
    demographics_survey_selected = demographics_survey_original[
        demographics_survey_original["uid"].isin(matched_uid)
    ]
    # make an empty dataframe. Include all the columns in the human survey or the demographic survey
    combined_information = pd.DataFrame(
        columns=list(
            set(
                human_data_selected.columns.tolist()
                + demographics_survey_selected.columns.tolist()
            )
        )
    )
    for qkey_uid in qkey_uid_list:
        row_to_add = pd.concat(
            [
                human_data_selected[human_data_selected["QKEY"] == qkey_uid[0]].reset_index(
                    drop=True
                ),
                demographics_survey_selected[
                    demographics_survey_selected["uid"] == qkey_uid[1]
                ].reset_index(drop=True),
            ],
            axis=1,
        )
        combined_information = pd.concat(
            [combined_information, row_to_add], ignore_index=True
        )

    trait_list = "+".join(trait_of_interest["llm"])
    combined_information.to_pickle(
        f"/scratch/data/anthology/outputs/matching_human_virtual/{human_survey}/{cfg.matching_method}/{human_survey}_matching_to_method_{cfg.matching_method}_trait_of_interest_{trait_list}_concatenated_users.pkl"
    )

    qa_prompt_freeform = QA_FREEFORM_PROMPT[human_survey]
    qa_prompt = QA_MCQ_PROMPT[human_survey]
    demographic_trait_list = trait_of_interest[human_survey]
    for prompt_style in ["qa", "qa_freeform", "bio"]:
        demographics_survey = combined_information
        demographics_survey["full_passage"] = ""
        for uid, row in demographics_survey.iterrows():
            random.shuffle(demographic_trait_list)
            demographics_survey.at[uid, "full_passage"] = (
                row["backstory"].strip() + "\n\n"
            )
            if prompt_style == "qa":
                for trait in demographic_trait_list:
                    if (
                        np.isnan(row[trait])
                        or "Refused" in meta.variable_value_labels[trait][int(row[trait])]
                        or "DK US" in meta.variable_value_labels[trait][int(row[trait])]
                    ):
                        continue
                    top_choice = chr(int(row[trait]) - 1 + 65)
                    demographics_survey.at[uid, "full_passage"] += (
                        qa_prompt[trait] + f" ({top_choice})\n\n"
                    )

            elif prompt_style == "qa_freeform":
                for trait in demographic_trait_list:
                    if (
                        np.isnan(row[trait])
                        or "Refused" in meta.variable_value_labels[trait][int(row[trait])]
                        or "DK US" in meta.variable_value_labels[trait][int(row[trait])]
                    ):
                        continue
                    demographics_survey.at[uid, "full_passage"] += (
                        qa_prompt_freeform[trait]
                        + "\n\n"
                        + "Answer: "
                        + value_to_string_map[trait]["choices"][str(int(row[trait]))]
                        + "\n\n"
                    )

            elif prompt_style == "bio":
                demographics_survey.at[
                    uid, "full_passage"
                ] += "Question: Please provide a demographic information of yourself.\n\nAnswer:"
                for trait in demographic_trait_list:
                    if (
                        np.isnan(row[trait])
                        or "Refused" in meta.variable_value_labels[trait][int(row[trait])]
                        or "DK US" in meta.variable_value_labels[trait][int(row[trait])]
                    ):
                        continue
                    top_choice = value_to_string_map[trait]["choices"][str(int(row[trait]))]
                    if trait == "F_CITIZEN_RECODE_FINAL":
                        if top_choice == "Not US Citizen":
                            demographics_survey.at[
                                uid, "full_passage"
                            ] += f" I am not a US citizen."
                        elif top_choice == "US Citizen":
                            demographics_survey.at[
                                uid, "full_passage"
                            ] += f" I am a US citizen."
                    elif trait == llm_column_to_atp_column["income_level_category_13"]:
                        top_choice = top_choice.split("[")[0].strip().lower()
                        demographics_survey.at[
                            uid, "full_passage"
                        ] += f" My annual income is {top_choice}."
                    elif trait == llm_column_to_atp_column["race"]:
                        if top_choice == "Other":
                            demographics_survey.at[
                                uid, "full_passage"
                            ] += f" I consider my race as other."
                        else:
                            demographics_survey.at[
                                uid, "full_passage"
                            ] += f" I consider my race as {top_choice}."
                    elif trait == llm_column_to_atp_column["age_category_4"]:
                        demographics_survey.at[
                            uid, "full_passage"
                        ] += f" My age is {top_choice}."
                    elif trait == llm_column_to_atp_column["gender"]:
                        demographics_survey.at[
                            uid, "full_passage"
                        ] += f" I consider my gender as {top_choice.lower()}."
                    elif trait == llm_column_to_atp_column["education_level"]:
                        demographics_survey.at[
                            uid, "full_passage"
                        ] += f" My highest level of education is {top_choice.lower()}."
                    elif trait == llm_column_to_atp_column["political_affiliation"]:
                        demographics_survey.at[
                            uid, "full_passage"
                        ] += " I consider my political affiliation as "
                        if top_choice == "Democrat" or top_choice == "Republican":
                            demographics_survey.at[
                                uid, "full_passage"
                            ] += f"a {top_choice}."
                        else:
                            if top_choice == "Independent":
                                demographics_survey.at[
                                    uid, "full_passage"
                                ] += f"an {top_choice}."
                            elif top_choice == "Something else":
                                demographics_survey.at[
                                    uid, "full_passage"
                                ] += f"{top_choice.lower()}."
                    elif trait == llm_column_to_atp_column["region"]:
                        demographics_survey.at[
                            uid, "full_passage"
                        ] += f" I live in the {top_choice}."
                    elif trait == llm_column_to_atp_column["religion"]:
                        if top_choice == "Nothing in particular":
                            demographics_survey.at[
                                uid, "full_passage"
                            ] += f" I have no particular religion."
                        elif top_choice == "Something else":
                            demographics_survey.at[uid, "full_passage"] += ""
                        else:
                            demographics_survey.at[
                                uid, "full_passage"
                            ] += f" I am a {top_choice}."
                    else:
                        raise ValueError(f"trait {trait} not found")
                demographics_survey.at[uid, "full_passage"] += "\n\n"

            else:
                raise ValueError(f"Invalid prompt style: {prompt_style}")
        # save the file to dataframe
        demographics_survey.to_csv(
            f"/scratch/data/anthology/outputs/matching_human_virtual/{human_survey}/{cfg.matching_method}/{human_survey}_matching_to_method_{cfg.matching_method}_trait_of_interest_{trait_list}_{prompt_style}_appended_backstory.csv",
            index=False,
        )
        
    return


if __name__ == "__main__":
    my_app()