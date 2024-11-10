import pandas as pd
from copy import deepcopy
from anthology.utils.survey_utils import generate_answer_forcing_prompt


def generate_baseline_prompt_prefix(
    survey_data_df: pd.DataFrame,  # contains the respondent info
    demographics_df: pd.DataFrame,  # contains demographic questions info
    prompt_style: str,
    include_answer_forcing: bool = True,
) -> list:
    """
    Return value: a list of dictionaries, where each dict contains the following:
        - "uid": a person's uid based on the row index in human survey data
        - "demographic_info": a person's demographic information
        - "full_passage": baseline prompt
    """
    prompt_prefix_list = []

    for survey_idx, row in survey_data_df.iterrows():  # choose each respondent
        demographic_info = {}
        if prompt_style in ["nothing", "qa"]:
            prefix = ""
        elif prompt_style == "bio":
            prefix = "Below you will be asked to provide a short description of your demographic information, and then answer some questions.\n\nDescription:"
        elif prompt_style == "portray":
            prefix = "Answer the following questions as if you are a person with the following demographic information provided below.\n"

        # shuffle the order of demographic variables fed into the baseline prompt
        demographics_df_copy = deepcopy(demographics_df)
        shuffled_columns = (
            demographics_df_copy.columns.to_series().sample(frac=1).tolist()
        )
        demographics_df_copy = demographics_df_copy[shuffled_columns]

        for demographic_var in demographics_df_copy.columns:  # choose demographic trait
            # retreive demographic trait for current respondent
            topic = demographics_df_copy[demographic_var].topic
            question = demographics_df_copy[demographic_var].question_body
            choices = demographics_df_copy[demographic_var].choices
            try:
                answer = int(row[demographic_var])
            except:
                continue
            # find a string in choices whose key matches the answer
            try:
                string_answer = choices[str(answer)]
                if "Refused" in string_answer:
                    continue
            except:
                continue

            #### generate rule-based prefix according to the prompt_style ####
            if prompt_style == "qa":
                converted_answer_choice = chr(answer + 64)
                # generate a formatted question
                mcq_choice_str = ""
                available_choices = []
                # iterate over choices and generate formatted string
                for option_idx, choice in choices.items():
                    if "Refused" in choice:
                        continue
                    option_idx_int = int(option_idx)
                    choice_symbol = chr(int(option_idx) + 64)
                    if option_idx_int >= 26:
                        choice_symbol = f"A{chr(int(option_idx) - 26 + 65)}"
                    mcq_choice_str += f"({choice_symbol}) {choice}\n"
                    available_choices.append(choice_symbol)
                mcq_choice_str = mcq_choice_str.strip()
                answer_forcing_prompt = (
                    generate_answer_forcing_prompt(len(available_choices))
                    if include_answer_forcing
                    else ""
                )
                formatted_question = f"Question: {question}\n{mcq_choice_str}\n{answer_forcing_prompt}\nAnswer:"

                # add formatted question to prefix
                prefix += f"{formatted_question} ({converted_answer_choice})\n\n"

            elif prompt_style == "portray":
                topic_replaced = topic.replace("_", " ")
                prefix += f"{topic_replaced}: {string_answer}\n"

            elif prompt_style == "bio":
                if topic == "race":
                    if "Other" in string_answer:
                        string_answer = "other"
                    demographic_portrayal = f" I consider my race as {string_answer}."
                elif topic == "gender":
                    demographic_portrayal = (
                        f" I consider my gender as {string_answer.lower()}."
                    )
                elif topic == "age":
                    demographic_portrayal = f" My age is {string_answer}."
                elif topic == "education":
                    string_answer = string_answer.lower()
                    demographic_portrayal = (
                        f" My highest level of education is {string_answer}."
                    )
                elif topic == "income":
                    demographic_portrayal = f" My annual income is {string_answer}."
                elif topic == "political_affiliation":
                    demographic_portrayal = " I consider my political affiliation as "
                    if string_answer == "Democrat" or string_answer == "Republican":
                        demographic_portrayal += f"a {string_answer}."
                    else:
                        if string_answer == "Independent":
                            demographic_portrayal += f"an {string_answer}."
                        elif string_answer == "Something else":
                            demographic_portrayal += f"{string_answer.lower()}."
                elif topic == "region":
                    demographic_portrayal = f" I live in the {string_answer}."
                elif topic == "religion":
                    if string_answer in "Nothing in particular":
                        demographic_portrayal = " I have no particular religion."
                    elif string_answer == "Something else":
                        demographic_portrayal = ""
                    else:
                        demographic_portrayal = f" I am a {string_answer}."
                prefix += demographic_portrayal

            demographic_info[f"{demographic_var}"] = answer

        demographic_info["QKEY"] = row["QKEY"]
        demographic_info["full_passage"] = prefix.strip() + "\n\n"
        prompt_prefix_list.append(demographic_info)

    return prompt_prefix_list
