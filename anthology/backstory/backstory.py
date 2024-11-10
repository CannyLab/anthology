import pandas as pd
import pathlib


def load_backstory(
    backstory_path: str,
    backstory_type: str,
    num_backstory: int = -1,
    shuffle_backstory: bool = True,
):
    # check file exists in the path
    assert pathlib.Path(backstory_path).exists(), "Backstory file does not exist"

    backstory_file_name = pathlib.Path(backstory_path).name

    if backstory_file_name.endswith(".csv"):
        backstory_df = pd.read_csv(backstory_path)
    elif backstory_file_name.endswith(".pkl") or backstory_file_name.endswith(
        ".pickle"
    ):
        backstory_df = pd.read_pickle(backstory_path)

    if "Unnamed: 0" in backstory_df.columns:
        backstory_df.drop(columns=["Unnamed: 0"], inplace=True)

    if "full_passage" in backstory_df.columns:
        backstory_df = backstory_df.dropna(subset=["full_passage"])

    # remove the text that is too short
    if "backstory_generated_token_length" in backstory_df.columns:
        backstory_df = backstory_df[
            backstory_df["backstory_generated_token_length"] > 40
        ]
    backstory_df = backstory_df.reset_index(drop=True)

    if num_backstory > 0 and len(backstory_df) > num_backstory:
        if shuffle_backstory:
            backstory_df = backstory_df.sample(n=num_backstory).reset_index(drop=True)
        else:
            backstory_df = backstory_df.iloc[:num_backstory].reset_index(drop=True)

    if "uid" in backstory_df.columns:
        backstory_df = backstory_df.sort_values(by="uid").reset_index(drop=True)

    return backstory_df
