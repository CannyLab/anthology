import os
import pathlib
from datetime import datetime
import pandas as pd

from typing import Union

def get_config_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Get the path to the config file

    Args:
        path (Union[str, pathlib.Path]): Path to the config file

    Returns:
        pathlib.Path: Path to the config file
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    
    return pathlib.Path(__file__).resolve().parents[2] / "configs" / path


def save_result_to_csv(output: dict, output_data_path: str):
    df = pd.DataFrame.from_dict(output, orient="index")
    df.reset_index(inplace=True, drop=True)
    df.to_csv(output_data_path)


def save_result_to_pkl(output: dict, output_data_path: str):
    df = pd.DataFrame.from_dict(output, orient="index")
    df.to_pickle(output_data_path)


def publish_result(output: Union[dict, pd.DataFrame], publish_dir: str, filename: str):
    if not os.path.isdir(publish_dir):
        os.mkdir(publish_dir)
    publish_result_path = os.path.join(publish_dir, filename)
    if os.path.exists(publish_result_path):
        # Append timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if filename.endswith(".csv"):
            filename = filename.split(".")[0]
            new_filename = f"{filename}_{timestamp}.csv"
        elif filename.endswith(".pkl") or filename.endswith(".pickle"):
            filename = filename.split(".")[0]
            new_filename = f"{filename}_{timestamp}.pkl"

        publish_result_path = os.path.join(publish_dir, new_filename)

    if isinstance(output, dict):
        if publish_result_path.endswith(".csv"):
            save_result_to_csv(output, publish_result_path)
        elif publish_result_path.endswith(".pkl"):
            save_result_to_pkl(output, publish_result_path)
        else:
            raise ValueError(
                f"Unsupported file format: {os.path.basename(publish_result_path)}"
            )

    elif isinstance(output, pd.DataFrame):
        if publish_result_path.endswith(".csv"):
            output.to_csv(publish_result_path)
        elif publish_result_path.endswith(".pkl"):
            output.to_pickle(publish_result_path)
        else:
            raise ValueError(
                f"Unsupported file format: {os.path.basename(publish_result_path)}"
            )

    else:
        raise ValueError(f"Unsupported output type: {type(output)}")

    os.chmod(publish_result_path, 0o777)
