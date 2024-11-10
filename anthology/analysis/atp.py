import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
import math

from pingouin import cronbach_alpha
from scipy.stats import wasserstein_distance
from omegaconf import OmegaConf, DictConfig


def draw_heatmap(
    human_corr_matrix: pd.DataFrame,
    model_corr_matrix: pd.DataFrame,
    title: str,
    save_path: str,
    file_name: str,
):
    """
    Draw and save a heatmap comparing the correlation matrix of
    human and model data

    Args:
        human_corr_matrix (pd.DataFrame): correlation matrix of human data
        model_corr_matrix (pd.DataFrame): correlation matrix of model data
        title (str): title of the heatmap
        save_path (str): path to save the heatmap
        file_name (str): name of the file to save the heatmap

    Returns:
        None
    """
    # check file name in save_path exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    sns.heatmap(
        human_corr_matrix, cmap="YlGnBu", annot=True, vmin=-0.4, vmax=1, ax=ax[0]
    )
    sns.heatmap(
        model_corr_matrix, cmap="YlGnBu", annot=True, vmin=-0.4, vmax=1, ax=ax[1]
    )

    ax[0].set_title("Human")
    ax[1].set_title("Model")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{file_name}")
    plt.close()


def get_human_data(path: str) -> pd.DataFrame:
    """
    Read the human data from the path and rename the columns
    according to the mapping provided in map_human_to_model

    Args:
        path (str): path to the human data. The data should be in .sav format

    Returns:
        pd.DataFrame: human data with columns renamed according to the mapping
    """
    assert path.endswith(".sav"), "The file should be in .sav format"

    df, meta = pyreadstat.read_sav(path)
    # df = df.rename(columns=map_human_to_model)

    return df


def load_data(path: str) -> pd.DataFrame:
    """
    Load the survey result data performed on the LLMs.

    Args:
        path (str): path to the data file

    Returns:
        pd.DataFrame: data loaded from the path
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    else:
        raise ValueError("The file should be in CSV or pickle format")


def get_cronbach_alpha(data: pd.DataFrame, question_of_interest: list[str]) -> float:
    """
    Calculate the cronbach alpha for the questions of interest

    Args:
        data (pd.DataFrame): data to calculate cronbach alpha
        question_of_interest (list[str]): list of questions to calculate cronbach alpha

    Returns:
        float: cronbach alpha value
    """
    return cronbach_alpha(data[question_of_interest])[0]


def compare_human_to_model(
    human_data_dict: dict,
    model_data: pd.DataFrame,
    question_of_interest: list[str],
    likert_scale_dict: dict[str, int] = {},
    dist_type: str = "EMD",
) -> dict:
    """
    Compare the human survey data to the model survey data using the distributional distance metric
    specified by dist_type. The distance metric can be either "TV" or "EMD".

    Args:
        human_data_dict (dict): human survey data
        model_data (pd.DataFrame): model survey data
        question_of_interest (list[str]): list of questions to compare
        likert_scale_dict (dict): dictionary mapping question to likert scale
        dist_type (str): distance metric to use. Can be either "TV" or "EMD"

    Returns:
        dist_dict (dict): dictionary containing the distance between human and model data
    """
    dist_dict = {}

    for q in question_of_interest:
        if q in likert_scale_dict:
            likert_scale = likert_scale_dict[q]
        else:
            raise ValueError(f"Likert scale for question {q} not found")

        llm_response_distribution = (
            model_data[q].value_counts(normalize=True).sort_index()
        )

        # fill in missing values
        if len(llm_response_distribution) < likert_scale:
            llm_response_distribution = llm_response_distribution.reindex(
                range(1, likert_scale + 1), fill_value=0
            )

        try:
            if dist_type == "TV":
                dist = (
                    0.5
                    * np.abs(
                        human_data_dict[q] - llm_response_distribution.values
                    ).sum()
                )
            elif dist_type == "EMD":
                indices = list(llm_response_distribution.index)
                dist = wasserstein_distance(
                    indices,
                    indices,
                    human_data_dict[q],
                    llm_response_distribution.values,
                )
            else:
                raise ValueError(f"Distance type {dist_type} not supported")
        except:
            import pdb

            pdb.set_trace()

        dist_dict[q] = dist

    dist_dict[f"average {dist_type}"] = np.mean(list(dist_dict.values()))
    return dist_dict


def draw_fine_grained_responses(
    human_data_dict: dict,
    model_data: pd.DataFrame,
    question_of_interest: list[str],
    question_key_to_wording: dict[str, str] = {},
    likert_scale_dict: dict[str, int] = {},
    save_path: str = "",
    file_name: str = "",
):
    """
    Draw and save the fine-grained responses for the questions of interest

    Args:
        human_data_dict (dict): human survey data
        model_data (pd.DataFrame): model survey data
        question_of_interest (list[str]): list of questions to draw
        likert_scale_dict (dict): dictionary mapping question to likert scale
        save_path (str): path to save the fine-grained responses
        file_name (str): name of the file to save the fine-grained responses

    Returns:
        None
    """
    width = 0.2

    for q in question_of_interest:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        likert_scale = likert_scale_dict[q]

        human_num = 2500  # len(human_data_dict[q])
        llm_num = len(model_data[q])

        llm_response_distribution = (
            model_data[q].value_counts(normalize=True).sort_index()
        )

        if len(llm_response_distribution) < likert_scale:
            llm_response_distribution = llm_response_distribution.reindex(
                range(1, likert_scale + 1),
                fill_value=0,
            ).values

        else:
            llm_response_distribution = llm_response_distribution.values

        human_response_distribution = human_data_dict[q]

        llm_response_dist_std = np.sqrt(
            llm_response_distribution * (1 - llm_response_distribution) / llm_num
        )
        human_response_dist_std = np.sqrt(
            human_response_distribution * (1 - human_response_distribution) / human_num
        )

        ax.bar(
            np.arange(1, likert_scale + 1) - width / 2,
            llm_response_distribution,
            width,
            yerr=llm_response_dist_std,
            label="Model",
        )

        ax.bar(
            np.arange(1, likert_scale + 1) + width / 2,
            human_response_distribution,
            width,
            yerr=human_response_dist_std,
            label="Human",
        )

        ax.set_ylim(0, 0.8)
        ax.legend(loc="upper left")
        ax.grid("on")
        ax.set_xticks(np.arange(1, likert_scale + 1))

        title = question_key_to_wording.get(q, q)

        # split if title is too long
        if len(title.split(" ")) > 5:
            title = title.split(" ")
            title = (
                " ".join(title[: len(title) // 2])
                + "\n"
                + " ".join(title[len(title) // 2 :])
            )
        ax.set_title(title)

        save_path_with_q = f"{save_path}/{q}"
        if save_path_with_q:
            if not os.path.exists(save_path_with_q):
                os.makedirs(save_path_with_q)
            plt.savefig(f"{save_path_with_q}/{file_name}.png")
            plt.close()


def get_compliance_rate(
    df: pd.DataFrame, question_of_interest: list[str]
) -> tuple[np.array, np.array]:
    """
    Get language model compliance rate for each question

    Args:
        df: DataFrame
        question_of_interest: list of questions
    Returns:
        average compliance rate, standard error
    """
    mean_list = []

    for key in question_of_interest:
        compliance_key = key + "_compliance"
        mean_list.append(df[compliance_key].mean())

    return np.mean(mean_list), np.std(mean_list) / math.sqrt(len(mean_list))


def remove_non_compliant_responses(
    df: pd.DataFrame, question_of_interest: list
) -> pd.DataFrame:
    """Remove language model's non-compliant responses from the dataframe

    Args:
        df: DataFrame

    Returns:
        DataFrame

    """
    new_df = df[question_of_interest]
    condition = new_df.apply(lambda x: x < 0, axis=1)
    df[condition] = np.nan

    condition = new_df.apply(
        lambda x: x > 90, axis=1
    )  # refused/non_compliant answer for ATP survey
    df[condition] = np.nan

    return df


def get_human_response(
    df: pd.DataFrame,
    question_of_interest: list[str],
    weight_data: pd.DataFrame,
) -> tuple[dict[str, np.array], dict[str, np.array]]:
    """Get the weighted human response distribution for each question

    Args:
        df (pd.DataFrame): human data
        question_of_interest (list[str]): list of questions to calculate the response for
        weight_data (pd.DataFrame): weight data

    Returns:
        tuple[dict[str, np.array], dict[str, np.array]]: avg_response_dict, std_err_dict
    """
    avg_response_dict, std_err_dict = {}, {}

    for question in question_of_interest:
        response_dist, response_std_err = get_human_response_distribution(
            df[question], weight_data
        )

        avg_response_dict[question] = response_dist
        std_err_dict[question] = response_std_err

    return avg_response_dict, std_err_dict


def get_human_response_distribution(
    df: pd.DataFrame, weight_data: pd.DataFrame, renormalize: bool = True
) -> tuple[np.array, np.array]:
    """Calculate the weighted distribution of the human responses

    Args:
        df (pd.DataFrame): human data
        weight_data (pd.DataFrame): weight data
        renormalize (bool, optional): renormalize the distribution. Defaults to True.

    Returns:
        tuple[np.array, np.array]: response_ratio_arr, response_std_err
    """
    data = df
    n = weight_data.sum()

    # the user response is in the form of a number
    response_num = data.value_counts().sort_index().index.astype(int).tolist()
    response_ratio_list = []
    response_std_err_list = []

    for num in response_num:
        nominal_index = df[data == num].index

        response_ratio = weight_data[nominal_index].sum() / n
        std_err = np.sqrt(response_ratio * (1 - response_ratio) / n)

        response_ratio_list.append(response_ratio)
        response_std_err_list.append(std_err)

    if renormalize:
        total = sum(response_ratio_list)

        response_ratio_list = [ratio / total for ratio in response_ratio_list]
        response_std_err_list = [
            std_err / np.sqrt(total) for std_err in response_std_err_list
        ]

    response_ratio_arr = np.array(response_ratio_list)
    response_std_err = np.array(response_std_err_list)

    return response_ratio_arr, response_std_err


def get_statistical_distance(data1: np.array, data2: np.array, dist_type: str = "EMD"):
    """Calculate the statistical distance between two distributions

    Args:
        data1 (np.array): distribution 1
        data2 (np.array): distribution 2
        dist_type (str, optional): distance type. Defaults to earth mover's distance (EMD). Possible values: earth mover's distance (EMD), total variation (TV).

    Raises:
        NotImplementedError: Invalid distance type, choose from earth mover's distance (EMD) or total variation (TV)

    Returns:
        float: statistical distance between the two distributions
    """
    if dist_type == "EMD":
        dist1 = np.arange(1, len(data1) + 1)
        dist2 = np.arange(1, len(data2) + 1)

        return wasserstein_distance(dist1, dist2, data1, data2)

    elif dist_type == "TV":
        return 0.5 * np.abs(data1 - data2).sum()

    else:
        raise NotImplementedError


def distance_lower_bound(
    human_data: pd.DataFrame,
    question_of_interest: list[str],
    num_iter: int = 20,
    weight_column: str = "WEIGHT_W34",
    dist_type: str = "EMD",
) -> dict[str, float]:
    """Calculate the lower bound of the distance between two distributions
    Divide the human data into two parts randomly, calculate the distance between the two distributions for each question, and estimate the average distance

    Args:
        human_data (pd.DataFrame): human data
        question_of_interest (list[str]): list of questions to calculate the distance for
        num_iter (int, optional): number of samples to estimate the average distance. Defaults to 20.
        weight_column (str, optional): weight column. Defaults to "WEIGHT_W34".
        dist_type (str, optional): distance type. Defaults to "EMD". Possible values: EMD, TV.

    Returns:
        dist_dict_avg (dict): dictionary containing the average distance between the two distributions for each question
    """
    dist_list = []

    for _ in range(num_iter):
        # randomly divide human_data equally
        # Shuffle the DataFrame and reset the index
        df_shuffled = human_data.sample(frac=1).reset_index(drop=True)

        # Calculate the split index
        split_index = len(df_shuffled) // 2

        # Split the DataFrame into two parts
        human_data_part1 = df_shuffled.iloc[:split_index]
        human_data_part2 = df_shuffled.iloc[split_index:]

        part1_dist_dict = get_human_response(
            human_data_part1,
            question_of_interest,
            human_data_part1[weight_column],
        )[0]

        part2_dist_dict = get_human_response(
            human_data_part2,
            question_of_interest,
            human_data_part2[weight_column],
        )[0]

        dist_dict = {}
        for q in question_of_interest:
            dist1 = part1_dist_dict[q]
            dist2 = part2_dist_dict[q]

            distance = get_statistical_distance(dist1, dist2, dist_type)

            dist_dict[q] = distance

        dist_list.append(dist_dict)

    dist_df = pd.DataFrame(dist_list)

    dist_dict_avg = dist_df.mean(axis=0).to_dict()
    dist_dict_avg[f"average {dist_type}"] = dist_df.mean().mean()

    return dist_dict_avg


def get_weighted_cov(
    X: np.array, w: np.array, corr: bool = False
) -> tuple[np.array, np.array]:
    """Calculate the weighted covariance matrix

    Args:
        X (np.array): data
        w (np.array): weights
        corr (bool, optional): calculate the correlation matrix. Defaults to False.

    Returns:
        mean (np.array): mean of the data
        cov (np.array): covariance matrix of the data (or correlation matrix if corr=True)
    """
    # normalize the weights
    w = w / w.sum()

    # nan values in X will cause the covariance to be nan
    # remove nan values from X and w
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    w = w[mask]

    # calculate the weighted mean
    mean = np.average(X, weights=w, axis=0)

    cov = np.cov(X, aweights=w, rowvar=False)

    if corr:
        v = np.sqrt(np.diag(cov))
        outer_v = np.outer(v, v)
        cov = cov / outer_v

    return mean, cov


def cov_matrix_distance(cov1: np.array, cov2: np.array, dist_type="CBS") -> float:
    """Calculate the distance between two covariance matrices

    Args:
        cov1 (np.array): covariance matrix 1
        cov2 (np.array): covariance matrix 2
        loss_type (str, optional): loss type. Defaults to "CBS". Possible values: CBS, Frobenius, L1.

    Returns:
        float: distance between the two covariance matrices
    """
    if dist_type == "CBS":
        return 1 - np.trace(cov1 @ cov2) / (np.linalg.norm(cov1) * np.linalg.norm(cov2))
    elif dist_type == "Frobenius":
        return np.sqrt(((cov1 - cov2) ** 2).sum())
    elif dist_type == "L1":
        return np.abs(cov1 - cov2).sum()


def matrix_dist_lower_bound(
    human_data: pd.DataFrame,
    question_of_interest: list[str],
    wave: int = 34,
    num_iter: int = 20,
    dist_type: str = "CBS",
) -> float:
    """Calculate the lower bound of the distance between two weighted covariance matrices
    Divide the human data into two parts randomly, calculate the weighted covariance matrix for each part, and calculate the distance between the two covariance matrices
    Repeat the process num_iter times and estimate the average distance between the two covariance matrices

    Args:
        human_data (pd.DataFrame): human data
        question_of_interest (list[str]): list of questions to calculate the covariance matrix for
        num_iter (int, optional): number of samples to estimate the average distance. Defaults to 20.
        dist_type (str, optional): distance type. Defaults to "CBS". Possible values: CBS, Frobenius, L1.

    Returns:
        float: average distance between the two covariance matrices
    """
    dist_list = []

    for _ in range(num_iter):
        # divide human data into two parts randomly
        # Shuffle the DataFrame and reset the index
        df_shuffled = human_data.sample(frac=1).reset_index(drop=True)

        # Calculate the split index
        split_index = len(df_shuffled) // 2

        # Split the DataFrame into two parts
        human_data_part1 = df_shuffled.iloc[:split_index]
        human_data_part2 = df_shuffled.iloc[split_index:]

        human_data_part1_X = human_data_part1[question_of_interest].values
        human_data_part1_w = human_data_part1[f"WEIGHT_W{wave}"].values

        human_data_part2_X = human_data_part2[question_of_interest].values
        human_data_part2_w = human_data_part2[f"WEIGHT_W{wave}"].values

        _, human_data_part1_weighted_cov = get_weighted_cov(
            human_data_part1_X, human_data_part1_w, corr=True
        )
        _, human_data_part2_weighted_cov = get_weighted_cov(
            human_data_part2_X, human_data_part2_w, corr=True
        )

        # calculate the distance between the two weighted covariance matrices
        distance = cov_matrix_distance(
            human_data_part1_weighted_cov, human_data_part2_weighted_cov, dist_type
        )

        dist_list.append(distance)

    return np.mean(dist_list)


def parse_questions(
    cfg: DictConfig,
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """Parse the questions from the config file

    Args:
        cfg (DictConfig): config file

    Returns:
        questions (list[str]): list of questions
        question_wording (dict[str, str]): dictionary mapping question to question wording
        likert_scale_dict (dict[str, int]): dictionary mapping question to likert scale
    """
    questions = []
    question_wording = {}
    likert_scale_dict = {}

    for qkey in cfg.questions:
        questions.append(qkey)

        question_wording[qkey] = cfg.questions[qkey].question_body
        likert_scale_dict[qkey] = len(cfg.questions[qkey].choices)

    return questions, question_wording, likert_scale_dict
