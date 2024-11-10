import logging
import hydra
import pandas as pd
import os

from omegaconf import OmegaConf
from tqdm import tqdm

from anthology.utils.random_utils import set_random_seed
from anthology.utils.data_utils import get_config_path
from anthology.analysis.atp import (
    draw_fine_grained_responses,
    draw_heatmap,
    get_compliance_rate,
    get_cronbach_alpha,
    get_human_data,
    get_human_response,
    get_weighted_cov,
    compare_human_to_model,
    remove_non_compliant_responses,
    distance_lower_bound,
    matrix_dist_lower_bound,
    cov_matrix_distance,
    parse_questions,
    load_data,
)

log = logging.getLogger(__name__)

# config file path
config_path = get_config_path("analysis")


@hydra.main(config_path=str(config_path), config_name="ATP_analysis")
def my_app(cfg):
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    set_random_seed(cfg.random_seed)

    # get survey config
    survey_cfg = cfg.questionnaire.ATP

    # setup important variables
    # parse questions and likert scale
    question_of_interest, question_wordking, likert_scale_dict = parse_questions(
        survey_cfg
    )

    # human data weight
    weight = f"WEIGHT_W{survey_cfg.wave}"

    # save path
    experiment_result_file_path = f"{cfg.save_dir}/{cfg.experiment_name}"

    if cfg.output_data_name:
        experiment_result_file_name = (
            f"{cfg.output_data_name}_{cfg.output_time_stamp}.xlsx"
        )
    else:
        experiment_result_file_name = f"atp_w{survey_cfg.wave}_analysis"
        experiment_result_file_name = (
            f"{experiment_result_file_name}_{cfg.output_time_stamp}.xlsx"
        )

    # get experiment data
    # path to experiment repository
    experiment_repository = cfg.experiment_repository
    # experiment name
    experiment_name = cfg.experiment_name

    # download experiment repository from gdrive
    if cfg.use_gdrive:
        print("Downloading experiment repository from gdrive...")
        os.system(
            f"gdrive files export 1lEbKesJy32b4c_GjjI2TLzk8-ZSpMnmTvZdG0UwJg8o {experiment_repository} --overwrite"
        )
        os.chmod(experiment_repository, 0o777)

    experiment_repo_df = pd.read_excel(
        experiment_repository, sheet_name="experiments_final"
    )

    w99_experiment = experiment_repo_df[
        experiment_repo_df["experiment"] == experiment_name
    ]

    # get experiment data
    experiment_df_dict = {}
    for experiment in tqdm(w99_experiment.iterrows()):
        model = experiment[1]["model"]
        method = experiment[1]["method"]
        method_detail = experiment[1]["method_detail"]
        temperature = experiment[1]["temperature"]
        file_directory = experiment[1]["file directory"]
        demographic_variable = experiment[1]["demographic variable"]
        note = (
            str(experiment[1]["matching"])
            if not pd.isna(experiment[1]["matching"])
            else ""
        )
        exp_path = experiment[1]["file directory"]

        # Skip non-pickle files or incomplete experiments
        try:
            if not os.path.exists(exp_path) or not exp_path.endswith(".pkl"):
                continue
        except:
            continue

        result = load_data(exp_path)

        compliance_rate = get_compliance_rate(result, question_of_interest)[0]
        result = remove_non_compliant_responses(result, question_of_interest)

        experiment_dict = {
            "model": model,
            "method": method,
            "method_detail": method_detail,
            "temperature": temperature,
            "file_directory": file_directory,
            "demographic_variable": demographic_variable,
            "note": note,
            "compliance_rate": compliance_rate,
            "result": result,
        }

        key = f"{model}_{method}_{method_detail}_{temperature}"

        if demographic_variable:
            key = f"{key}_{demographic_variable}"

        if note == "nan":
            note = ""

        else:
            note = note.replace(" ", "_")
            note = note.replace(",", "_")
            key = f"{key}_{note}"

        experiment_df_dict[key] = experiment_dict

    # get human data
    human_data_path = survey_cfg.human_data_path

    human_data = get_human_data(human_data_path)
    human_data = remove_non_compliant_responses(human_data, question_of_interest)

    # Demographic variables
    demographic_variables = {}
    assert survey_cfg.demographics_metadata_path.endswith(".json")
    demographics_df = pd.read_json(survey_cfg.demographics_metadata_path)
    for col_name in demographics_df.columns:
        variable = demographics_df[col_name].topic
        choices = demographics_df[col_name].choices
        demographic_variables[variable] = {
            "column_id": col_name,
            "choices": {
                int(k): v
                for k, v in choices.items()
                if int(k) > 0 and "Refused" not in v
            },
        }

    # if subgroup is specified, filter human data
    subgroup = cfg.subgroup
    assert subgroup in list(demographic_variables.keys()) + [
        "all"
    ], f"Invalid subgroup {subgroup}"

    choices = {}

    subgroup_key = ""
    if subgroup != "all":
        subgroup_key = demographic_variables[subgroup]["column_id"]
        choices = demographic_variables[subgroup]["choices"]
    else:
        choices = {"all": "all"}
        subgroup_key = ""
    if subgroup_key:
        log.info(
            f"Target subgroup for analysis: {subgroup} with {subgroup_key} and choices {choices}"
        )
    else:
        log.info("Analyzing all data")

    for choice_idx, choice_str in choices.items():
        choice_str = (
            choice_str.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("'s", "")
            .replace(",", "")
        )
        print(f"Analyzing subgroup {choice_str}...")
        exp_dist_dict = {}
        human_data_slice = human_data
        if subgroup_key:
            human_data_slice = human_data_slice[
                human_data_slice[subgroup_key] == choice_idx
            ]
        if human_data_slice.empty:
            print(f"Subgroup {choice_str} is empty, skipping...")
            continue

        human_dist_dict = get_human_response(
            human_data_slice, question_of_interest, human_data_slice[weight]
        )[0]
        human_data_slice_X = human_data_slice[question_of_interest].values
        human_data_slice_w = human_data_slice[weight].values
        _, human_weighted_cov = get_weighted_cov(
            human_data_slice_X, human_data_slice_w, corr=True
        )
        dist_lower_bound = distance_lower_bound(
            human_data_slice,
            question_of_interest,
            weight_column=weight,
            dist_type=cfg.dist_type,
        )

        for key in tqdm(experiment_df_dict.keys()):
            experiment = experiment_df_dict[key]["result"]
            if subgroup_key:
                if subgroup_key not in experiment.columns:
                    experiment = experiment[experiment[cfg.subgroup] == choice_idx]
                else:
                    experiment = experiment[experiment[subgroup_key] == choice_idx]

            dist_dict = compare_human_to_model(
                human_dist_dict,
                experiment,
                question_of_interest,
                dist_type=cfg.dist_type,
                likert_scale_dict=likert_scale_dict,
            )
            exp_dist_dict[key] = dist_dict
            exp_dist_dict[key]["cbs"] = cov_matrix_distance(
                human_weighted_cov,
                experiment[question_of_interest].corr().values,
                dist_type="CBS",
            )
            exp_dist_dict[key]["frobenius"] = cov_matrix_distance(
                human_weighted_cov,
                experiment[question_of_interest].corr().values,
                dist_type="Frobenius",
            )
            exp_dist_dict[key]["l1"] = cov_matrix_distance(
                human_weighted_cov,
                experiment[question_of_interest].corr().values,
                dist_type="L1",
            )
            exp_dist_dict[key]["cronbach_alpha"] = get_cronbach_alpha(
                experiment, question_of_interest
            )

            exp_dist_dict[key]["method"] = experiment_df_dict[key]["method"]
            exp_dist_dict[key]["method_detail"] = experiment_df_dict[key][
                "method_detail"
            ]
            exp_dist_dict[key]["trait"] = experiment_df_dict[key][
                "demographic_variable"
            ]
            exp_dist_dict[key]["file_directory"] = experiment_df_dict[key][
                "file_directory"
            ]
            exp_dist_dict[key]["model"] = experiment_df_dict[key]["model"]
            exp_dist_dict[key]["temp"] = experiment_df_dict[key]["temperature"]
            exp_dist_dict[key]["note"] = experiment_df_dict[key]["note"]

        exp_dist_dict["human"] = dist_lower_bound
        exp_dist_dict["human"]["cbs"] = matrix_dist_lower_bound(
            human_data, question_of_interest, wave=survey_cfg.wave, dist_type="CBS"
        )
        exp_dist_dict["human"]["frobenius"] = matrix_dist_lower_bound(
            human_data,
            question_of_interest,
            wave=survey_cfg.wave,
            dist_type="Frobenius",
        )
        exp_dist_dict["human"]["l1"] = matrix_dist_lower_bound(
            human_data, question_of_interest, wave=survey_cfg.wave, dist_type="L1"
        )
        exp_dist_dict["human"]["cronbach_alpha"] = get_cronbach_alpha(
            human_data, question_of_interest
        )

        exp_dist_df = pd.DataFrame(exp_dist_dict).T
        exp_dist_df = exp_dist_df.sort_values(
            by=f"average {cfg.dist_type}", ascending=True
        )

        if not os.path.exists(experiment_result_file_path):
            os.makedirs(experiment_result_file_path)
        filename = (
            experiment_result_file_name
            if cfg.subgroup == "all"
            else f"{experiment_result_file_name.split('.xlsx')[0]}_{cfg.subgroup}_{choice_str}.xlsx"
        )
        with pd.ExcelWriter(f"{experiment_result_file_path}/{filename}") as writer:
            exp_dist_df.to_excel(writer)

        if cfg.use_gdrive:
            print("Uploading analysis result to gdrive...")
            os.system(
                f"gdrive files upload --parent 1j6v_YQov0-mvltVyVZf8GRhey5-jgb2R {experiment_result_file_path}/{experiment_result_file_name}"
            )

        print("Done!")

        # save correlation matrix heatmap
        save_path = (
            f"{experiment_result_file_path}/corr_heatmap_{cfg.experiment_name}"
            if cfg.subgroup == "all"
            else f"{experiment_result_file_path}/corr_heatmap_{cfg.experiment_name}_{cfg.subgroup}_{choice_str}"
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for key in tqdm(experiment_df_dict.keys()):
            experiment = experiment_df_dict[key]["result"]
            if subgroup_key:
                if subgroup_key not in experiment.columns:
                    experiment = experiment[experiment[cfg.subgroup] == choice_idx]
                else:
                    experiment = experiment[experiment[subgroup_key] == choice_idx]
            title = f"{key}"
            file_name = f"{key}.png"

            human_corr_matrix = human_data[question_of_interest].corr()
            model_corr_matrix = experiment[question_of_interest].corr()

            draw_heatmap(
                human_corr_matrix, model_corr_matrix, title, save_path, file_name
            )

        print("Done!")

        if cfg.use_gdrive:
            print("Uploading correlation heatmap to gdrive...")
            os.system(
                f"gdrive files upload --parent 1j6v_YQov0-mvltVyVZf8GRhey5-jgb2R --recursive {save_path}"
            )

        save_path = (
            f"{experiment_result_file_path}/fine_grained_responses_{cfg.experiment_name}"
            if cfg.subgroup == "all"
            else f"{experiment_result_file_path}/fine_grained_responses_{cfg.experiment_name}_{cfg.subgroup}_{choice_str}"
        )

        for key in tqdm(experiment_df_dict.keys()):
            experiment = experiment_df_dict[key]["result"]
            if subgroup_key:
                if subgroup_key not in experiment.columns:
                    experiment = experiment[experiment[cfg.subgroup] == choice_idx]
                else:
                    experiment = experiment[experiment[subgroup_key] == choice_idx]
            title = f"{key}"
            file_name = f"{key}"

            draw_fine_grained_responses(
                human_dist_dict,
                experiment,
                question_of_interest,
                likert_scale_dict=likert_scale_dict,
                question_key_to_wording=question_wordking,
                save_path=save_path,
                file_name=file_name,
            )

        print("Done!")
        if cfg.use_gdrive:
            print("Uploading fine grained responses to gdrive...")
            os.system(
                f"gdrive files upload --parent 1j6v_YQov0-mvltVyVZf8GRhey5-jgb2R --recursive {save_path}"
            )


if __name__ == "__main__":
    my_app()
