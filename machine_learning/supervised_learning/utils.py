import os
import json

from constants import ROOT_DIR, openface_feature_columns

def save_parameters(best_params, appendage=""):
    # Save best parameters to file

    parameter_path = os.path.join(ROOT_DIR, "data/out", "best_params_" + appendage + ".json")

    with open(parameter_path, "w") as f:
        json.dump(best_params, f)


def load_parameters(appendage=""):

    parameter_path = os.path.join(ROOT_DIR, "data/out", "best_params_" + appendage + ".json")

    # Load best parameters from file
    with open(parameter_path, "r") as f:
        best_params = json.load(f)

    return best_params


def get_features_openface(df, only_means=False):
    if only_means:
        cols = df.loc[:, df.columns.str.contains("mean")].columns
        return df[cols].values
    else:
        cols = df.loc[:, df.columns.str.contains('|'.join(openface_feature_columns))].columns
        return df[cols].values
