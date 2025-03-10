import os
import json

from constants import ROOT_DIR


parameter_path = os.path.join(ROOT_DIR, "data/out", "best_params.json")

def save_parameters(best_params):
    # Save best parameters to file
    with open(parameter_path, "w") as f:
        json.dump(best_params, f)


def load_parameters():
    # Load best parameters from file
    with open(parameter_path, "r") as f:
        best_params = json.load(f)

    return best_params