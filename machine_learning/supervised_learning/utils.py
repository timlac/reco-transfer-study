import os
import json
import numpy as np

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


def accuracy_per_emotion(y, y_pred):
    unique_emotions = np.unique(np.concatenate((y, y_pred)))  # Get all unique emotions
    accuracy_dict = {}

    for emotion in unique_emotions:
        mask = (y == emotion)  # Find occurrences of this emotion in true labels
        accuracy = np.mean(y_pred[mask] == emotion)  # Compute accuracy for this emotion
        accuracy_dict[emotion.item()] = accuracy.item()

    return accuracy_dict


def compare_conditions(accuracy_results, baseline="original_video_"):
    baseline_acc = accuracy_results[baseline]  # Reference accuracy

    performance_drop = {}

    for condition, acc_dict in accuracy_results.items():
        if condition == baseline:
            continue  # Skip baseline

        performance_drop[condition] = {
            emotion: acc_dict.get(emotion, 0) - baseline_acc.get(emotion, 0)
            for emotion in baseline_acc
        }

    return performance_drop


def main():
    y = np.array([1, 2, 1, 3, 2, 3, 3, 1, 2, 1])  # True labels
    y_pred = np.array([1, 2, 3, 3, 1, 3, 2, 1, 2, 3])  # Predicted labels

    result = accuracy_per_emotion(y, y_pred)
    print(result)  # Example output: {1: 0.75, 2: 0.666, 3: 0.5}

if __name__ == "__main__":
    main()

