import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper

from constants import openface_feature_columns, opensmile_feature_columns, ROOT_DIR
from machine_learning.supervised_learning.evaluate.evaluate_scores import evaluate_scores
from machine_learning.supervised_learning.parameter_search.svm_ps import param_search
from machine_learning.supervised_learning.visualizations.misc_plots import plot_conf_mat
from machine_learning.utils import get_splits

# TODO: Metahuman has basically chance level accuracy, need to investigate why


def pipeline(X, y , y_strings, condition):

    print(np.unique(y))

    # X = functional_scale_by(X, video_ids, "standard")
    X = StandardScaler().fit_transform(X)

    print(X.shape)  # Number of rows and columns in the scaled feature matrix

    # do grid search for best parameters
    gs = param_search(X, y, "accuracy")

    svc = SVC(**gs.best_params_)

    # evaluate classifier with different scoring methods
    mean_acc, std_acc  = evaluate_scores(X, y, svc, "accuracy")

    svc_proba = SVC(**gs.best_params_, probability=True)
    # plot_multi_class_auc_curve(svc_proba, X, y)

    mean_auc, std_auc = evaluate_scores(X, y, svc_proba, "roc_auc_ovr")
    mean_f1, std_f1 = evaluate_scores(X, y, svc, "f1_macro")

    # get the predictions using cross validation
    splits = get_splits(X, y)
    y_pred = cross_val_predict(svc, X, y, cv=splits, n_jobs=-1)

    y_pred_strings = Mapper.get_emotion_from_id(y_pred)

    # get classification report based on predictions
    report = metrics.classification_report(y_true=y_strings, y_pred=y_pred_strings)
    print(report)

    # plot confusion matrix
    conf_mat = metrics.confusion_matrix(y_strings, y_pred_strings, labels=np.unique(y_strings))
    plot_conf_mat(conf_mat, labels=np.unique(y_strings), condition=condition)
    print(conf_mat)


    scores = {
        "condition": condition,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_f1": mean_f1,
        "std_f1": std_f1
    }

    return scores


def main():
    # TODO: try training everything together and evaluate separately

    all_scores = []

    df_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))
    print(df_openface.shape)

    conditions = ["furhat", "metahuman", "original"]

    for c in conditions:
        print("\nRunning for condition:", c)

        df = df_openface[df_openface["condition"] == c]

        cols = df.loc[:, df.columns.str.contains('|'.join(openface_feature_columns))].columns
        print(cols)

        # X = df.loc[:, df.columns.str.contains('mean')].values

        X = df.loc[:, df.columns.str.contains('|'.join(openface_feature_columns))].values
        y = df["emotion_id"].values
        y_strings = df["emotion"].values

        print(X.shape)  # Number of rows and columns in the feature matrix
        print(y.shape)  # Number of rows in the target vector
        print(y_strings.shape)

        scores = pipeline(X, y, y_strings, c)
        all_scores.append(scores)

    c = "audio"
    df_opensmile = pd.read_csv(os.path.join(ROOT_DIR, "data/out/opensmile_data.csv"))
    print(df_opensmile.shape)

    print("\nRunning for opensmile data")

    X = df_opensmile[opensmile_feature_columns].values
    y = df_opensmile["emotion_id"].values
    y_strings = df_opensmile["emotion"].values

    print(X.shape)  # Number of rows and columns in the feature matrix
    print(y.shape)  # Number of rows in the target vector
    print(y_strings.shape)

    scores = pipeline(X, y, y_strings, c)
    all_scores.append(scores)

    df_scores = pd.DataFrame(all_scores)
    print(df_scores)


if __name__ == '__main__':
    main()
    # pipeline(X, y, y_strings)