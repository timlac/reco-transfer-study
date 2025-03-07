import pandas as pd
from sklearn.metrics import accuracy_score
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
from machine_learning.utils import get_splits, functional_scale_by


def train(X, y):
    clf = SVC()
    gs = param_search(X, y, clf, "accuracy")
    best_params = gs.best_params_
    svc = SVC(**best_params)
    svc_proba = SVC(**best_params, probability=True)
    return svc, svc_proba, best_params

def n_fold_cross_validation(svc, svc_proba, X, y, y_strings):
    # evaluate classifier with different scoring methods
    val_mean_acc, val_std_acc  = evaluate_scores(X, y, svc, "accuracy")

    evaluate_scores(X, y, svc_proba, "roc_auc_ovr")
    evaluate_scores(X, y, svc, "f1_macro")

    # get the predictions using cross validation
    splits = get_splits(X, y)
    y_pred = cross_val_predict(svc, X, y, cv=splits, n_jobs=-1)

    y_pred_strings = Mapper.get_emotion_from_id(y_pred)

    # get classification report based on predictions
    report = metrics.classification_report(y_true=y_strings, y_pred=y_pred_strings)
    print(report)

    # plot confusion matrix
    conf_mat = metrics.confusion_matrix(y_strings, y_pred_strings, labels=np.unique(y_strings))
    plot_conf_mat(conf_mat, labels=np.unique(y_strings))
    print(conf_mat)

def other_metrics(y, y_pred, y_strings):
    # get classification report based on predictions
    report = metrics.classification_report(y_true=y_strings, y_pred=y_pred_strings)
    print(report)

    # plot confusion matrix
    conf_mat = metrics.confusion_matrix(y_strings, y_pred_strings, labels=np.unique(y_strings))
    plot_conf_mat(conf_mat, labels=np.unique(y_strings))





def evaluate(svc, svc_proba, X, y, y_strings):
    y_pred = svc.predict(X)

    acc = accuracy_score(y, y_pred)
    f1_score = metrics.f1_score(y, y_pred, average="macro")
    roc_auc_ovr = metrics.roc_auc_score(y, svc_proba.predict_proba(X), multi_class="ovr")



    return val_mean_acc, val_std_acc



def pipeline(X, y , y_strings, video_ids):

    print(np.unique(y))

    # X = functional_scale_by(X, video_ids, "standard")
    X = StandardScaler().fit_transform(X)

    print(X.shape)  # Number of rows and columns in the scaled feature matrix

    clf = SVC()

    # do grid search for best parameters
    # gs = param_search(X, y, clf, "accuracy")
    #
    # best_params = gs.best_params_
    best_params = {'C': 10, 'class_weight': 'balanced', 'gamma': 0.1, 'kernel': 'rbf'}

    svc = SVC(**best_params)

    # evaluate classifier with different scoring methods
    val_mean_acc, val_std_acc  = evaluate_scores(X, y, svc, "accuracy")

    svc_proba = SVC(**best_params, probability=True)
    # plot_multi_class_auc_curve(svc_proba, X, y)

    evaluate_scores(X, y, svc_proba, "roc_auc_ovr")
    evaluate_scores(X, y, svc, "f1_macro")

    # get the predictions using cross validation
    splits = get_splits(X, y)
    y_pred = cross_val_predict(svc, X, y, cv=splits, n_jobs=-1)

    y_pred_strings = Mapper.get_emotion_from_id(y_pred)

    # get classification report based on predictions
    report = metrics.classification_report(y_true=y_strings, y_pred=y_pred_strings)
    print(report)

    # plot confusion matrix
    conf_mat = metrics.confusion_matrix(y_strings, y_pred_strings, labels=np.unique(y_strings))
    plot_conf_mat(conf_mat, labels=np.unique(y_strings))
    print(conf_mat)

    return val_mean_acc, val_std_acc


def main():
    acc_scores = []

    df = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"))

    print(df.isnull())
    nan_rows = df[df.isnull().T.any()]
    print(nan_rows)
    print(df.shape)
    cols = df.loc[:, df.columns.str.contains('|'.join(openface_feature_columns))].columns

    X = df.loc[:, df.columns.str.contains('mean')].values

    y = df["emotion_id"].values

    y_strings = df["emotion"].values

    video_ids = df["video_id"].values

    print(X.shape)  # Number of rows and columns in the feature matrix
    print(y.shape)  # Number of rows in the target vector
    print(y_strings.shape)

    val_mean_acc, val_std_acc = pipeline(X, y, y_strings, video_ids)

    #
    #
    #
    #
    # conditions = ["furhat", "metahuman", "original"]
    #
    # for c in conditions:
    #     print("\nRunning for condition:", c)
    #
    #     df = df_openface[df_openface["condition"] == c]
    #
    #     cols = df.loc[:, df.columns.str.contains('|'.join(openface_feature_columns))].columns
    #     print(cols)
    #
    #     # X = df.loc[:, df.columns.str.contains('mean')].values
    #
    #     X = df.loc[:, df.columns.str.contains('|'.join(openface_feature_columns))].values
    #     y = df["emotion_id"].values
    #     y_strings = df["emotion"].values
    #
    #     print(X.shape)  # Number of rows and columns in the feature matrix
    #     print(y.shape)  # Number of rows in the target vector
    #     print(y_strings.shape)
    #
    #     val_mean_acc, val_std_acc = pipeline(X, y, y_strings, c)
    #
    #     scores = {"condition": c, "val_mean_acc": val_mean_acc, "val_std_acc": val_std_acc}
    #     acc_scores.append(scores)
    #
    # c = "audio"
    # df_opensmile = pd.read_csv(os.path.join(ROOT_DIR, "data/out/opensmile_data.csv"))
    # print(df_opensmile.shape)
    #
    # print("\nRunning for opensmile data")
    #
    # X = df_opensmile[opensmile_feature_columns].values
    # y = df_opensmile["emotion_id"].values
    # y_strings = df_opensmile["emotion"].values
    #
    # print(X.shape)  # Number of rows and columns in the feature matrix
    # print(y.shape)  # Number of rows in the target vector
    # print(y_strings.shape)
    #
    # val_mean_acc, val_std_acc = pipeline(X, y, y_strings, c)
    # scores = {"condition": c, "val_mean_acc": val_mean_acc, "val_std_acc": val_std_acc}
    #
    # acc_scores.append(scores)
    #
    # df_scores = pd.DataFrame(acc_scores)
    # print(df_scores)


if __name__ == '__main__':
    main()
    # pipeline(X, y, y_strings)