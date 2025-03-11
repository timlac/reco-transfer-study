import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper

from constants import ROOT_DIR, opensmile_feature_columns
from machine_learning.supervised_learning.evaluate.evaluate_scores import evaluate_scores
from machine_learning.supervised_learning.parameter_search.svm_ps import param_search
from machine_learning.supervised_learning.utils import save_parameters, load_parameters, get_features_openface
from machine_learning.supervised_learning.visualizations.misc_plots import plot_conf_mat
from machine_learning.utils import get_splits


def n_fold_cross_validation(best_params, X, y):
    svc = SVC(**best_params)
    svc_proba = SVC(**best_params, probability=True)

    # evaluate classifier with different scoring methods
    evaluate_scores(X, y, svc, "accuracy")
    evaluate_scores(X, y, svc_proba, "roc_auc_ovr")
    evaluate_scores(X, y, svc, "f1_macro")

    # get the predictions using cross validation
    splits = get_splits(X, y)
    y_pred = cross_val_predict(svc, X, y, cv=splits, n_jobs=-1)

    return y_pred


def get_metrics(y, y_pred, condition=None):
    # get classification report based on predictions
    report = metrics.classification_report(y_true=y, y_pred=y_pred)
    print(report)

    y_strings = Mapper.get_emotion_from_id(y)
    y_pred_strings = Mapper.get_emotion_from_id(y_pred)

    # plot confusion matrix
    conf_mat = metrics.confusion_matrix(y_strings, y_pred_strings, labels=np.unique(y_strings))
    plot_conf_mat(conf_mat, labels=np.unique(y_strings), condition=condition)


def train_external():
    df_external_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"))
    df_external_opensmile = pd.read_csv(os.path.join(ROOT_DIR, "data/out/opensmile_data_external.csv"))

    X_train_openface = get_features_openface(df_external_openface)
    X_train_opensmile = df_external_opensmile[opensmile_feature_columns].values


def train_setup(X, y, feature_type, load_params=False):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if load_params:
        best_params = load_parameters()
    else:
        gs = param_search(X, y, "accuracy")
        save_parameters(gs.best_params_)
        best_params = gs.best_params_

    # Evaluate the model on the training data
    y_pred = n_fold_cross_validation(best_params, X_train, y_train)
    get_metrics(y, y_pred, condition="external")

    # train the model
    clf = SVC(**best_params)
    clf.fit(X, y)

    clf_proba = SVC(**best_params, probability=True)
    clf_proba.fit(X, y)









def main():

    # TODO: Do the corresponding implementation for audio and audio + video for each condition (furhat, metahuman, original)

    df_external = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"))
    df_conditions = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))
    conditions = ["furhat", "metahuman", "original"]

    load_params = False

    print(df_external.isnull())
    nan_rows = df_external[df_external.isnull().T.any()]
    print(nan_rows)
    print(df_external.shape)

    X_train = get_features_openface(df_external)

    y_train = df_external["emotion_id"].values

    video_ids = df_external["video_id"].values

    print(X_train.shape)  # Number of rows and columns in the feature matrix
    print(y_train.shape)  # Number of rows in the target vector

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if load_params:
        best_params = load_parameters()
    else:
        gs = param_search(X_train, y_train, "accuracy")
        save_parameters(gs.best_params_)
        best_params = gs.best_params_


    # Evaluate the model on the training data
    y_pred = n_fold_cross_validation(best_params, X_train, y_train)
    get_metrics(y_train, y_pred, condition="external")

    # train the model
    clf = SVC(**best_params)
    clf.fit(X_train, y_train)

    clf_proba = SVC(**best_params, probability=True)
    clf_proba.fit(X_train, y_train)

    # Evaluate model on test data in different conditions
    all_scores = []
    for c in conditions:
        print("\nEvaluating for condition:", c)

        df = df_conditions[df_conditions["condition"] == c]

        X_test = get_features_openface(df)

        y_test = df["emotion_id"].values

        print(X_test.shape)  # Number of rows and columns in the feature matrix
        print(y_test.shape)  # Number of rows in the target vector

        X_test = StandardScaler().fit_transform(X_test)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred, average="macro")
        roc_auc_ovr = metrics.roc_auc_score(y_test, clf_proba.predict_proba(X_test), multi_class="ovr")

        get_metrics(y_test, y_pred, condition=c)

        scores = {"condition": c, "accuracy": acc, "f1_score": f1_score, "roc_auc_ovr": roc_auc_ovr}

        all_scores.append(scores)

    df_scores = pd.DataFrame(all_scores)
    print(df_scores)



if __name__ == '__main__':
    main()
    # pipeline(X, y, y_strings)