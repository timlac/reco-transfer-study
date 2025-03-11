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
from machine_learning.supervised_learning.utils import save_parameters, load_parameters, get_features_openface, \
    accuracy_per_emotion
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
    conf_mat = metrics.confusion_matrix(y_strings, y_pred_strings, labels=np.unique(y_strings), normalize="true")
    plot_conf_mat(conf_mat, labels=np.unique(y_strings), condition=condition)


def train_external():
    df_external_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"))
    df_external_opensmile = pd.read_csv(os.path.join(ROOT_DIR, "data/out/opensmile_data_external.csv"))

    df_merged = pd.merge(df_external_openface, df_external_opensmile, on=["filename", "emotion_id", "video_id"], how="inner")

    print(df_merged.isnull())

    features = ["audio", "video", "multimodal"]

    y_train = df_merged["emotion_id"].values

    scores = []
    results = []

    for feat in features:
        if feat == "audio":
            X_train = df_merged[opensmile_feature_columns].values
            clf, clf_proba = train_setup(X_train, y_train, feat, load_params=True)
            s, res = test_multimodal(clf, clf_proba, video=False, audio=True)
        elif feat == "video":
            X_train = get_features_openface(df_merged)
            clf, clf_proba = train_setup(X_train, y_train, feat, load_params=True)
            s, res = test_multimodal(clf, clf_proba, video=True, audio=False)
        else:
            X_train_openface = get_features_openface(df_merged)
            X_train_opensmile = df_merged[opensmile_feature_columns].values
            X_train = np.concatenate((X_train_openface, X_train_opensmile), axis=1)
            print("X shape in train:", X_train.shape)
            clf, clf_proba = train_setup(X_train, y_train, feat, load_params=True)
            s, res =test_multimodal(clf, clf_proba, video=True, audio=True)

        scores.extend(s)
        results.extend(res)

    scores_df = pd.DataFrame(scores)
    print(scores_df)

    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(ROOT_DIR, "data/out/multimodal_results.csv"), index=False)

def train_setup(X, y, feature_type, load_params=False):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if load_params:
        best_params = load_parameters(feature_type)
    else:
        gs = param_search(X, y, "accuracy", mock=False)
        save_parameters(gs.best_params_, feature_type)
        best_params = gs.best_params_

    # Evaluate the model on the training data
    # y_pred = n_fold_cross_validation(best_params, X, y)
    # get_metrics(y, y_pred, condition=feature_type)

    # train the model
    clf = SVC(**best_params)
    clf.fit(X, y)

    clf_proba = SVC(**best_params, probability=True)
    clf_proba.fit(X, y)

    return clf, clf_proba


def test_multimodal(clf, clf_proba, video=True, audio=False):
    df_openface = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))
    df_opensmile = pd.read_csv(os.path.join(ROOT_DIR, "data/out/opensmile_data.csv"))

    df_merged = pd.merge(df_openface, df_opensmile, on=["filename", "emotion_id", "video_id"], how="inner")

    conditions = ["furhat", "metahuman", "original"]

    all_scores = []

    results = []

    for condition in conditions:
        df = df_merged[df_merged["condition"] == condition]

        if video and audio:
            print("Video and audio")
            X_openface = get_features_openface(df)
            X_opensmile = df[opensmile_feature_columns].values
            X = np.concatenate((X_openface, X_opensmile), axis=1)
        elif video:
            print("Video")
            X = get_features_openface(df)
        elif audio:
            print("Audio")
            X = df[opensmile_feature_columns].values
        else:
            raise ValueError("Please select either video or audio or both.")

        X = StandardScaler().fit_transform(X)

        print("X shape in test:", X.shape)

        y = df["emotion_id"].values

        y_pred = clf.predict(X)

        df_results = pd.DataFrame({"condition": condition + "_" + audio*"audio_" + video*"video",
                                   "y": y, "y_pred": y_pred})
        results.append(df_results)

        acc = accuracy_score(y, y_pred)
        f1_score = metrics.f1_score(y, y_pred, average="macro")
        roc_auc_ovr = metrics.roc_auc_score(y, clf_proba.predict_proba(X), multi_class="ovr")

        metric_string = condition + " " + audio*"audio_" + video*"video"

        get_metrics(y, y_pred, condition=metric_string)

        scores = {"condition": condition, "accuracy": acc, "f1_score": f1_score, "roc_auc_ovr": roc_auc_ovr,
                  "video": video, "audio": audio}

        all_scores.append(scores)



    return all_scores, results


if __name__ == '__main__':
    train_external()
    # pipeline(X, y, y_strings)