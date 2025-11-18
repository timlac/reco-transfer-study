import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from sklearn.inspection import permutation_importance

from constants import ROOT_DIR, opensmile_feature_columns, var_name_to_feat


from machine_learning.supervised_learning.utils import load_parameters, get_features_openface
from machine_learning.utils import get_splits

feature_type = "video"

df_external = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"))
df_internal = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))

cols = df_external.loc[:, df_external.columns.str.contains("mean")].columns

X_train = get_features_openface(df_external, only_means=True)
y_train = df_external["emotion_id"].values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

best_params = load_parameters(feature_type)
svc = SVC(**best_params)
svc.fit(X_train, y_train)


importances = None

conditions = ["furhat", "metahuman", "original"]

for condition in conditions:
    df = df_internal[df_internal["condition"] == condition].copy()
    X_test = get_features_openface(df, only_means=True)
    y_test = df["emotion_id"].values

    X_test = StandardScaler().fit_transform(X_test)

    result = permutation_importance(svc, X_test, y_test, n_repeats=20, n_jobs=-1, scoring="accuracy", random_state=42)

    # print("Condition:", condition)
    # print("Mean importances:", np.mean(result['importances'], axis=1))
    # print("Std importances:", np.std(result['importances'], axis=1))


    d = {"importances": result['importances'],
        "importances_mean": np.mean(result['importances'], axis=1),
        "importances_std": np.std(result['importances'], axis=1)}

    importances = result['importances']

    print(d)

    perm_sorted_idx = d['importances_mean'].argsort()

    fig, ax = plt.subplots(figsize=(8, 24))

    ax.boxplot(
        importances[perm_sorted_idx].T,
        vert=False,
        labels=cols[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")

    ax.set_title(
        f"Permutation Importances ({condition}) on all features\n(with respect to validation set using 5 fold cross validation)")
    ax.set_xlabel("Decrease in ROC AUC score")
    ax.figure.tight_layout()
    plt.show()


    top_n = 10

    fig, ax = plt.subplots(figsize=(8, 4))

    # Get the top N features and their importances
    top_features = cols[perm_sorted_idx][-top_n:]
    top_importances = importances[perm_sorted_idx].T[:, -top_n:]

    top_features_pretty = []
    for i in top_features:
        top_features_pretty.append(var_name_to_feat[i])

    # Create the boxplot for the top features
    ax.boxplot(top_importances, vert=False, labels=top_features_pretty)
    ax.axvline(x=0, color="k", linestyle="--")

    ax.set_title(f"Permutation Importances ({condition}) on Top 10 features")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.show()