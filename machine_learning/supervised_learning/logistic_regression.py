import pandas as pd
import os

from sklearn.linear_model import LogisticRegression, SGDClassifier

from constants import ROOT_DIR
from machine_learning.supervised_learning.evaluate.evaluate_scores import evaluate_scores
from machine_learning.supervised_learning.utils import get_features_openface
from machine_learning.supervised_learning.parameter_search.log_reg_ps import param_search_logistic

df_external = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data_external.csv"))
df_conditions = pd.read_csv(os.path.join(ROOT_DIR, "data/out/openface_data.csv"))
conditions = ["furhat", "metahuman", "original"]

X_train = get_features_openface(df_external)
y_train = df_external["emotion_id"].values

sgd_logistic = SGDClassifier(
    loss="log_loss",  # Equivalent to logistic regression
    penalty="l1",  # Supports L1 sparsity
    alpha=0.0001,  # Equivalent to 1/C in LogisticRegression
    max_iter=1000,  # Fewer iterations than saga
    tol=1e-3,  # Stop early if loss doesn't improve
    n_jobs=-1  # Use all available CPU cores
)

evaluate_scores(X_train, y_train, sgd_logistic, "accuracy")
evaluate_scores(X_train, y_train, sgd_logistic, "roc_auc_ovr")


# # param_search_logistic(X_train, y_train, "accuracy")
#
# lasso_logistic = LogisticRegression(
#     penalty='l2',
#     solver="lbfgs",
#     C=0.1,
#     max_iter=5000,
# )
# #
# #


