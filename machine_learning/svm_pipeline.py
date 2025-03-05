import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper

from constants import feature_columns
from machine_learning.evaluate.evaluate_scores import evaluate_scores
from machine_learning.parameter_search.svm_ps import param_search
from machine_learning.utils import functional_scale_by, get_splits, plot_conf_mat
from machine_learning.visualizations.auc import plot_multi_class_auc_curve


def pipeline():



def main():
    df_full = pd.read_csv("../data/out/openface_data.csv")
    print(df_full.shape)

    conditions = ["furhat", "metahuman", "original"]

    for c in conditions:
        df = df_full[df_full["condition"] == c]

        X = df.loc[:, df.columns.str.contains('|'.join(feature_columns))].values
        y = df["emotion_id"].values
        y_strings = df["emotion"].values

        print(X.shape)  # Number of rows and columns in the feature matrix
        print(y.shape)  # Number of rows in the target vector



# X = functional_scale_by(X, video_ids, "standard")
X = StandardScaler().fit_transform(X)

print(X.shape)  # Number of rows and columns in the scaled feature matrix

clf = SVC()

# do grid search for best parameters
gs = param_search(X, y, clf, "accuracy")

svc = SVC(**gs.best_params_)

# evaluate classifier with different scoring methods
evaluate_scores(X, y, svc, "accuracy")

svc_proba = SVC(**gs.best_params_, probability=True)
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