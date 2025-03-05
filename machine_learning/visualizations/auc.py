import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from machine_learning.utils import get_splits

# Function is a bit too complex, and way too many classes in the legend...

def plot_multi_class_auc_curve(svc, x, y):
    # Identify unique class labels
    classes = np.unique(y)
    n_classes = len(classes)

    # Binarize the labels (one-hot encoding)
    y_bin = label_binarize(y, classes=classes)

    # Mean FPR for interpolation
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 6))

    tprs = []
    aucs = []

    splits = get_splits(x, y)
    for train, test in splits:
        svc.fit(x[train], y[train])

        y_prob = svc.predict_proba(x[test])  # Probabilities for each class

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[test, i], y_prob[:, i])
            roc_auc = roc_auc_score(y_bin[test, i], y_prob[:, i])
            aucs.append(roc_auc)

            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

            plt.plot(fpr, tpr, alpha=0.3, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    # Compute macro-average ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='-', linewidth=2,
             label=r'Macro-average (AUC = %0.2f)' % mean_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    # plt.legend(loc='lower right')
    leg = plt.legend(loc='lower right', fontsize=8)
    for leg_obj in leg.legend_handles:
        leg_obj.set_picker(True)  # Make each legend item selectable
    plt.show()
