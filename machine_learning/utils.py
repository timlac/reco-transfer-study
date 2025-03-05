import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def functional_scale_by(x, groups, method):
    """
    :param x: a matrix with shape (observations, features)
    :param groups: np Array
    :param method: cls Method
    :return: scaled x matrix
    """
    for group in np.unique(groups):
        if method == "min_max":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        else:
            raise RuntimeError("Something went wrong, no scaling method chosen")

        rows = np.where(groups == group)
        x[rows] = scaler.fit_transform(x[rows])
    return x


def get_splits(x, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    return skf.split(x, y)


def plot_conf_mat(cm, labels=None):
    print("labels", labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')  # 'd' for integer values

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    plt.subplots_adjust(bottom=.25, left=.25)

    plt.tight_layout()
    plt.show()