import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# TODO: Check that is actually works as intended, right now it actually seems to make svm accuracy worse...

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

        rows = np.where(groups == group)[0]
        x[rows] = scaler.fit_transform(x[rows])
    return x


def get_splits(x, y):
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=10)
    return skf.split(x, y)


