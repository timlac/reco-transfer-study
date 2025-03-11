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


def main():
    groups = np.array(['A205', 'A323', 'A207', 'A67', 'A426', 'A407', 'A303', 'A227', 'A200', 'A417', 'A207', 'A207', 'A424', 'A417',
     'A437', 'A205', 'A334', 'A426', 'A424', 'A102', 'A332', 'A323', 'A102', 'A334', 'A425', 'A435', 'A424', 'A337',
     'A72', 'A426', 'A417', 'A417', 'A332', 'A426', 'A200', 'A207', 'A67', 'A426', 'A67', 'A220', 'A438', 'A227',
     'A221', 'A220', 'A67', 'A91', 'A405', 'A303', 'A417', 'A72', 'A220', 'A426', 'A417', 'A435', 'A67', 'A424', 'A205',
     'A327', 'A438', 'A426', 'A417', 'A102', 'A435', 'A67'])

    print(np.where(groups == 'A205')[0])

    for group in np.unique(groups):
        print(f"\ngroup:  {group}")

        rows = np.where(groups == group)[0]

        print(rows)

if __name__ == "__main__":
    main()