from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


def param_search(x, y, clf, scoring_method, mock=False):
    if mock:
        parameters = {
            'class_weight': ['balanced'],
        }
    else:
        parameters = {
            'class_weight': ['balanced'],

            "C": [0.1, 1, 5, 10, 25, 50, 75, 100],

            "gamma": [1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001],

            # TODO: commented out 'linear' because slow
            "kernel": ['rbf', 'poly', 'sigmoid']
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    gs = GridSearchCV(estimator=clf,
                      param_grid=parameters,
                      scoring=scoring_method,
                      cv=skf.split(x, y),
                      n_jobs=-1,
                      verbose=3
                      )

    print("running param search")
    gs.fit(x, y)
    print("finished param search")
    print(gs.best_params_)
    return gs