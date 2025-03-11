from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

def param_search_logistic(x, y, scoring_method, mock=False):
    """
    Perform hyperparameter search for Logistic Regression.

    Parameters:
        x (array-like): Feature set.
        y (array-like): Target labels.
        scoring_method (str): Scoring metric.
        mock (bool): If True, runs a minimal search.

    Returns:
        GridSearchCV object with best parameters.
    """
    clf = LogisticRegression(solver="saga", max_iter=500)

    if mock:
        parameters = {
            'class_weight': ['balanced'],
        }
    else:
        parameters = {
            'class_weight': ['balanced', None],
            'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l1', 'l2', 'elasticnet'],  # Different regularization types
            'l1_ratio': [0.1, 0.5, 0.9] if 'elasticnet' in ['l1', 'l2', 'elasticnet'] else [None]  # Only for elasticnet
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    gs = GridSearchCV(
        estimator=clf,
        param_grid=parameters,
        scoring=scoring_method,
        cv=skf.split(x, y),
        n_jobs=-1,
        verbose=3
    )

    print("Running param search for Logistic Regression")
    gs.fit(x, y)
    print("Finished param search")
    print(gs.best_params_)

    return gs
