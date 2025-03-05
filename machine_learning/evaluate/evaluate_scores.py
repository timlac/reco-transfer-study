from sklearn.model_selection import StratifiedKFold, cross_validate

import numpy as np


def evaluate_scores(x, y, clf, scoring_method):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    splits = skf.split(x, y)

    # get scores
    scores = cross_validate(X=x, y=y,
                            estimator=clf,
                            scoring=[scoring_method],
                            cv=splits,
                            n_jobs=-1,
                            return_train_score=True
                            )

    print('\nprinting {} measures'.format(scoring_method))
    print('avg (train):', np.mean(scores['train_{}'.format(scoring_method)]))
    print('std (train):', np.std(scores['train_{}'.format(scoring_method)]))
    print('avg (validation):', np.mean(scores['test_{}'.format(scoring_method)]))
    print('std (validation):', np.std(scores['test_{}'.format(scoring_method)]))