from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from shadowselect import ShadowFeatureSelector


@parametrize_with_checks(
    [
        ShadowFeatureSelector(
            RandomForestClassifier(n_estimators=5, random_state=0),
            importance="native",
            n_iter=2,
            keep_tentative=True,
            min_minority=0,
            random_state=0,
        )
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)
