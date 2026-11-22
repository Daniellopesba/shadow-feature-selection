from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from shadowselect import ShadowFeatureSelector, UnivariateScreen


@parametrize_with_checks(
    [
        ShadowFeatureSelector(
            RandomForestClassifier(n_estimators=5, random_state=0),
            importance="native",
            n_iter=12,
            n_shadows=3,
            keep_tentative=True,
            min_minority=0,
            random_state=0,
        ),
        UnivariateScreen(),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)
