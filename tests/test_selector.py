import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from shadowselect import ImbalanceWarning, ShadowFeatureSelector, assess_imbalance
from shadowselect.simulation import make_dataset


@pytest.fixture(scope="module")
def easy():
    return make_dataset(n=1500, p=0.2, delta=1.0, n_real=3, n_noise=3, seed=0)


def rf():
    return RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=1)


def test_native_importance_recovers_support(easy):
    X, y = easy
    sel = ShadowFeatureSelector(rf(), importance="native", n_iter=10, random_state=0).fit(X, y)
    assert sel.get_support().tolist() == [True] * 3 + [False] * 3
    assert sel.transform(X).shape == (len(y), 3)


def test_permutation_importance_recovers_support(easy):
    X, y = easy
    sel = ShadowFeatureSelector(rf(), n_iter=8, random_state=0).fit(X, y)
    assert sel.get_support()[:3].all()
    assert not sel.get_support()[3:].any()


def test_default_estimator_runs(easy):
    X, y = easy
    sel = ShadowFeatureSelector(n_iter=3, random_state=0, alpha=0.5).fit(X, y)
    assert sel.importance_history_.shape == (3, 6)


def test_reproducible(easy):
    X, y = easy
    a = ShadowFeatureSelector(rf(), importance="native", n_iter=4, random_state=1).fit(X, y)
    b = ShadowFeatureSelector(rf(), importance="native", n_iter=4, random_state=1).fit(X, y)
    np.testing.assert_array_equal(a.importance_history_, b.importance_history_)


@pytest.mark.parametrize("shadow,n_shadows", [("gaussian", 10), ("permute", 8), ("permute", None)])
def test_shadow_options(easy, shadow, n_shadows):
    X, y = easy
    sel = ShadowFeatureSelector(
        rf(), importance="native", shadow=shadow, n_shadows=n_shadows, n_iter=3, random_state=0
    ).fit(X, y)
    assert sel.hits_.shape == (6,)


def test_decisions_and_keep_tentative(easy):
    X, y = easy
    sel = ShadowFeatureSelector(rf(), importance="native", n_iter=2, random_state=0).fit(X, y)
    # two rounds can never reach significance: everything tentative
    assert set(sel.decision_) == {"tentative"}
    assert not sel.get_support().any()
    sel2 = ShadowFeatureSelector(
        rf(), importance="native", n_iter=2, keep_tentative=True, random_state=0
    ).fit(X, y)
    assert sel2.get_support().all()


def test_imbalance_warning():
    X, y = make_dataset(n=2000, p=0.01, delta=1.0, n_real=2, n_noise=2, seed=0)
    with pytest.warns(ImbalanceWarning):
        ShadowFeatureSelector(rf(), importance="native", n_iter=1).fit(X, y)


def test_no_warning_when_enough_minority(easy):
    X, y = easy
    with warnings.catch_warnings():
        warnings.simplefilter("error", ImbalanceWarning)
        ShadowFeatureSelector(rf(), importance="native", n_iter=1).fit(X, y)


def test_pandas_names_and_summary(easy):
    pd = pytest.importorskip("pandas")
    X, y = easy
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    sel = ShadowFeatureSelector(rf(), importance="native", n_iter=4, random_state=0).fit(df, y)
    s = sel.summary()
    assert list(s.feature) == list(df.columns)
    kept = [c for c, k in zip(df.columns, sel.support_, strict=True) if k]
    assert list(sel.get_feature_names_out()) == kept


def test_in_pipeline(easy):
    X, y = easy
    pipe = make_pipeline(
        ShadowFeatureSelector(rf(), importance="native", n_iter=10, random_state=0),
        LogisticRegression(),
    ).fit(X, y)
    assert pipe.score(X, y) > 0.7


def test_native_without_importances_raises(easy):
    X, y = easy
    with pytest.raises(TypeError):
        ShadowFeatureSelector(LogisticRegression(), importance="native", n_iter=1).fit(X, y)


@pytest.mark.parametrize(
    "kw",
    [
        dict(shadow="bad"),
        dict(importance="bad"),
        dict(n_iter=0),
        dict(threshold_quantile=0),
        dict(alpha=1.5),
        dict(n_shadows=0),
    ],
)
def test_bad_params(easy, kw):
    X, y = easy
    with pytest.raises(ValueError):
        ShadowFeatureSelector(rf(), **kw).fit(X, y)


def test_single_class_raises():
    with pytest.raises(ValueError):
        ShadowFeatureSelector(rf()).fit(np.zeros((10, 2)), np.zeros(10))


def test_xgboost_native():
    xgb = pytest.importorskip("xgboost")
    X, y = make_dataset(n=1500, p=0.2, delta=1.0, n_real=2, n_noise=2, seed=0)
    est = xgb.XGBClassifier(n_estimators=50, max_depth=3, verbosity=0)
    sel = ShadowFeatureSelector(est, importance="native", n_iter=10, random_state=0).fit(X, y)
    assert sel.get_support().tolist() == [True, True, False, False]


def test_assess_imbalance():
    y = np.r_[np.ones(20), np.zeros(9980)]
    r = assess_imbalance(y, n_rep=100)
    assert r.n_minority == 20
    assert "Unreliable" in r.recommendation
    big = assess_imbalance(np.r_[np.ones(2000), np.zeros(8000)], n_rep=100)
    assert big.min_detectable_delta < r.min_detectable_delta


def test_too_few_iterations_warns(easy):
    X, y = easy
    with pytest.warns(UserWarning, match="too small"):
        ShadowFeatureSelector(rf(), importance="native", n_iter=3).fit(X, y)
