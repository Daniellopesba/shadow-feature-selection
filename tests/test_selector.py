import warnings

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from shadowselect import (
    ImbalanceWarning,
    LearnerWarning,
    ShadowFeatureSelector,
    assess_imbalance,
    null_keep_rate,
)
from shadowselect.simulation import make_dataset

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def easy():
    return make_dataset(n=1500, p=0.2, delta=1.0, n_real=3, n_noise=3, seed=0)


def rf(**kw):
    return RandomForestClassifier(n_estimators=30, max_depth=5, n_jobs=1, **kw)


class RandomImportance(ClassifierMixin, BaseEstimator):
    """Learner whose importances are pure noise: shadows look like real features."""

    calls: list = []

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = rng.random(X.shape[1])
        self.classes_ = np.unique(y)
        RandomImportance.calls.append(
            (X.shape, None if sample_weight is None else len(sample_weight))
        )
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.classes_[0])

    def predict_proba(self, X):
        p = np.zeros((X.shape[0], len(self.classes_)))
        p[:, 0] = 1
        return p


# ----------------------------------------------------------------- behaviour
def test_native_importance_recovers_support(easy):
    X, y = easy
    sel = ShadowFeatureSelector(
        rf(), importance="native", n_iter=10, keep_tentative=False, random_state=0
    ).fit(X, y)
    assert sel.get_support().tolist() == [True] * 3 + [False] * 3
    assert sel.transform(X).shape == (len(y), 3)


def test_permutation_importance_recovers_support(easy):
    X, y = easy
    sel = ShadowFeatureSelector(
        rf(), importance="permutation", n_iter=10, keep_tentative=False, random_state=0
    ).fit(X, y)
    assert sel.get_support()[:3].all()
    assert not sel.get_support()[3:].any()


def test_default_estimator(easy):
    X, y = easy
    sel = ShadowFeatureSelector(random_state=0).fit(X, y)
    assert sel.importance_mode_ == "native"
    assert sel.n_shadows_ == 10
    assert sel.get_support().tolist() == [True] * 3 + [False] * 3


def test_auto_importance_falls_back_to_permutation(easy):
    X, y = easy
    sel = ShadowFeatureSelector(LogisticRegression(), n_iter=10, random_state=0).fit(X, y)
    assert sel.importance_mode_ == "permutation"


def test_defaults_match_paper_guidance():
    s = ShadowFeatureSelector()
    assert s.keep_tentative is True
    assert s.n_shadows == 10
    assert s.threshold_quantile == 1.0


def test_reproducible_and_independent_of_n_jobs(easy):
    X, y = easy
    kw = dict(importance="native", n_iter=8, random_state=1)
    a = ShadowFeatureSelector(rf(), **kw).fit(X, y)
    b = ShadowFeatureSelector(rf(), **kw).fit(X, y)
    c = ShadowFeatureSelector(rf(), n_jobs=3, **kw).fit(X, y)
    np.testing.assert_array_equal(a.importance_history_, b.importance_history_)
    np.testing.assert_array_equal(a.importance_history_, c.importance_history_)


def test_pipeline_estimator_is_seeded(easy):
    X, y = easy
    est = make_pipeline(rf())  # random_state lives at "randomforestclassifier__random_state"
    a = ShadowFeatureSelector(est, importance="permutation", n_iter=8, n_repeats=1, random_state=3)
    b = ShadowFeatureSelector(est, importance="permutation", n_iter=8, n_repeats=1, random_state=3)
    np.testing.assert_array_equal(a.fit(X, y).importance_history_, b.fit(X, y).importance_history_)


@pytest.mark.parametrize(
    "shadow,n_shadows,expected", [("gaussian", 10, 10), ("permute", 4, 4), ("permute", 15, 15)]
)
def test_shadow_options(easy, shadow, n_shadows, expected):
    X, y = easy
    sel = ShadowFeatureSelector(
        rf(), importance="native", shadow=shadow, n_shadows=n_shadows, n_iter=8, random_state=0
    ).fit(X, y)
    assert sel.hits_.shape == (6,)
    assert sel.shadow_importance_history_.shape == (8, expected)


def test_one_shadow_per_feature(easy):
    X, y = easy
    sel = ShadowFeatureSelector(rf(), n_shadows=None, n_iter=8, random_state=0).fit(X, y)
    assert sel.n_shadows_ == 6


def test_keep_tentative(easy):
    X, y = easy
    sel = ShadowFeatureSelector(RandomImportance(), importance="native", n_iter=8, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LearnerWarning)
        sel.fit(X, y)
    tent = sel.decision_ == "tentative"
    np.testing.assert_array_equal(sel.support_, sel.decision_ != "rejected")
    sel.set_params(keep_tentative=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LearnerWarning)
        sel.fit(X, y)
    assert not sel.support_[tent].any()


def test_too_few_rounds_is_an_error(easy):
    X, y = easy
    with pytest.raises(ValueError, match=r"n_iter >= 7"):
        ShadowFeatureSelector(rf(), n_iter=3).fit(X, y)


def test_bh_rejects_more_noise_than_bonferroni():
    X, y = make_dataset(n=3000, p=0.1, delta=1.0, n_real=2, n_noise=38, seed=0)
    kw = dict(importance="native", n_iter=12, random_state=0)
    bonf = ShadowFeatureSelector(rf(), **kw).fit(X, y)
    bh = ShadowFeatureSelector(rf(), correction="bh", **kw).fit(X, y)
    assert (bh.decision_ == "rejected").sum() >= (bonf.decision_ == "rejected").sum()
    assert bh.support_[:2].all()


def test_expected_null_keep_rate(easy):
    X, y = easy
    sel = ShadowFeatureSelector(rf(), n_iter=10, random_state=0).fit(X, y)
    assert sel.expected_null_keep_rate_ == pytest.approx(null_keep_rate(6, 10, 1 / 11))


def test_null_keep_rate_grows_with_features_and_falls_with_rounds():
    assert null_keep_rate(200, 20, 1 / 11) > null_keep_rate(10, 20, 1 / 11)
    assert null_keep_rate(200, 40, 1 / 11) < null_keep_rate(200, 20, 1 / 11)
    assert null_keep_rate(10, 20, 1 / 11, keep_tentative=False) < 1e-6


# ------------------------------------------------------------------ warnings
def test_imbalance_warning():
    X, y = make_dataset(n=2000, p=0.01, delta=1.0, n_real=2, n_noise=2, seed=0)
    with pytest.warns(ImbalanceWarning, match="Below about 25"):
        ShadowFeatureSelector(rf(), importance="native", n_iter=8).fit(X, y)
    X, y = make_dataset(n=2000, p=0.03, delta=1.0, n_real=2, n_noise=2, seed=0)
    with pytest.warns(ImbalanceWarning, match="marginal band"):
        ShadowFeatureSelector(rf(), importance="native", n_iter=8).fit(X, y)


def test_no_warning_when_enough_minority(easy):
    X, y = easy
    with warnings.catch_warnings():
        warnings.simplefilter("error", ImbalanceWarning)
        ShadowFeatureSelector(rf(), importance="native", n_iter=8).fit(X, y)


def test_learner_warning_when_shadows_look_like_real_features(easy):
    X, y = easy
    with pytest.warns(LearnerWarning, match="memorising"):
        ShadowFeatureSelector(RandomImportance(), importance="native", n_iter=10).fit(X, y)


# ---------------------------------------------------------------- inputs
def test_pandas_names_and_summary(easy):
    X, y = easy
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    sel = ShadowFeatureSelector(rf(), importance="native", n_iter=8, random_state=0).fit(df, y)
    s = sel.summary()
    assert list(s.feature) == list(df.columns)
    assert {"hit_rate", "median_shadow_threshold", "delta_hat", "s_hat", "underpowered"} <= set(
        s.columns
    )
    n_min = sel.n_minority_
    np.testing.assert_allclose(s.s_hat, n_min * s.delta_hat**2)
    assert (s.s_hat[:3] > 25).all() and not s.underpowered[:3].any()
    kept = [c for c, k in zip(df.columns, sel.support_, strict=True) if k]
    assert list(sel.get_feature_names_out()) == kept


def test_refit_on_array_forgets_column_names(easy):
    X, y = easy
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    sel = ShadowFeatureSelector(rf(), n_iter=8, random_state=0).fit(df, y)
    assert hasattr(sel, "feature_names_in_")
    sel.fit(X, y)
    assert not hasattr(sel, "feature_names_in_")
    assert sel.get_feature_names_out()[0] == "x0"


def test_missing_values(easy):
    X, y = easy
    X = X.copy()
    X[::5, 0] = np.nan
    sel = ShadowFeatureSelector(rf(), n_iter=8, random_state=0).fit(X, y)
    assert sel.support_[0]
    assert np.isfinite(sel.effect_size_[0])


def test_float32_is_kept(easy):
    X, y = easy
    from shadowselect._validation import prepare_X

    assert prepare_X(X.astype(np.float32))[0].dtype == np.float32
    ShadowFeatureSelector(rf(), n_iter=8, random_state=0).fit(X.astype(np.float32), y)


def test_bool_and_nullable_columns(easy):
    X, y = easy
    df = pd.DataFrame(X[:, :3], columns=list("abc"))
    df["flag"] = X[:, 3] > 0
    df["count"] = pd.array(np.where(np.arange(len(y)) % 4 == 0, None, (X[:, 4] > 0)), "Int64")
    sel = ShadowFeatureSelector(rf(), n_iter=8, random_state=0).fit(df, y)
    assert list(sel.get_feature_names_out()[:3]) == list("abc")


def test_sparse_input(easy):
    X, y = easy
    sel = ShadowFeatureSelector(rf(), n_iter=8, random_state=0).fit(sparse.csr_matrix(X), y)
    assert sel.support_[:3].all()
    assert sel.transform(sparse.csr_matrix(X)).shape == (len(y), sel.support_.sum())


def test_categorical_dataframe_with_xgboost(easy):
    xgb = pytest.importorskip("xgboost")
    X, y = easy
    df = pd.DataFrame(X[:, :4].copy(), columns=list("abcd"))
    rng = np.random.default_rng(0)
    neg = rng.choice(["u", "v", "w"], len(y), p=[0.15, 0.15, 0.7])
    df["cat_signal"] = pd.Categorical(np.where(y == 1, rng.choice(["u", "v"], len(y)), neg))
    df["cat_noise"] = pd.Categorical(rng.choice(list("pqrs"), len(y)))
    df.loc[::7, "a"] = np.nan
    est = xgb.XGBClassifier(n_estimators=30, max_depth=3, enable_categorical=True, verbosity=0)
    sel = ShadowFeatureSelector(est, n_iter=10, random_state=0).fit(df, y)
    out = set(sel.get_feature_names_out())
    assert {"a", "b", "c", "cat_signal"} <= out
    assert "cat_noise" not in out or sel.decision_[-1] == "tentative"
    assert np.isnan(sel.effect_size_[-1])


def test_categorical_dataframe_with_lightgbm(easy):
    lgb = pytest.importorskip("lightgbm")
    X, y = easy
    df = pd.DataFrame(X, columns=list("abcdef"))
    neg = np.random.default_rng(0).choice(["u", "w"], len(y), p=[0.3, 0.7])
    df["cat"] = pd.Categorical(np.where(y == 1, "u", neg))
    est = lgb.LGBMClassifier(n_estimators=30, num_leaves=7, verbose=-1)
    sel = ShadowFeatureSelector(est, n_iter=10, random_state=0).fit(df, y)
    assert sel.support_[[0, 1, 2, 6]].all()
    assert any("mean gain" in a for a in sel.estimator_adjustments_)
    out = sel.transform(df)
    assert out["cat"].dtype == "category" and list(out.columns) == list(sel.get_feature_names_out())
    assert est.get_params()["importance_type"] == "split"  # user's object untouched


def test_xgboost_2_gets_prevalence_intercept(monkeypatch, easy):
    xgb = pytest.importorskip("xgboost")
    X, y = easy
    monkeypatch.setattr(xgb, "__version__", "2.1.3")
    sel = ShadowFeatureSelector(xgb.XGBClassifier(n_estimators=5, verbosity=0), n_iter=8)
    base, notes = sel._base_estimator(y, np.unique(y))
    assert base.get_params()["base_score"] == pytest.approx(y.mean())
    assert notes and "prevalence" in notes[0]
    monkeypatch.setattr(xgb, "__version__", "3.0.0")
    base, notes = sel._base_estimator(y, np.unique(y))
    assert base.get_params()["base_score"] is None and not notes


def test_continuous_target_rejected(easy):
    X, _ = easy
    with pytest.raises(ValueError, match="continuous"):
        ShadowFeatureSelector(rf()).fit(X, np.linspace(0, 1, len(X)))


def test_single_class_raises():
    with pytest.raises(ValueError, match="two classes"):
        ShadowFeatureSelector(rf()).fit(np.zeros((10, 2)), np.zeros(10))


# ------------------------------------------------------------ fit parameters
def test_sample_weight_is_forwarded(easy):
    X, y = easy
    RandomImportance.calls.clear()
    w = np.ones(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LearnerWarning)
        ShadowFeatureSelector(RandomImportance(), importance="native", n_iter=8, n_shadows=4).fit(
            X, y, sample_weight=w
        )
    assert RandomImportance.calls[-1] == ((len(y), 10), len(y))


def test_sample_weight_is_split_in_permutation_mode(easy):
    X, y = easy
    RandomImportance.calls.clear()
    ShadowFeatureSelector(
        RandomImportance(), importance="permutation", n_iter=8, n_repeats=1, test_size=0.2
    ).fit(X, y, sample_weight=np.ones(len(y)))
    assert RandomImportance.calls[-1][1] == int(len(y) * 0.8)


def test_sample_weight_through_pipeline(easy):
    X, y = easy
    pipe = make_pipeline(
        ShadowFeatureSelector(rf(), n_iter=8, random_state=0), LogisticRegression()
    )
    pipe.fit(X, y, shadowfeatureselector__sample_weight=np.ones(len(y)))
    assert pipe.score(X, y) > 0.7


def test_in_pipeline(easy):
    X, y = easy
    pipe = make_pipeline(
        ShadowFeatureSelector(rf(), importance="native", n_iter=10, random_state=0),
        LogisticRegression(),
    ).fit(X, y)
    assert pipe.score(X, y) > 0.7


def test_verbose_prints(easy, capsys):
    X, y = easy
    ShadowFeatureSelector(rf(), n_iter=8, verbose=1, random_state=0).fit(X, y)
    out = capsys.readouterr().out
    assert "round 8/8" in out


def test_native_without_importances_raises(easy):
    X, y = easy
    with pytest.raises(TypeError):
        ShadowFeatureSelector(LogisticRegression(), importance="native", n_iter=8).fit(X, y)


@pytest.mark.parametrize(
    "kw",
    [
        dict(shadow="bad"),
        dict(importance="bad"),
        dict(n_iter=0),
        dict(threshold_quantile=0),
        dict(alpha=1.5),
        dict(alpha="x"),
        dict(test_size=1.0),
        dict(n_repeats=0),
        dict(min_minority=-1),
        dict(n_shadows=0),
        dict(correction="holm"),
    ],
)
def test_bad_params(easy, kw):
    X, y = easy
    with pytest.raises(ValueError):
        ShadowFeatureSelector(rf(), **kw).fit(X, y)


def test_xgboost_native():
    xgb = pytest.importorskip("xgboost")
    X, y = make_dataset(n=1500, p=0.2, delta=1.0, n_real=2, n_noise=2, seed=0)
    est = xgb.XGBClassifier(n_estimators=50, max_depth=3, verbosity=0)
    sel = ShadowFeatureSelector(
        est, importance="native", n_iter=10, keep_tentative=False, random_state=0
    ).fit(X, y)
    assert sel.get_support().tolist() == [True, True, False, False]


# ---------------------------------------------------------------- diagnostics
def test_assess_imbalance():
    y = np.r_[np.ones(20), np.zeros(9980)]
    r = assess_imbalance(y, n_rep=100)
    assert r.n_minority == 20
    assert "Unreliable" in r.recommendation
    assert r.reliable_delta == pytest.approx(np.sqrt(25 / 20))
    big = assess_imbalance(np.r_[np.ones(400), np.zeros(8000)], n_rep=100, n_features=200)
    assert big.min_detectable_delta < r.min_detectable_delta
    assert big.null_keep_rate == pytest.approx(null_keep_rate(200, 20, 1 / 11))
    assert "irrelevant features" in str(big)
