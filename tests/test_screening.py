import numpy as np
import pytest
from scipy import stats

from shadowselect import UnivariateScreen, benjamini_hochberg
from shadowselect.simulation import make_dataset


def test_bh_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.5])
    # reference: statsmodels multipletests(method="fdr_bh")
    np.testing.assert_allclose(benjamini_hochberg(p), [0.05, 0.2 / 3, 0.2 / 3, 0.25, 0.5])
    if hasattr(stats, "false_discovery_control"):
        np.testing.assert_allclose(benjamini_hochberg(p), stats.false_discovery_control(p))


def test_screen_recovers_support_at_small_n_min():
    X, y = make_dataset(n=20000, p=0.0025, delta=0.7, n_real=3, n_noise=3, seed=2)
    sc = UnivariateScreen().fit(X, y)
    assert sc.n_minority_ == 50
    assert sc.get_support().tolist() == [True] * 3 + [False] * 3
    assert (sc.effect_[:3] > 0.1).all()
    assert sc.transform(X).shape == (len(y), 3)


def test_pvalues_match_scipy():
    X, y = make_dataset(n=500, p=0.1, delta=0.5, n_real=2, n_noise=2, seed=1)
    sc = UnivariateScreen(correction="none").fit(X, y)
    ref = [stats.mannwhitneyu(X[y == 1, j], X[y == 0, j]).pvalue for j in range(4)]
    np.testing.assert_allclose(sc.pvalues_, ref)


def test_bonferroni_is_stricter():
    X, y = make_dataset(n=4000, p=0.01, delta=0.4, n_real=5, n_noise=20, seed=2)
    bh = UnivariateScreen(alpha=0.2).fit(X, y).get_support().sum()
    bf = UnivariateScreen(alpha=0.2, correction="bonferroni").fit(X, y).get_support().sum()
    assert bf <= bh


def test_summary_and_names():
    pd = pytest.importorskip("pandas")
    X, y = make_dataset(n=1000, p=0.1, delta=1.0, n_real=2, n_noise=2, seed=0)
    df = pd.DataFrame(X, columns=list("abcd"))
    sc = UnivariateScreen().fit(df, y)
    assert list(sc.summary().feature) == list("abcd")
    assert list(sc.get_feature_names_out()) == ["a", "b"]


@pytest.mark.parametrize("kw", [dict(alpha=0), dict(correction="holm")])
def test_bad_params(kw):
    X, y = make_dataset(n=200, p=0.2, delta=1.0, n_real=1, n_noise=1, seed=0)
    with pytest.raises(ValueError):
        UnivariateScreen(**kw).fit(X, y)


def test_single_class_rejected():
    with pytest.raises(ValueError, match="two classes"):
        UnivariateScreen().fit(np.zeros((6, 2)), [1] * 6)
