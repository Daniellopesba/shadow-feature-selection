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


def test_continuous_target_rejected():
    X, _ = make_dataset(n=200, p=0.2, delta=1.0, n_real=1, n_noise=1, seed=0)
    with pytest.raises(ValueError, match="continuous"):
        UnivariateScreen().fit(X, np.linspace(0, 1, 200))


@pytest.mark.parametrize("block_size", [1, 3, 64])
def test_block_size_does_not_change_results(block_size):
    X, y = make_dataset(n=2000, p=0.05, delta=0.5, n_real=3, n_noise=7, seed=4)
    ref = UnivariateScreen(block_size=32).fit(X, y)
    sc = UnivariateScreen(block_size=block_size).fit(X, y)
    np.testing.assert_allclose(sc.pvalues_, ref.pvalues_)
    np.testing.assert_allclose(sc.effect_, ref.effect_)


def test_missing_values_are_dropped_per_feature():
    X, y = make_dataset(n=3000, p=0.05, delta=0.8, n_real=2, n_noise=2, seed=0)
    Xn = X.copy()
    Xn[::4, 0] = np.nan
    sc = UnivariateScreen(correction="none").fit(Xn, y)
    ok = ~np.isnan(Xn[:, 0])
    ref = stats.mannwhitneyu(X[ok & (y == 1), 0], X[ok & (y == 0), 0]).pvalue
    assert sc.pvalues_[0] == pytest.approx(ref)
    assert sc.missing_fraction_[0] == pytest.approx(0.25)
    np.testing.assert_allclose(
        sc.pvalues_[1:], UnivariateScreen(correction="none").fit(X, y).pvalues_[1:]
    )


def test_categorical_columns_use_chi2():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X, y = make_dataset(n=3000, p=0.05, delta=0.8, n_real=1, n_noise=1, seed=0)
    df = pd.DataFrame(X, columns=["num", "noise"])
    df["cat_signal"] = pd.Categorical(np.where(y == 1, rng.choice(["a", "b"], len(y)), "c"))
    df["cat_noise"] = pd.Categorical(rng.choice(list("abcd"), len(y)))
    df.loc[::9, "cat_noise"] = np.nan
    df["flag"] = rng.random(len(y)) < 0.5
    sc = UnivariateScreen().fit(df, y)
    s = sc.summary().set_index("feature")
    assert s.loc["cat_signal", "test"] == "chi2" and s.loc["cat_signal", "kept"]
    assert s.loc["cat_noise", "test"] == "chi2" and not s.loc["cat_noise", "kept"]
    assert s.loc["cat_noise", "missing_fraction"] > 0.1
    assert s.loc["flag", "test"] == "mann-whitney"
    assert {"num", "cat_signal"} <= set(sc.get_feature_names_out())


def test_sparse_input_matches_dense():
    from scipy import sparse

    X, y = make_dataset(n=1000, p=0.1, delta=0.6, n_real=2, n_noise=3, seed=1)
    X[np.abs(X) < 0.5] = 0.0
    a = UnivariateScreen().fit(X, y)
    b = UnivariateScreen().fit(sparse.csr_matrix(X), y)
    np.testing.assert_allclose(a.pvalues_, b.pvalues_)
