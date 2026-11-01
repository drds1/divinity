"""
Forward selection of trend and seasonal feature groups.

Groups are scored with BIC plus an extra penalty for searching over many
candidates (in the spirit of the extended BIC of Chen and Chen, 2008).
Without that extra term, trying dozens of candidate periods on pure noise
will eventually find one that looks significant by chance.

Selection can also be run on quasi-differenced data (y_t - rho * y_{t-1}),
which is feasible GLS for AR(1) errors. Ordinary least squares assumes
independent errors, so on a random walk or a strongly autocorrelated series
it finds "significant" trends and cycles that are not there.
"""

from dataclasses import dataclass, field

import numpy as np

from .features import design_matrix


@dataclass
class SelectionResult:
    """
    Outcome of forward selection.

    :param groups: selected FeatureGroups, in the order they were added.
    :param history: list of (group name, score) for each accepted step,
        starting with ("constant", score).
    """

    groups: list
    history: list = field(default_factory=list)


def ols_rss(X, y):
    """
    Residual sum of squares of an ordinary least squares fit.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def score(rss, n, n_params, search_penalty):
    """
    BIC for a Gaussian linear model plus the multiple search penalty.
    """
    rss = max(rss, 1e-12 * n)
    return n * np.log(rss / n) + n_params * np.log(n) + search_penalty


def lag1_autocorrelation(resid):
    """
    AR(1) coefficient of a residual series, using only adjacent pairs where
    both values are observed. Clipped to [-0.95, 1].
    """
    r = np.asarray(resid, dtype=float)
    a, b = r[1:], r[:-1]
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.sum(b[ok] ** 2) == 0:
        return 0.0
    rho = np.sum(a[ok] * b[ok]) / np.sum(b[ok] ** 2)
    return float(np.clip(rho, -0.95, 1.0))


def quasi_difference(values, rho):
    """
    Apply x_t - rho * x_{t-1} to a 1D or 2D array (first row is dropped).
    """
    values = np.asarray(values, dtype=float)
    if rho == 0:
        return values
    return values[1:] - rho * values[:-1]


def forward_select(
    y, candidates, pool_sizes, drop_columns=(), gamma=1.0, forced=(), rho=0.0
):
    """
    Greedily add the candidate group that lowers the score most, until no
    group lowers it.

    A harmonic k > 1 of a period is only eligible once harmonic k - 1 of the
    same period has been selected.

    :param y: training series (NaN allowed, those rows are ignored).
    :param candidates: list of FeatureGroup.
    :param pool_sizes: dict of FeatureGroup to the number of alternatives it
        was chosen from. Each selected group adds 2 * gamma * log(pool size)
        to the score.
    :param drop_columns: degenerate column names to leave out.
    :param gamma: strength of the search penalty. 0 gives plain BIC.
    :param forced: groups that are always included and carry no penalty.
    :param rho: AR(1) coefficient of the errors. Non zero values select on
        quasi-differenced data.
    :return: SelectionResult
    """
    y = np.asarray(y, dtype=float)
    n_train = len(y)
    t = np.arange(n_train)

    forced = list(forced)
    candidates = [g for g in candidates if g not in forced]
    full = design_matrix(t, forced + candidates, n_train, drop_columns)
    columns_of = {
        g: [c for c in design_matrix(t[:1], [g], n_train).columns if c in full.columns]
        for g in forced + candidates
    }
    candidates = [g for g in candidates if columns_of[g]]
    col_pos = {c: i for i, c in enumerate(full.columns)}

    y_t = quasi_difference(y, rho)
    X_t = quasi_difference(np.hstack([np.ones((n_train, 1)), full.to_numpy()]), rho)
    observed = ~np.isnan(y_t)
    y_obs = y_t[observed]
    X_obs = X_t[observed]
    n = len(y_obs)

    def evaluate(groups):
        idx = [0] + [1 + col_pos[c] for g in forced + groups for c in columns_of[g]]
        X = X_obs[:, idx]
        penalty = sum(2 * gamma * np.log(max(pool_sizes[g], 1)) for g in groups)
        return score(ols_rss(X, y_obs), n, X.shape[1], penalty)

    selected = []
    current = evaluate(selected)
    history = [("constant", current)]
    while True:
        eligible = [
            g
            for g in candidates
            if g not in selected
            and (
                g.kind == "trend"
                or g.harmonic == 1
                or any(
                    s.kind == "season"
                    and s.period == g.period
                    and s.harmonic == g.harmonic - 1
                    for s in selected
                )
            )
        ]
        # never use more parameters than about a third of the data
        n_cols = sum(len(columns_of[g]) for g in forced + selected)
        eligible = [g for g in eligible if n_cols + len(columns_of[g]) + 1 < n / 3]
        if not eligible:
            break
        scores = [evaluate(selected + [g]) for g in eligible]
        best = int(np.argmin(scores))
        if scores[best] >= current:
            break
        selected.append(eligible[best])
        current = scores[best]
        history.append((eligible[best].name, current))
    return SelectionResult(forced + selected, history)
