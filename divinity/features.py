"""
Deterministic features (trend and Fourier seasonality) as functions of the
integer time step. Features depend only on the time step and the training
length, so the fitted model does not change with the forecast horizon.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureGroup:
    """
    A set of columns that is added to or left out of the model as one unit.

    :param kind: "trend" or "season".
    :param period: seasonal period in time steps (seasonal groups only).
    :param harmonic: Fourier harmonic number, 1 is the fundamental.
    :param detected: True if the period was found in the data rather than
        supplied by the user or the frequency defaults.
    """

    kind: str
    period: float = None
    harmonic: int = None
    detected: bool = False

    @property
    def name(self):
        if self.kind == "trend":
            return "trend"
        return "P={:g},k={}".format(self.period, self.harmonic)

    def columns(self, t, n_train):
        """
        Return a dict of column name to values for time steps ``t``.
        """
        t = np.asarray(t, dtype=float)
        if self.kind == "trend":
            # scaled by training length so coefficients are well conditioned
            return {"trend": t / n_train}
        w = 2 * np.pi * self.harmonic / self.period
        return {
            "sin(" + self.name + ")": np.sin(w * t),
            "cos(" + self.name + ")": np.cos(w * t),
        }


def seasonal_groups(periods, max_harmonics, detected=False):
    """
    Build the candidate Fourier groups for each period. A period P can carry
    at most floor(P / 2) harmonics before they alias onto each other.
    """
    groups = []
    for p in periods:
        n_harmonics = int(min(max_harmonics, max(1, np.floor(p / 2))))
        for k in range(1, n_harmonics + 1):
            groups.append(FeatureGroup("season", float(p), k, detected))
    return groups


def design_matrix(t, groups, n_train, drop_columns=()):
    """
    Assemble the feature matrix for the given groups, without a constant.

    :param t: integer time steps (training or future).
    :param groups: iterable of FeatureGroup.
    :param n_train: length of the training series, used for scaling.
    :param drop_columns: column names to leave out (e.g. columns that are
        identically zero over the training data).
    """
    cols = {}
    for g in groups:
        for name, values in g.columns(t, n_train).items():
            if name not in drop_columns:
                cols[name] = values
    return pd.DataFrame(cols, index=np.asarray(t))


def degenerate_columns(groups, n_train, tol=1e-8):
    """
    Names of columns that are (numerically) zero over the training steps,
    e.g. sin(pi t) for a period of 2. These carry no information.
    """
    t = np.arange(n_train)
    out = set()
    for g in groups:
        for name, values in g.columns(t, n_train).items():
            if np.max(np.abs(values)) < tol:
                out.add(name)
    return out
