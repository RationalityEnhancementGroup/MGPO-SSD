import numpy as np
import pandas as pd
from typing import Tuple, List

def generate_hitid_string(hitids: str) -> Tuple[str]:
    ids = [x.strip() for x in hitids.split(",")]
    return tuple(ids)

def pooled_std_helper(s1: float, s2: float, n1: int, n2: int) -> float:
    """ Computes the pooled standard deviation of two samples.

    Args:
        s1 (float): Std of group 1
        s2 (float): Std of group 2
        n1 (int): Number of observations of group 1
        n2 (int): Number of observations of group 2

    Returns:
        float: Pooled standard deviation
    """
    return float(np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 - 1 + n2 - 1)))

def pooled_std(ns: List[int], stds: List[float]) -> float:
    assert len(ns) > 1 and len(stds) > 1 and len(ns) == len(stds)
    p_n, p_std = ns[0], stds[0]
    for n, std in zip(ns[1:], stds[1:]):
        p_std = pooled_std_helper(p_std, std, p_n, n)
        p_n += n
    return p_std

def corr_repeated_measures(df: pd.DataFrame) -> float:
    """ Calculates correlation among repeated measures using a fisher z transformation.
    https://stats.stackexchange.com/questions/44134/correlation-among-repeated-measures-i-need-an-explanation

    Args:
        df (pd.DataFrame): DataFrame where participants are rows and repeated measures columns. E.g. df.pivot(index="Participant", columns='TrialId')['ExpectedScore']

    Returns:
        float: Correlation
    """
    correlation = df.corr().to_numpy()
    tri = np.triu_indices_from(correlation, 1)

    fisher_z = lambda r: np.arctanh(r)
    inv_fisher_z = lambda z: np.tanh(z)

    upper_correlations = correlation[tri]

    mean_corr = inv_fisher_z(np.mean(fisher_z(upper_correlations)))
    return float(mean_corr)
