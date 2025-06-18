from collections import Counter
from statistics import pvariance
from typing import List


def compute_cohens_kappa(ratings1: List[int], ratings2: List[int]) -> float:
    """Calculate Cohen's Kappa for two rating lists."""
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating lists must have the same length")
    n = len(ratings1)
    if n == 0:
        return 0.0

    categories = set(ratings1) | set(ratings2)
    obs_agreement = sum(r1 == r2 for r1, r2 in zip(ratings1, ratings2)) / n

    counts1 = Counter(ratings1)
    counts2 = Counter(ratings2)
    exp_agreement = sum((counts1[c] / n) * (counts2[c] / n) for c in categories)

    if exp_agreement == 1.0:
        return 1.0
    return (obs_agreement - exp_agreement) / (1 - exp_agreement)


def compute_cronbach_alpha(ratings_list: List[List[float]]) -> float:
    """Compute Cronbach's alpha for multiple rating lists."""
    if not ratings_list:
        return 0.0

    num_raters = len(ratings_list)
    num_items = len(ratings_list[0])
    for r in ratings_list:
        if len(r) != num_items:
            raise ValueError("All rating lists must have the same length")

    item_variances = [pvariance(r) for r in ratings_list]
    total_scores = [sum(r[i] for r in ratings_list) for i in range(num_items)]
    total_variance = pvariance(total_scores)

    if total_variance == 0 or num_raters < 2:
        return 0.0

    alpha = (num_raters / (num_raters - 1)) * (
        1 - sum(item_variances) / total_variance
    )
    return alpha
