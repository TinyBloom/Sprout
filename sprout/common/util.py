import numpy as np
from scipy.stats import rankdata


# Use sigmoid function to adjust the distribution
def adjusted_sigmoid(x, center=0.2, steepness=10):
    """
    Adjusted sigmoid function, which controls the center and steepness
    center: The center point (0-1), the smaller it is, the higher the overall score
    steepness: Steepness, the larger the value, the steeper the curve
    """
    return 100 / (1 + np.exp(-steepness * (x / 100 - center)))


def segment_mapping(scores):
    """
    Use piecewise functions to map scores
    Piecewise functions can adopt different transformation strategies for different score ranges
    """
    result = np.zeros_like(scores)

    # Divide the scores into different segments
    negative_mask = scores < 0
    very_low_mask = (scores >= 0) & (scores < 1)
    low_mask = (scores >= 1) & (scores < 10)
    high_low_mask = (scores >= 10) & (scores < 40)
    mid_mask = (scores >= 40) & (scores < 60)
    mid_high_mask = (scores >= 60) & (scores < 85)
    high_mask = scores >= 85

    # Negative score range: Map to the range 0 - 1
    result[negative_mask] = (scores[negative_mask] + 5) / 10

    # Low score range: Map to the range 10 - 20
    result[very_low_mask] = 10 + (scores[very_low_mask]) * 20

    # Low score range: Map to the range 60 - 65
    result[low_mask] = 60 + (scores[low_mask] / 10) * 5

    # Low score range: Map to the range 65 - 70
    result[high_low_mask] = 65 + ((scores[high_low_mask] - 10) / 30) * 5

    # Mid score range: Map to the range 70 - 80
    result[mid_mask] = 70 + ((scores[mid_mask] - 40) / 20) * 10

    # Mid score range: Map to the range 80 - 90
    result[mid_high_mask] = 80 + ((scores[mid_high_mask] - 60) / 26) * 10

    # High score range: Map to the range 90 - 100
    result[high_mask] = 90 + ((scores[high_mask] - 85) / 16) * 10

    return result


# Remap scores based on rank
def percentile_remap(scores, target_median=55):
    """
    Remap the scores so that the median reaches the target value
    scores: Original scores
    target_median: Desired median value (0-100)
    """
    ranks = rankdata(scores) - 1  # Ranking starting from 0
    n = len(scores)

    # Calculate the target distribution
    # Here, an exponential function is used to create a distribution biased towards higher scores
    target_percentiles = np.exp(np.log(target_median / 100) * (1 - ranks / n)) * 100

    return target_percentiles


def adjust_health_score(scores, min_boost=20, mid_target=70, power=0.7):
    """
    Comprehensive adjustment of health scores

    Parameters:
    scores: Original health scores (0-100)
    min_boost: Minimum score boost (0-100)
    mid_target: Target value for middle scores (50) (0-100)
    power: Power function parameter (0-1), smaller values result in more significant boosting

    Returns:
    Adjusted health scores (0-100)
    """
    # 1. Boost low scores using the power function
    boosted = 100 * (scores / 100) ** power

    # 2. Ensure that the minimum score reaches the min_boost
    if min_boost > 0:
        boosted = min_boost + (100 - min_boost) * (boosted / 100)

    # 3. Adjust according to the middle target value
    if mid_target != 50:
        # Find the target value for score 50
        target_for_50 = min_boost + (100 - min_boost) * (50 / 100) ** power

        # Calculate the required adjustment factor
        adjustment_factor = (mid_target - target_for_50) / (50 - target_for_50)

        # Apply the adjustment
        below_mid = boosted < mid_target
        above_mid = ~below_mid

        if adjustment_factor > 0:
            # Stretch scores below the middle value
            boosted[below_mid] = target_for_50 + (
                boosted[below_mid] - target_for_50
            ) * (1 + adjustment_factor)
            # Compress scores above the middle value
            factor_high = (100 - mid_target) / (100 - target_for_50)
            boosted[above_mid] = (
                mid_target + (boosted[above_mid] - mid_target) * factor_high
            )

    # 4. Ensure scores are within the 0-100 range
    return np.clip(boosted, 0, 100)


def percentile_scoring(scores, base_scores):
    """
    Scoring based on percentile ranges sorted in ascending order
    """
    # Sort the base_scores in ascending order
    sorted_base = np.sort(base_scores)[::-1]
    n = len(sorted_base)

    # Calculate the percentile points
    p99 = sorted_base[int(n * 0.9999)]  # 99.99% cutoff point
    p99_99 = sorted_base[int(n * 0.99999)]  # 99.99% cutoff point

    # Initialize result array
    result = np.zeros_like(scores)

    # Segmented scoring
    top1_mask = scores >= p99
    next0_99_mask = (scores < p99) & (scores >= p99_99)
    bottom0_01_mask = (scores < p99_99) & (scores >= sorted_base.min())
    below_min_mask = scores < sorted_base.min()

    # Top 1%: 90-100 points
    result[top1_mask] = 90 + (scores[top1_mask] - p99) / (sorted_base.max() - p99) * 10

    # Next 0.99%: 60-90 points
    result[next0_99_mask] = 60 + (scores[next0_99_mask] - p99_99) / (p99 - p99_99) * 30

    # Remaining 0.01%: 20-50 points
    result[bottom0_01_mask] = (
        20
        + (scores[bottom0_01_mask] - sorted_base.min())
        / (p99_99 - sorted_base.min())
        * 30
    )

    # Scores below the minimum: 0-10 points
    result[below_min_mask] = (scores[below_min_mask] / sorted_base.min()) * 10

    return np.clip(result, 0, 100)
