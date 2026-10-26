# SPDX-License-Identifier: Apache-2.0
"""Route the local math dataset labels to the upstream math verifier."""

from verl.utils.reward_score.math_dapo import compute_score as math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return math_score(solution_str, ground_truth)
