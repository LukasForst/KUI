from typing import Dict, Tuple

import kuimaze
from kuimaze import ACTION


def find_policy_via_value_iteration(
        problem: kuimaze.MDPMaze,
        discount_factor: float,
        epsilon: float) -> Dict[Tuple[int, int], ACTION]:
    pass


def find_policy_via_policy_iteration(
        problem: kuimaze.MDPMaze,
        discount_factor: float) -> Dict[Tuple[int, int], ACTION]:
    pass
