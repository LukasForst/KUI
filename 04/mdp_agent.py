import math
import random
from typing import Dict, Tuple, Optional

import kuimaze
from kuimaze import ACTION

State = any
Policy = Dict[State, ACTION]
Evaluation = Dict[State, float]


def find_policy_via_policy_iteration(
        problem: kuimaze.MDPMaze,
        discount_factor: float) -> Policy:
    """
    Finds policy using policy iteration.
    """
    policy, evaluation = init(problem)

    policy_updated = True
    while policy_updated:
        evaluate_policy(policy, evaluation, problem, discount_factor)

        policy_updated = False
        for state in problem.get_all_states():
            # find better action if possible
            best_action, best_evaluation = find_best_action(state, problem, policy, evaluation, discount_factor)
            # update policy if the action is better
            if best_evaluation > evaluation[state.x, state.y]:
                policy_updated = True
                policy[state.x, state.y] = best_action

    return policy


def evaluate_policy(
        policy: Policy,
        evaluation: Evaluation,
        problem: kuimaze.MDPMaze,
        discount_factor: float):
    """
    Populates evaluation with correct values.
    """
    for state in problem.get_all_states():
        # goal state always has its own value
        if problem.is_goal_state(state):
            evaluation[state.x, state.y] = problem.get_state_reward(state)
            continue

        state_eval = sum([probability * evaluation[outcome]
                          for outcome, probability in
                          problem.get_next_states_and_probs(state, policy[state.x, state.y])])

        evaluation[state.x, state.y] = state.reward + discount_factor * state_eval


def find_policy_via_value_iteration(
        problem: kuimaze.MDPMaze,
        discount_factor: float,
        epsilon: float) -> Policy:
    """
    Uses Value iteration to find policy.
    """
    policy, evaluation = init(problem)
    # iterate until change is insignificant
    should_continue = True
    while should_continue:
        change = run_value_iteration(policy, evaluation, problem, discount_factor)
        should_continue = change > epsilon

    return policy


def run_value_iteration(
        policy: Policy,
        evaluation: Evaluation,
        problem: kuimaze.MDPMaze,
        discount_factor: float) -> float:
    """
    Modifies policy and evaluation with a single value iteration step.
    Returns a maximum difference in evaluations.
    """
    change = -math.inf
    for state in problem.get_all_states():
        best_action, best_evaluation = find_best_action(state, problem, policy, evaluation, discount_factor)

        change = max(change, abs(best_evaluation - evaluation[state.x, state.y]))
        evaluation[state.x, state.y] = best_evaluation
        policy[state.x, state.y] = best_action

    return change


def find_best_action(
        state,
        problem: kuimaze.MDPMaze,
        policy: Policy,
        evaluation: Evaluation,
        discount_factor: float) -> Tuple[Optional[ACTION], float]:
    """
    Finds best action for given state, returns action and action evaluation.
    """
    if problem.is_goal_state(state):
        return None, problem.get_state_reward(state)

    best_action, best_evaluation = policy[state.x, state.y], -math.inf
    # evaluate all actions to which we can get from current state
    for action in problem.get_actions(state):
        action_eval = sum([probability * evaluation[outcome]
                           for outcome, probability in problem.get_next_states_and_probs(state, action)])
        # select one with best evaluation
        if best_evaluation < action_eval:
            best_action, best_evaluation = action, action_eval

    return best_action, state.reward + discount_factor * best_evaluation


def init(problem: kuimaze.MDPMaze) -> Tuple[Policy, Evaluation]:
    """
    Creates basic data structures.
    """
    policy = dict()
    evaluation = dict()
    for state in problem.get_all_states():
        evaluation[state.x, state.y] = problem.get_state_reward(state)

        if problem.is_goal_state(state):
            policy[state.x, state.y] = None
            continue

        actions = [action for action in problem.get_actions(state)]
        policy[state.x, state.y] = random.choice(actions)

    return policy, evaluation
