import math
import random
from typing import Dict, Tuple

import kuimaze
from kuimaze import ACTION

DEBUG = False

State = Tuple[int, int]
Policy = Dict[State, ACTION]
Evaluation = Dict[State, float]


def find_policy_via_policy_iteration(
        problem: kuimaze.MDPMaze,
        discount_factor: float) -> Policy:
    policy, evaluation = init(problem)

    policy_updated = True
    while policy_updated:
        evaluate_policy(policy, evaluation, problem, discount_factor)

        policy_updated = False
        for state in problem.get_all_states():
            if problem.is_goal_state(state):
                continue

            best_action, best_evaluation = policy[state.x, state.y], -math.inf
            for action in problem.get_actions(state):
                action_eval = 0
                for outcome, probability in problem.get_next_states_and_probs(state, action):
                    action_eval += probability * evaluation[outcome.x, outcome.y]

                if best_evaluation < action_eval:
                    best_action, best_evaluation = action, action_eval
            best_evaluation = state.reward + discount_factor * best_evaluation
            if best_evaluation > evaluation[state.x, state.y]:
                policy_updated = True
                policy[state.x, state.y] = best_action

    return policy


def evaluate_policy(
        policy: Policy,
        evaluation: Evaluation,
        problem: kuimaze.MDPMaze,
        discount_factor: float):
    for state in problem.get_all_states():
        if problem.is_goal_state(state):
            evaluation[state.x, state.y] = problem.get_state_reward(state)
            continue

        state_eval = 0
        for outcome, probability in problem.get_next_states_and_probs(state, policy[state.x, state.y]):
            state_eval += probability * evaluation[outcome.x, outcome.y]

        evaluation[state.x, state.y] = state.reward + discount_factor * state_eval


def find_policy_via_value_iteration(
        problem: kuimaze.MDPMaze,
        discount_factor: float,
        epsilon: float) -> Policy:
    policy, evaluation = init(problem)
    change = run_value_iteration(policy, evaluation, problem, discount_factor)

    while change > epsilon:
        change = run_value_iteration(policy, evaluation, problem, discount_factor)
        visualize(problem, policy, evaluation)

    return policy


def run_value_iteration(
        policy: Policy,
        evaluation: Evaluation,
        problem: kuimaze.MDPMaze,
        discount_factor: float):
    change = 0

    for state in problem.get_all_states():
        if problem.is_goal_state(state):
            continue

        best_action, best_evaluation = policy[state.x, state.y], -math.inf
        for action in problem.get_actions(state):
            action_eval = 0
            for outcome, probability in problem.get_next_states_and_probs(state, action):
                action_eval += probability * evaluation[outcome.x, outcome.y]

            if best_evaluation < action_eval:
                best_action, best_evaluation = action, action_eval

        policy[state.x, state.y] = best_action
        new_eval = state.reward + discount_factor * best_evaluation

        change = max(change, abs(new_eval - evaluation[state.x, state.y]))
        evaluation[state.x, state.y] = new_eval

    return change


def init(problem: kuimaze.MDPMaze) -> Tuple[Policy, Evaluation]:
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


def visualize(problem, policy, evaluation):
    if not DEBUG:
        return

    problem.visualise(get_visualisation_values(problem, policy))
    problem.render()

    problem.visualise(get_visualisation_values(problem, evaluation))
    problem.render()

    print(policy)


def get_visualisation_values(problem, dic):
    ret = []
    for state in problem.get_all_states():
        ret.append({'x': state.x, 'y': state.y, 'value': dic[state.x, state.y]})
    return ret
