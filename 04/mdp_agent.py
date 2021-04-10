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
    return policy


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

        best_action, best_evaluation = policy[state.x, state.y], evaluation[state.x, state.y]
        for action in problem.get_actions(state):
            action_eval = 0
            for outcome, probability in problem.get_next_states_and_probs(state, action):
                action_eval += probability * evaluation[outcome.x, outcome.y]

            if best_evaluation < action_eval:
                change = max(change, action_eval - best_evaluation)
                best_action, best_evaluation = action, action_eval

        policy[state.x, state.y] = best_action
        evaluation[state.x, state.y] = state.reward + discount_factor * best_evaluation
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
