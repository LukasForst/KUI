import random
import time
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Dict

from kuimaze import HardMaze, ACTION
from kuimaze.gym_wrapper import state as State

Policy = Dict[State, ACTION]
TIME_LIMIT = 20 * 0.95


def learn_policy(env: HardMaze):
    learner = PolicyLearner(env)
    learner.sarsa_learn()
    return learner.get_policy()


class PolicyLearner:
    def __init__(self, problem: HardMaze, max_episodes=10000, discount_factor=0.8, c=1000):
        self.problem = problem
        self.max_episodes = max_episodes
        self.c = c
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: 0)
        self.visits = Counter()
        self.q_change_threshold = 1e-5

    def sarsa_learn(self):
        start_time = time.time()
        for episode in range(self.max_episodes):
            old_q_values = deepcopy(self.q_values)
            self._execute_episode()
            max_q_change = max(abs(value - old_q_values[state_action]) for state_action, value in self.q_values.items())
            if max_q_change <= self.q_change_threshold or (time.time() - start_time) > TIME_LIMIT:
                break
            else:
                print(f'Max Q value change after episode {episode} is {max_q_change}')
        return self.q_values

    def get_policy(self):
        return {state: self._greedy_policy(state) for state in self.visits.keys()}

    def _execute_episode(self):
        (*state, _) = self.problem.reset()
        state = State(*state)

        action = self._epsilon_greedy(state)

        is_done = False
        while not is_done:
            (*next_state, _), reward, is_done, _ = self.problem.step(action)
            next_state = State(*next_state)

            alpha = self._get_scheduled_value_from_visits(state)

            next_action = self._epsilon_greedy(state)
            update_value = alpha * (reward + self.discount_factor * self.q_values[(next_state, next_action)] -
                                    self.q_values[state, action])

            self.q_values[(state, action)] += update_value
            state, action = next_state, next_action

    def _epsilon_greedy(self, state):
        epsilon = self._get_scheduled_value_from_visits(state)
        action = self._greedy_policy(state) if random.random() > epsilon else self._random_policy()
        self.visits[state] += 1
        return action

    def _get_scheduled_value_from_visits(self, state):
        # scheduling for the alpha and epsilon - should be GLIE - Greedy in the Limit with Infinite Exploration
        # i.e., the more time the state is visited the more greedy it behaves, but there should still be a
        # non-zero chance of any action
        return self.c / ((self.c - 1) + self.visits[state])

    def _random_policy(self):
        return self.problem.action_space.sample()

    def _greedy_policy(self, state):
        return max(
            (action for action in range(self.problem.action_space.n)),
            key=lambda action: self.q_values[state, action]
        )
