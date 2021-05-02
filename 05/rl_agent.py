import random
import time
from collections import defaultdict

import kuimaze


def learn_policy(env: kuimaze.HardMaze):
    return RLAgent(env).learn()


class RLAgent:
    def __init__(self, env: kuimaze.HardMaze):
        self.env = env

        self.q_values = defaultdict(lambda: 0)  # initialized Q values for each state and action
        self.states = set()  # set of visited states for which we have the policy
        self.time_limit_seconds = 19  # the real limit is 20 seconds, but we want to be sure to finish on time
        self.start_time = 0  # jut so the idea doesn't complain
        # hyper parameters
        self.gamma = 0.9  # discount factor
        self.alpha = 0.6  # set learning rate arbitrary
        self.exploration_threshold = 0.9  # when the algorithm should pick exploration over exploitation

    def learn(self):
        self.start_time = time.time()
        while self._is_time():
            self._execute_episode()
        return {state: self._greedy_policy(state) for state in self.states}

    def _is_time(self):
        return time.time() - self.start_time < self.time_limit_seconds

    def _execute_episode(self):
        obv = self.env.reset()
        state = obv[0:2]

        action = self._select_action(state)
        done = False
        while not done:
            obv, reward, done, _ = self.env.step(action)
            next_state = obv[0:2]

            next_action = self._select_action(next_state)
            self.q_values[state, action] += self.alpha * (
                    reward + self.gamma * self.q_values[next_state, next_action] - self.q_values[state, action]
            )

            state, action = next_state, next_action

    def _select_action(self, state):
        action = self._greedy_policy(state) \
            if random.random() < self.exploration_threshold \
            else self.env.action_space.sample()  # use random policy

        self.states.add(state)  # mark state as visited
        return action

    def _greedy_policy(self, state):
        possible_actions = [action for action in range(self.env.action_space.n)]
        return max(possible_actions, key=lambda action: self.q_values[state, action])
