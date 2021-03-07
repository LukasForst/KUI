import heapq
import math
from collections import defaultdict
from math import sqrt

import kuimaze


class Agent(kuimaze.BaseAgent):
    """
    Simple example of agent class that inherits kuimaze.BaseAgent class
    """

    # noinspection PyMissingConstructor
    def __init__(self, environment):
        # setup basic variables, so pycharm doesn't complain
        self.goal = (0, 0)
        self.position = (0, 0)
        self.environment = environment

    def find_path(self):
        """
        Method that must be implemented by you.
        Expects to return a path_section as a list of positions [(x1, y1), (x2, y2), ... ].
        """

        observation = self.environment.reset()  # must be called first, it is necessary for maze initialization
        goal = observation[1][0:2]
        start = observation[0][0:2]  # initial state (x, y)

        open_set = PQ()
        open_set.add(start, math.inf)
        # for node n, came_from[n] is the node immediately preceding it on the cheapest path from start
        came_from = {}
        # for node n, score[n] is the cost of the cheapest path from start to n currently known
        score = defaultdict(lambda: math.inf)
        score[start] = 0

        while not open_set.is_empty():
            current = open_set.pop()
            if current == goal:
                return self.reconstruct(came_from, current)

            for neighbor, cost in self.environment.expand(current):
                current_score = score[current] + cost
                if current_score < score[neighbor]:
                    came_from[neighbor] = current
                    score[neighbor] = current_score

                    open_set.add(neighbor, score[neighbor] + self.euclidean_distance(neighbor, goal))

            # self.environment.render()

        return None

    @staticmethod
    def euclidean_distance(start, end):
        x1, y1 = start
        x2, y2 = end
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    @staticmethod
    def reconstruct(came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path = [current] + total_path
        return total_path


class PQ(object):
    def __init__(self):
        self._heap = []
        self._exists = set()

    def add(self, position, cost):
        # check if the position already exists in the heap
        if position not in self._exists:
            heapq.heappush(self._heap, (cost, position))
            self._exists.add(position)
        else:
            # if it does, check the cost of the record
            idx, old_cost = next(((idx, old_cost) for idx, (old_cost, p) in enumerate(self._heap) if p == position))
            if old_cost > cost:
                # if the old one is greater, update the value and rebuild the heap
                self._heap[idx] = (cost, position)
                heapq.heapify(self._heap)

    def pop(self):
        _, position = heapq.heappop(self._heap)
        self._exists.remove(position)
        return position

    def is_empty(self):
        return len(self._heap) == 0
