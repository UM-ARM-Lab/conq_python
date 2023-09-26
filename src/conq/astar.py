""" generic A-Star path searching algorithm """

from abc import ABC, abstractmethod
from math import inf as infinity
from typing import Callable, Dict, Iterable, Union, TypeVar, Generic

import matplotlib.pyplot as plt
import numpy as np
import time

import sortedcontainers

# introduce generic type
T = TypeVar("T")


class SearchNode(Generic[T]):
    """Representation of a search node"""

    __slots__ = ("data", "gscore", "fscore", "closed", "came_from", "in_openset")

    def __init__(
            self, data: T, gscore: float = infinity, fscore: float = infinity
    ) -> None:
        self.data = data
        self.gscore = gscore
        self.fscore = fscore
        self.closed = False
        self.in_openset = False
        self.came_from: Union[None, SearchNode[T]] = None

    def __lt__(self, b: "SearchNode[T]") -> bool:
        """Natural order is based on the fscore value & is used by heapq operations"""
        return self.fscore < b.fscore


class SearchNodeDict(Dict[T, SearchNode[T]]):
    """A dict that returns a new SearchNode when a key is missing"""

    def __missing__(self, k) -> SearchNode[T]:
        v = SearchNode(k)
        self.__setitem__(k, v)
        return v


SNType = TypeVar("SNType", bound=SearchNode)


class OpenSet(Generic[SNType]):
    def __init__(self) -> None:
        self.sortedlist = sortedcontainers.SortedList(key=lambda x: x.fscore)

    def push(self, item: SNType) -> None:
        item.in_openset = True
        self.sortedlist.add(item)

    def pop(self) -> SNType:
        item = self.sortedlist.pop(0)
        item.in_openset = False
        return item

    def remove(self, item: SNType) -> None:
        self.sortedlist.remove(item)
        item.in_openset = False

    def __len__(self) -> int:
        return len(self.sortedlist)


class AStar(ABC, Generic[T]):
    __slots__ = ()

    @abstractmethod
    def heuristic_cost_estimate(self, current: T, goal: T) -> float:
        """
        Computes the estimated (rough) distance between a node and the goal.
        The second parameter is always the goal.
        This method must be implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def distance_between(self, n1: T, n2: T) -> float:
        """
        Gives the real distance between two adjacent nodes n1 and n2 (i.e n2
        belongs to the list of n1's neighbors).
        n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
        This method must be implemented in a subclass.
        """

    @abstractmethod
    def neighbors(self, node: T) -> Iterable[T]:
        """
        For a given node, returns (or yields) the list of its neighbors.
        This method must be implemented in a subclass.
        """
        raise NotImplementedError

    def is_goal_reached(self, current: T, goal: T) -> bool:
        """
        Returns true when we can consider that 'current' is the goal.
        The default implementation simply compares `current == goal`, but this
        method can be overwritten in a subclass to provide more refined checks.
        """
        return current == goal

    def reconstruct_path(self, last: SearchNode, reversePath=False) -> Iterable[T]:
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from

        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    # TODO: remember to remove np arr from parameter list
    def astar(
            self, hose_points, start: T, goal: T, reversePath: bool = False
    ) -> Union[Iterable[T], None]:
        if self.is_goal_reached(start, goal):
            return [start]

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-1, 3)
        ax.set_ylim(-2, 1)
        x = np.array([])
        y = np.array([])
        yaws = np.array([])
        # draw the start and goal
        ax.quiver(start[0], start[1], np.cos(start[2]), np.sin(start[2]), color='b')
        ax.quiver(goal[0], goal[1], np.cos(goal[2]), np.sin(goal[2]), color='g')
        self.draw_obstacles(fig, ax)
        plt.draw()
        fig.show()

        openSet: OpenSet[SearchNode[T]] = OpenSet()
        searchNodes: SearchNodeDict[T] = SearchNodeDict()
        startNode = searchNodes[start] = SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal)
        )
        openSet.push(startNode)

        i = 0
        while openSet:
            current = openSet.pop()
            x = np.append(x, current.data[0])
            y = np.append(y, current.data[1])
            yaws = np.append(yaws, current.data[2])

            if i % 100 == 0:
                ax.quiver(x, y, np.cos(yaws), np.sin(yaws), color='b')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                fig.show()

            i += 1

            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)

            current.closed = True

            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue

                tentative_gscore = current.gscore + self.distance_between(
                    current.data, neighbor.data
                )

                if tentative_gscore >= neighbor.gscore:
                    continue

                neighbor_from_openset = neighbor.in_openset

                if neighbor_from_openset:
                    # we have to remove the item from the heap, as its score has changed
                    openSet.remove(neighbor)

                # update the node
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                hscore = self.heuristic_cost_estimate(neighbor.data, goal)
                neighbor.fscore = tentative_gscore + hscore

                # print(f"{str(neighbor.data):20s} {neighbor.gscore:.2f} {hscore:.2f} {neighbor.fscore:.2f}")
                openSet.push(neighbor)

        return None

    def draw_obstacles(self, fig, ax):
        pass


U = TypeVar("U")


def find_path(
        start: U,
        goal: U,
        neighbors_fnct: Callable[[U], Iterable[U]],
        reversePath=False,
        heuristic_cost_estimate_fnct: Callable[[U, U], float] = lambda a, b: infinity,
        distance_between_fnct: Callable[[U, U], float] = lambda a, b: 1.0,
        is_goal_reached_fnct: Callable[[U, U], bool] = lambda a, b: a == b,
) -> Union[Iterable[U], None]:
    """A non-class version of the path finding algorithm"""

    class FindPath(AStar):
        def heuristic_cost_estimate(self, current: U, goal: U) -> float:
            return heuristic_cost_estimate_fnct(current, goal)  # type: ignore

        def distance_between(self, n1: U, n2: U) -> float:
            return distance_between_fnct(n1, n2)

        def neighbors(self, node) -> Iterable[U]:
            return neighbors_fnct(node)  # type: ignore

        def is_goal_reached(self, current: U, goal: U) -> bool:
            return is_goal_reached_fnct(current, goal)

    return FindPath().astar(start, goal, reversePath)
