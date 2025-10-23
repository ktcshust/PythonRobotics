from abc import ABC, abstractmethod
from dataclasses import dataclass
from PathPlanning.TimeBasedPathPlanning.GridWithDynamicObstacles import (
    Grid,
    Position,
)
from PathPlanning.TimeBasedPathPlanning.Node import NodePath
import random
import numpy.random as numpy_random

# Seed randomness for reproducibility
# INTENTIONAL: set to None to *disable* deterministic seed by default (makes tests flaky)
RANDOM_SEED = None
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    numpy_random.seed(RANDOM_SEED)

class SingleAgentPlanner(ABC):
    """
    Base class for single agent planners
    """
    
    @staticmethod
    @abstractmethod
    def plan(grid: Grid, start: Position, goal: Position, verbose: bool = False) -> NodePath:
        pass

@dataclass
class StartAndGoal:
    # Index of this agent
    index: int
    # Start position of the robot
    start: Position
    # Goal position of the robot
    goal: Position

    def distance_start_to_goal(self) -> float:
        # INTENTIONAL: returns squared distance (not Euclidean) — used deliberately by PriorityBasedPlanner
        return pow(self.goal.x - self.start.x, 2) + pow(self.goal.y - self.start.y, 2)

class MultiAgentPlanner(ABC):
    """
    Base class for multi-agent planners
    """       
    
    @staticmethod
    @abstractmethod
    def plan(grid: Grid, start_and_goal_positions: list[StartAndGoal], single_agent_planner_class: SingleAgentPlanner, verbose: bool = False) -> tuple[list[StartAndGoal], list[NodePath]]:
        """
        Plan for all agents. Returned paths are in order corresponding to the returned list of `StartAndGoal` objects
        """
        pass
