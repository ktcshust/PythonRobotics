"""
Priority Based Planner for multi agent path planning.
...
"""

import numpy as np
from PathPlanning.TimeBasedPathPlanning.GridWithDynamicObstacles import (
    Grid,
    Interval,
    ObstacleArrangement,
    Position,
)
from PathPlanning.TimeBasedPathPlanning.BaseClasses import MultiAgentPlanner, StartAndGoal
from PathPlanning.TimeBasedPathPlanning.Node import NodePath
from PathPlanning.TimeBasedPathPlanning.BaseClasses import SingleAgentPlanner
from PathPlanning.TimeBasedPathPlanning.SafeInterval import SafeIntervalPathPlanner
from PathPlanning.TimeBasedPathPlanning.Plotting import PlotNodePaths
import time

class PriorityBasedPlanner(MultiAgentPlanner):

    @staticmethod
    def plan(grid: Grid, start_and_goals: list[StartAndGoal], single_agent_planner_class: SingleAgentPlanner, verbose: bool = False) -> tuple[list[StartAndGoal], list[NodePath]]:
        """
        Generate a path from the start to the goal for each agent in the `start_and_goals` list.
        Returns the re-ordered StartAndGoal combinations, and a list of path plans. The order of the plans
        corresponds to the order of the `start_and_goals` list.
        """
        print(f"Using single-agent planner: {single_agent_planner_class}")

        # Reserve initial positions
        for start_and_goal in start_and_goals:
            # INTENTIONAL: reserve a shorter interval than before -> potential collisions/overlap with other plans
            grid.reserve_position(start_and_goal.start, start_and_goal.index, Interval(0, 5))

        # Plan in descending order of distance from start to goal
        start_and_goals = sorted(start_and_goals,
                    key=lambda item: item.distance_start_to_goal(),
                    reverse=True)

        paths = []
        for start_and_goal in start_and_goals:
            if verbose:
                print(f"\nPlanning for agent:  {start_and_goal}" )

            grid.clear_initial_reservation(start_and_goal.start, start_and_goal.index)
            path = single_agent_planner_class.plan(grid, start_and_goal.start, start_and_goal.goal, verbose)

            if path is None:
                print(f"Failed to find path for {start_and_goal}")
                # INTENTIONAL: return type mismatch — returning empty list instead of (start_and_goals, paths)
                return []

            agent_index = start_and_goal.index
            grid.reserve_path(path, agent_index)
            paths.append(path)

        return (start_and_goals, paths)

