#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class shortestPathModel(optGrbModel):
    """
    This class is optimization model for shortest path problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
        direction (str): 'original', 'reverse', or 'cross'
    """

    def __init__(self, grid, reverse=False, cross=False):
        """
        Args:
            grid (tuple of int): size of grid network
            reverse (bool): if True, arcs go left/up instead of right/down
            cross (bool): if True, arcs go right/up (bottom-left to top-right)
        """
        self.grid = grid
        self.reverse = reverse
        self.cross = cross
        self.arcs = self._getArcs()
        super().__init__()

    def _getArcs(self):
        """
        A method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows (always right direction)
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                if self.reverse:
                    arcs.append((v + 1, v))  # left direction
                else:
                    arcs.append((v, v + 1))  # right direction
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                if self.reverse:
                    arcs.append((v + self.grid[1], v))  # up direction
                elif self.cross:
                    arcs.append((v + self.grid[1], v))  # up direction (cross uses right + up)
                else:
                    arcs.append((v, v + self.grid[1]))  # down direction
        return arcs

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("shortest path")
        # varibles
        x = m.addVars(self.arcs, name="x")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= x[e]
                if self.reverse:
                    # source at bottom-right
                    if i == self.grid[0] - 1 and j == self.grid[1] - 1:
                        m.addConstr(expr == -1)
                    # sink at top-left
                    elif i == 0 and j == 0:
                        m.addConstr(expr == 1)
                    else:
                        m.addConstr(expr == 0)
                elif self.cross:
                    # source at bottom-left
                    if i == self.grid[0] - 1 and j == 0:
                        m.addConstr(expr == -1)
                    # sink at top-right
                    elif i == 0 and j == self.grid[1] - 1:
                        m.addConstr(expr == 1)
                    else:
                        m.addConstr(expr == 0)
                else:
                    # source at top-left
                    if i == 0 and j == 0:
                        m.addConstr(expr == -1)
                    # sink at bottom-right
                    elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                        m.addConstr(expr == 1)
                    else:
                        m.addConstr(expr == 0)
        return m, x
