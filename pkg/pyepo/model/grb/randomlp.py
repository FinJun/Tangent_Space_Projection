#!/usr/bin/env python
# coding: utf-8
"""
Random LP model for PyEPO.

Generates random feasible LP instances:
    min  c'x
    s.t. Ax <= b
         x >= 0
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class randomLPModel(optGrbModel):
    """
    Random LP optimization model.

    Attributes:
        n (int): Number of decision variables
        m (int): Number of inequality constraints
        A (np.ndarray): Constraint matrix (m x n)
        b (np.ndarray): RHS vector (m,)
    """

    def __init__(self, n, m=None, seed=None):
        """
        Initialize random LP model.

        Args:
            n (int): Number of variables
            m (int): Number of inequality constraints (default: n//2)
            seed (int): Random seed for reproducibility
        """
        self.n = n
        self.m = m if m is not None else n // 2
        self.lp_seed = seed

        # Generate random constraints
        self.A, self.b = self._generate_constraints(seed)

        # Call parent init (this will call _getModel)
        super().__init__()

    def _generate_constraints(self, seed=None):
        """Generate random feasible constraints."""
        if seed is not None:
            np.random.seed(seed)

        # Random constraint matrix
        A = np.random.randn(self.m, self.n)

        # Generate a strictly feasible point x0 > 0
        x0 = np.abs(np.random.randn(self.n)) + 0.5

        # Set b to make x0 strictly feasible: Ax0 < b
        slack = np.abs(np.random.randn(self.m)) + 0.5
        b = A @ x0 + slack

        return A, b

    def _getModel(self):
        """
        Build the Gurobi model.

        Returns:
            tuple: (model, variables dict)
        """
        # Create model
        m = gp.Model("randomLP")

        # Decision variables: x >= 0
        x = {}
        for i in range(self.n):
            x[i] = m.addVar(lb=0, name=f"x_{i}")

        m.update()

        # Constraints: Ax <= b
        for j in range(self.m):
            expr = gp.quicksum(self.A[j, i] * x[i] for i in range(self.n))
            m.addConstr(expr <= self.b[j], name=f"c_{j}")

        # Set model sense to minimize
        m.modelSense = GRB.MINIMIZE

        # Set dummy objective (will be updated later)
        m.setObjective(gp.quicksum(x[i] for i in range(self.n)))

        return m, x

    @property
    def num_cost(self):
        """Number of cost coefficients."""
        return self.n


def genData(num_data, num_features, n, m=None, deg=1, noise_width=0.5, seed=135):
    """
    Generate synthetic data for random LP.

    Args:
        num_data (int): Number of data points
        num_features (int): Number of input features
        n (int): Number of LP variables (cost dimension)
        m (int): Number of constraints (not used here, just for consistency)
        deg (int): Polynomial degree for feature transformation
        noise_width (float): Width of uniform noise
        seed (int): Random seed

    Returns:
        x (np.ndarray): Features (num_data, num_features)
        c (np.ndarray): Cost vectors (num_data, n)
    """
    np.random.seed(seed)

    # Generate features
    x = np.random.rand(num_data, num_features)

    # Generate random mapping from features to costs
    # Use polynomial features
    B = np.random.randn(num_features ** deg, n)

    # Compute costs
    c = []
    for i in range(num_data):
        # Simple polynomial: just use x^deg
        x_poly = x[i] ** deg
        cost = x_poly @ B[:num_features, :]

        # Add noise
        noise = np.random.uniform(-noise_width, noise_width, n)
        cost = cost + noise

        # Keep costs positive
        cost = np.abs(cost) + 0.1

        c.append(cost)

    return x, np.array(c)
