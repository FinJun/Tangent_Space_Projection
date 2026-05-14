"""
LAVA: Learning to Approximate the Value function with Adversarial training.

From: "Solver-Free Neural Network Training for Decision-Focused Learning"

This module implements LAVA loss and adjacent vertices computation.
Exactly matches the original implementation from the solver-free paper.
"""

import time
from collections import deque

import numpy as np
import scipy
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm


class LAVALoss(nn.Module):
    """
    LAVA Loss function.

    Computes loss based on objective difference between predicted solution
    and adjacent vertices.

    Exactly matches the original implementation from solver-free paper.

    Args:
        threshold: Minimum objective difference to include in loss (default: 0.0)
    """

    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, cp, adj_verts, w_rel, mm=1):
        """
        Compute LAVA loss.

        Args:
            cp: Predicted costs (batch_size, n_edges)
            adj_verts: Adjacent vertices (batch_size, max_adj, n_edges)
            w_rel: Optimal solution for true costs (batch_size, n_edges)
            mm: Minimization/maximization multiplier (1 for min, -1 for max)

        Returns:
            Scalar loss
        """
        # Exactly matches original: (adj_verts - w_rel.unsqueeze(1)) * mm * cp.unsqueeze(1)
        diffs = (adj_verts - w_rel.unsqueeze(1)) * mm * cp.unsqueeze(1)
        obj_diffs = torch.sum(diffs, dim=-1)

        # Exactly matches original: obj_diffs[obj_diffs < self.threshold] = 0
        obj_diffs[obj_diffs < self.threshold] = 0

        # Set to 0 if the row is all zeros (padding)
        is_padding = torch.all(adj_verts == 0, dim=-1)
        obj_diffs[is_padding] = 0

        return obj_diffs.mean()


# ============================================================================
# Helper functions from utils/misc.py (original implementation)
# ============================================================================

def find_partial_lexipos(A, required_rows):
    """
    Find an m-×-m selection and ordering of columns of A so that
    only the rows in `required_rows` are lexicopositive.

    Required for initial application of the TNP-rule, as this rule requires a lexicopositive basis to start with.

    Geue, F. (1993). An improved N-tree algorithm for the enumeration of all neighbors of a degenerate vertex.
    Annals of Operations Research, 46(2), 361-391.

    Returns:
        - A list of m column indices (ordered), or None if no solution or timeout after 3 seconds.
    """
    start_time = time.time()
    m, n = A.shape
    R = list(required_rows)

    # Columns with positive and negative entries for each required row
    pos_cols = {i: list(np.where(A[i] > 0)[0]) for i in R}
    neg_cols = {i: set(np.where(A[i] < 0)[0]) for i in R}

    # Sort required rows by fewest positive options, to prune sooner
    R_order = sorted(R, key=lambda i: len(pos_cols[i]))

    pivots = {}
    graph = {}  # ordering constraints between columns

    def has_cycle():
        visited = {}

        def dfs(u):
            visited[u] = 1
            for v in graph.get(u, ()):
                if visited.get(v, 0) == 1:
                    return True
                if visited.get(v, 0) == 0 and dfs(v):
                    return True
            visited[u] = 2
            return False

        for u in graph:
            if visited.get(u, 0) == 0 and dfs(u):
                return True
        return False

    def backtrack_req(k):
        # Check for timeout
        if time.time() - start_time > 3:
            return None

        # All required rows assigned
        if k == len(R_order):
            return True
        row = R_order[k]
        for c in pos_cols[row]:
            # Tentatively assign column c as pivot for this row
            old_graph = {u: set(v) for u, v in graph.items()}
            pivots[row] = c
            graph.setdefault(c, set())

            # Add constraints among required pivots
            for prev_row in R_order[:k]:
                prev_c = pivots[prev_row]
                # If c is negative in prev_row, prev_c -> c
                if c in neg_cols[prev_row]:
                    graph[prev_c].add(c)
                # If prev_c is negative in this row, c -> prev_c
                if prev_c in neg_cols[row]:
                    graph[c].add(prev_c)

            if not has_cycle():
                result = backtrack_req(k + 1)
                if result is None:  # Timeout occurred
                    return None
                if result:
                    return True

            # Backtrack
            graph.clear()
            graph.update(old_graph)
            pivots.pop(row, None)
        return False

    # Assign required-row pivots
    if backtrack_req(0) is None:  # Check for timeout
        return None
    if not backtrack_req(0):
        return None

    # Build set of distinct pivot columns
    P = set(pivots.values())

    # Add extra columns to reach m total distinct columns
    extras_needed = m - len(P)
    extras = [c for c in range(n) if c not in P][:extras_needed]
    if len(extras) < extras_needed:
        return None

    # Combine into full column set S
    S = list(P) + extras

    # Add graph nodes for extras and edges to prevent negatives before pivots
    for c in extras:
        graph.setdefault(c, set())
        for i in R:
            if c in neg_cols[i]:
                graph[pivots[i]].add(c)

    # Topological sort on S
    in_deg = {c: 0 for c in S}
    for u in graph:
        for v in graph[u]:
            if v in in_deg:
                in_deg[v] += 1

    q = deque([c for c in S if in_deg[c] == 0])
    order = []
    while q:
        # Check for timeout
        if time.time() - start_time > 3:
            return None

        u = q.popleft()
        order.append(u)
        for v in graph.get(u, ()):
            if v in in_deg:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)

    return order if len(order) == m else None


def search_for_transition_column(A, basic_indices, non_basic_indices, sol):
    """
    Find an initial transition column to start application of the TNP-rule.

    For more information on transition columns and the TNP-rule, see:

    Geue, F. (1993). An improved N-tree algorithm for the enumeration of all neighbors of a degenerate vertex.
    Annals of Operations Research, 46(2), 361-391.
    """
    while True:
        A_basic = A[:, basic_indices]
        dirs = np.linalg.solve(-A_basic, A[:, non_basic_indices])
        basic_var_values = sol[basic_indices]

        # Add numerical stability threshold
        EPSILON = 1e-10

        i = np.random.choice(len(non_basic_indices))
        entering_var_index = non_basic_indices[i]

        dir = dirs[:, i]

        indices_dir_neg = np.where(dir < -EPSILON)[0]
        basic_var_values_dir_neg = basic_var_values[indices_dir_neg]
        negative_dir_values = dir[indices_dir_neg]
        ratios = -basic_var_values_dir_neg / negative_dir_values

        if ratios.size != 0:
            min_ratio = np.min(ratios)
            min_ratio_indices = indices_dir_neg[np.where(ratios == min_ratio)[0]]
            if min_ratio > 0:
                break
            else:
                min_ratio_index = np.random.choice(min_ratio_indices)
                leaving_var_index = basic_indices[min_ratio_index]

                new_basic_indices = np.copy(basic_indices)
                new_basic_indices[min_ratio_index] = entering_var_index
                new_non_basic_indices = np.copy(non_basic_indices)
                new_non_basic_indices[i] = leaving_var_index

                basic_indices = new_basic_indices
                non_basic_indices = new_non_basic_indices

    return basic_indices, non_basic_indices


def is_lexicofeasible(A):
    """
    Check whether each row of `A` satisfies lexicographic feasibility.

    A matrix is lexicographically feasible if the first non-zero entry
    in every row is strictly positive. This helper routine is used as
    a check before applying the TNP-rule.
    """
    for row in A:
        # Find the first non-zero element
        non_zero_elements = row[row != 0]
        if (len(non_zero_elements) > 0 and non_zero_elements[0] <= 0):
            return False
    return True


# ============================================================================
# Adjacent Vertices Computation (from dataset_augmented.py)
# ============================================================================

def convert_to_slack_form(model):
    """Convert Gurobi model to slack form. Exactly matches original."""
    slack_model = gp.Model("Slack_Form")
    slack_model.setParam('OutputFlag', 0)

    # Copy existing decision variables, making all variables continuous
    var_map = {v.varName: slack_model.addVar(lb=v.lb, vtype='C', name=v.varName)
               for v in model.getVars()}

    slack_model.update()

    for i, constr in enumerate(model.getConstrs()):
        sense = constr.sense
        lhs = gp.LinExpr()
        row = model.getRow(constr)
        for j in range(row.size()):
            lhs += row.getCoeff(j) * var_map[row.getVar(j).varName]
        rhs = constr.rhs

        if sense == GRB.LESS_EQUAL:
            slack_var = slack_model.addVar(lb=0, name=f"slack_{i}")
            slack_model.addConstr(lhs + slack_var == rhs)
        elif sense == GRB.GREATER_EQUAL:
            slack_var = slack_model.addVar(lb=0, name=f"slack_{i}")
            slack_model.addConstr(-lhs + slack_var == -rhs)
        else:
            slack_model.addConstr(lhs == rhs)

    # Add explicit upper bound constraints
    for var in model.getVars():
        if var.ub != float('inf'):
            slack_var = slack_model.addVar(lb=0, name=f"slack_{var.varName}_ub")
            slack_model.addConstr(var_map[var.varName] + slack_var == var.ub)

    # Copy objective
    obj_expr = gp.LinExpr()
    for var in model.getVars():
        obj_expr += var.obj * var_map[var.varName]
    slack_model.setObjective(obj_expr, model.ModelSense)

    slack_model.update()
    return slack_model


def get_constraints_matrix(model):
    """Get constraint matrix from slack model. Exactly matches original."""
    xs = model.getVars()
    A = []
    b = []
    for constr in model.getConstrs():
        if constr.sense != GRB.EQUAL:
            raise Exception("Constraints must be in form Ax == b")
        a_i = []
        for x in xs:
            a_i.append(model.getCoeff(constr, x))
        b_i = constr.rhs
        A.append(a_i)
        b.append(b_i)
    return np.array(A), np.array(b)


def get_adjacent_vertices_non_degenerate_case(A, basic_indices, non_basic_indices, sol, use_scipy):
    """Compute adjacent vertices for non-degenerate case. Exactly matches original."""
    A_basic = A[:, basic_indices]

    if use_scipy:
        dirs = scipy.linalg.solve(-A_basic, A[:, non_basic_indices])
    else:
        dirs = np.linalg.solve(-A_basic, A[:, non_basic_indices])

    basic_var_values = sol[basic_indices]

    adjacent_vertices = []
    for i, entering_var_index in enumerate(non_basic_indices):
        dir = dirs[:, i]

        indices_dir_neg = np.where(dir < 0)[0]
        basic_var_values_dir_neg = basic_var_values[indices_dir_neg]
        negative_dir_values = dir[indices_dir_neg]
        ratios = -basic_var_values_dir_neg / negative_dir_values

        if ratios.size != 0:
            min_ratio = np.min(ratios)
            complete_dir = np.zeros(A.shape[1])
            complete_dir[basic_indices] = dir
            complete_dir[entering_var_index] = 1
            new_sol = sol + min_ratio * complete_dir
            adjacent_vertices.append(new_sol)

    return adjacent_vertices


def get_adjacent_vertices_degenerate_case_helper(A, basic_indices, non_basic_indices, sol, t, B_hat, use_scipy):
    """Helper for degenerate case. Exactly matches original."""
    A_basic = A[:, basic_indices]

    if use_scipy:
        B_hat_prime = scipy.linalg.solve(A_basic, B_hat)
    else:
        B_hat_prime = np.linalg.solve(A_basic, B_hat)

    basic_var_values = sol[basic_indices]
    A_non_basic = A[:, non_basic_indices]

    if use_scipy:
        dirs = scipy.linalg.solve(A_basic, A_non_basic)
    else:
        dirs = np.linalg.solve(A_basic, A_non_basic)

    t_index = np.where(non_basic_indices == t)[0]
    x_B = sol[basic_indices]

    # Set numerical stability threshold
    EPSILON = 1e-10

    # Make sure t is a transition column
    indices_basic_var_zero = np.where(basic_var_values == 0)[0]
    transition_column = dirs[indices_basic_var_zero, :]
    transition_column = transition_column[:, t_index]
    assert np.all(transition_column <= EPSILON), "Transition column must have all elements <= 0"

    adjacent_vertices = []
    new_basic_indices_list = []
    new_non_basic_indices_list = []

    for j, entering_var_index in enumerate(non_basic_indices):
        dir = dirs[:, j]
        indices_dir_pos = np.where(dirs[:, j] > EPSILON)[0]
        basic_var_values_dir_pos = basic_var_values[indices_dir_pos]
        positive_dir_values = dir[indices_dir_pos]
        ratios = basic_var_values_dir_pos / positive_dir_values

        if ratios.size != 0:
            min_ratio = np.min(ratios)
            min_ratio_indices = indices_dir_pos[np.where(ratios == min_ratio)[0]]

            if min_ratio > 0:
                # Transition column
                complete_dir = np.zeros(A.shape[1])
                complete_dir[basic_indices] = dirs[:, j]
                complete_dir[entering_var_index] = -1
                new_sol = sol - min_ratio * complete_dir
                adjacent_vertices.append(new_sol)

            elif min_ratio == 0:
                # Find i
                ratios_2 = dirs[min_ratio_indices, t_index] / dirs[min_ratio_indices, j]
                max_value = np.max(ratios_2)
                num_maximizers = np.sum(ratios_2 == max_value)
                if num_maximizers == 1:
                    i = min_ratio_indices[np.argmax(ratios_2)]
                else:
                    indices = [q for q in range(len(ratios_2)) if ratios_2[q] == max_value]

                    # Identify indices i where dir[i] > 0
                    candidates = min_ratio_indices[indices]

                    # Construct lexicographic comparison vectors
                    lex_vectors = []
                    for i in candidates:
                        ratio = x_B[i] / dir[i]
                        row_vector = B_hat_prime[i, :] / dir[i]
                        lex_vector = np.concatenate([[ratio], row_vector])
                        lex_vectors.append((i, lex_vector))

                    # Choose i with lexicographically smallest vector
                    i = min(lex_vectors, key=lambda x: tuple(x[1]))[0]

                # Perform pivot
                leaving_var_index = basic_indices[i]
                new_basic_indices = np.copy(basic_indices)
                new_basic_indices[i] = entering_var_index
                new_non_basic_indices = np.copy(non_basic_indices)
                new_non_basic_indices[j] = leaving_var_index

                # Store results
                new_basic_indices_list.append(new_basic_indices)
                new_non_basic_indices_list.append(new_non_basic_indices)

    return adjacent_vertices, new_basic_indices_list, new_non_basic_indices_list


def get_adjacent_vertices_degenerate_case(A, basic_indices, non_basic_indices, sol, use_scipy, use_tnp_rule):
    """Compute adjacent vertices for degenerate case. Exactly matches original."""
    basic_indices = np.array(sorted(basic_indices))
    non_basic_indices = np.array(sorted(non_basic_indices))

    # First determine an initial t (to then construct the t-transition degeneracy graph)
    t_found = False
    while not t_found:
        A_basic = A[:, basic_indices]
        basic_var_values = sol[basic_indices]
        A_non_basic = A[:, non_basic_indices]

        if use_scipy:
            dirs = scipy.linalg.solve(A_basic, A_non_basic)
        else:
            dirs = np.linalg.solve(A_basic, A_non_basic)

        indices_basic_var_zero = np.where(basic_var_values == 0)[0]
        dirs_indices_basic_var_zero = dirs[indices_basic_var_zero, :]
        indices_transition_columns = np.where(np.all(dirs_indices_basic_var_zero <= 0, axis=0))[0]
        if len(indices_transition_columns) > 0:
            t_found = True
        else:
            basic_indices, non_basic_indices = search_for_transition_column(A, basic_indices, non_basic_indices, sol)

    t = non_basic_indices[indices_transition_columns][0]

    B = A[:, basic_indices]
    B_inv = np.linalg.inv(B)
    x_B = sol[basic_indices]
    elements_still_to_fix = np.where(x_B == 0)[0]

    if use_tnp_rule:
        tmp = B_inv @ -A
        cols = find_partial_lexipos(tmp, required_rows=elements_still_to_fix)
        if cols is not None:
            basic_indices_2 = cols
            B_hat = -A[:, basic_indices_2]

            if use_scipy:
                B_hat_prime = scipy.linalg.solve(B, B_hat)
            else:
                B_hat_prime = np.linalg.solve(B, B_hat)

            B_inv_L = np.concatenate([np.expand_dims(x_B, axis=1), B_hat_prime], axis=1)
            # B_inv_L needs to be lexicofeasible for TNP rule to work
            if not is_lexicofeasible(B_inv_L):
                raise ValueError("Lexicofeasibility check failed")
        else:
            B_hat = A[:, basic_indices]
    else:
        B_hat = A[:, basic_indices]

    all_adjacent_vertices = set()
    visited_bases = {tuple(sorted(basic_indices))}

    # Initialize queue with initial basis
    queue = [(basic_indices, non_basic_indices)]
    iteration = 0
    while queue:
        iteration += 1
        # Early stopping if desired:
        if iteration > 250:
            break
        curr_basic_indices, curr_non_basic_indices = queue.pop(0)

        # Find adjacent vertices for current solution
        adjacent_vertices, new_basic_indices_list, new_non_basic_indices_list = get_adjacent_vertices_degenerate_case_helper(
            A, curr_basic_indices, curr_non_basic_indices, sol, t, B_hat, use_scipy
        )

        for adjacent_vertex in adjacent_vertices:
            all_adjacent_vertices.add(tuple(adjacent_vertex))

        # Add to queue
        for new_basic_indices, new_non_basic_indices in zip(new_basic_indices_list, new_non_basic_indices_list):
            new_basic_indices_tuple = tuple(sorted(new_basic_indices))
            if new_basic_indices_tuple not in visited_bases:
                visited_bases.add(new_basic_indices_tuple)
                queue.append((sorted(new_basic_indices), sorted(new_non_basic_indices)))

    return all_adjacent_vertices


def get_adjacent_vertices(slack_model, A):
    """
    Get adjacent vertices for a solution. Exactly matches original _get_adjacent_vertices method.
    """
    all_vars = slack_model.getVars()
    sol = np.array([var.x for var in all_vars])

    basis_status = np.array([var.VBasis for var in all_vars])  # 0 for basic, -1 or 1 for non-basic
    basic_indices = np.where(basis_status == 0)[0]
    non_basic_indices = np.where(basis_status != 0)[0]

    sigma = np.sum(sol[basic_indices] < 1e-10)

    use_scipy = True

    if sigma == 0:
        # Non-degenerate vertex
        return get_adjacent_vertices_non_degenerate_case(A, basic_indices, non_basic_indices, sol, use_scipy)
    else:
        # Degenerate vertex; use lexicographic pivoting or TNP rule
        try:
            use_tnp_rule = True
            return get_adjacent_vertices_degenerate_case(A, basic_indices, non_basic_indices, sol, use_scipy,
                                                         use_tnp_rule)
        except ValueError:
            # If TNP rule fails (lexicofeasibility issue), retry without it
            use_tnp_rule = False
            return get_adjacent_vertices_degenerate_case(A, basic_indices, non_basic_indices, sol, use_scipy,
                                                         use_tnp_rule)


def compute_adjacent_vertices_for_instance(opt_model, sol, cost, slack_model, A, num_vars, cache):
    """
    Compute adjacent vertices for a single instance. Matches original dataset logic.
    """
    # Create cache key from solution
    cache_key = tuple(sol)

    # Check cache for adjacent vertices
    if cache_key in cache:
        return cache[cache_key]

    # Set objective and solve
    slack_model_obj = gp.quicksum(cost[k] * slack_model.getVars()[k] for k in range(num_vars))
    slack_model.setObjective(slack_model_obj, opt_model._model.ModelSense)
    slack_model.optimize()

    if slack_model.Status != GRB.OPTIMAL:
        cache[cache_key] = []
        return []

    # Get adjacent vertices
    adjacent_vertices = get_adjacent_vertices(slack_model, A)
    adjacent_vertices = [np.array(av) for av in adjacent_vertices]

    # Discard slack variables
    for j in range(len(adjacent_vertices)):
        adjacent_vertices[j] = adjacent_vertices[j][:num_vars]

    # Remove duplicates
    adjacent_vertices = list({tuple(arr.tolist()): arr for arr in adjacent_vertices}.values())

    # Remove the original solution if present
    if len(adjacent_vertices) > 1:
        sol_array = np.array(sol)
        adjacent_vertices = [v for v in adjacent_vertices if not np.array_equal(v, sol_array)]

    cache[cache_key] = adjacent_vertices
    return adjacent_vertices


def precompute_adjacent_vertices(opt_model, costs, solutions, verbose=True):
    """
    Precompute adjacent vertices for all instances. Matches original dataset logic.

    Args:
        opt_model: PyEPO optimization model
        costs: Cost vectors (n_samples, n_edges)
        solutions: Optimal solutions (n_samples, n_edges)
        verbose: Whether to show progress bar

    Returns:
        List of adjacent vertices for each instance
        Computation time
    """
    model = opt_model._model
    num_vars = len(model.getVars())

    # Convert to slack form once
    t0 = time.time()
    slack_model = convert_to_slack_form(model)
    A, _ = get_constraints_matrix(slack_model)
    adj_vert_computation_time = time.time() - t0

    # Optimize once to initialize
    slack_model.optimize()

    cache = {}
    all_adj_verts = []

    iterator = tqdm(range(len(costs)), desc="Computing adjacent vertices") if verbose else range(len(costs))

    for i in iterator:
        sol = solutions[i]
        cost = costs[i]

        tic = time.time()
        try:
            adj_verts = compute_adjacent_vertices_for_instance(
                opt_model, sol, cost, slack_model, A, num_vars, cache
            )
        except Exception as e:
            adj_verts = []
        adj_vert_computation_time += time.time() - tic

        if len(adj_verts) > 0:
            all_adj_verts.append(np.array(adj_verts))
        else:
            all_adj_verts.append(np.zeros((1, num_vars)))

    return all_adj_verts, adj_vert_computation_time


def pad_adjacent_vertices(adj_verts_list):
    """
    Pad adjacent vertices to same size for batching.

    Args:
        adj_verts_list: List of adjacent vertices arrays

    Returns:
        Padded tensor (n_samples, max_adj, n_edges)
    """
    max_adj = max(len(av) for av in adj_verts_list)
    n_edges = adj_verts_list[0].shape[-1] if len(adj_verts_list[0]) > 0 else 0

    padded = []
    for av in adj_verts_list:
        if len(av) == 0:
            padded.append(np.zeros((max_adj, n_edges)))
        else:
            pad_size = max_adj - len(av)
            if pad_size > 0:
                padding = np.zeros((pad_size, n_edges))
                padded.append(np.vstack([av, padding]))
            else:
                padded.append(av)

    return torch.tensor(np.array(padded), dtype=torch.float32)


# ============================================================================
# shortestPathModelLAVA - Matches solver-free's model (without sink constraint)
# ============================================================================

class shortestPathModelLAVA:
    """
    Shortest path model matching solver-free's implementation.
    Key difference from PyEPO: no sink constraint (for LAVA compatibility).
    """

    def __init__(self, grid):
        self.grid = grid
        self.arcs = self._getArcs()
        self._model = self._getModel()
        self.modelSense = GRB.MINIMIZE

    def _getArcs(self):
        arcs = []
        for i in range(self.grid[0]):
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def _getModel(self):
        m = gp.Model("shortest path")
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.arcs, name="x")
        m.modelSense = GRB.MINIMIZE

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    if v == e[1]:
                        expr += x[e]
                    elif v == e[0]:
                        expr -= x[e]
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    pass  # No sink constraint (matches solver-free)
                else:
                    m.addConstr(expr == 0)

        m.update()
        self._vars = x
        return m

    def setObj(self, cost):
        obj = gp.quicksum(cost[i] * self._vars[e] for i, e in enumerate(self.arcs))
        self._model.setObjective(obj, GRB.MINIMIZE)

    def solve(self):
        self._model.optimize()
        sol = np.array([self._vars[e].x for e in self.arcs])
        obj = self._model.objVal
        return sol, obj


# ============================================================================
# knapsackModelLAVA - LP relaxation of knapsack for LAVA
# ============================================================================

class knapsackModelLAVA:
    """
    Knapsack model (LP relaxation) for LAVA.
    Uses continuous variables with upper bound 1 (LP relaxation of binary).
    """

    def __init__(self, weights, capacities):
        """
        Args:
            weights: Weight matrix (num_constraints, num_items)
            capacities: Capacity for each constraint
        """
        self.weights = np.array(weights)
        self.capacities = np.array(capacities)
        self.num_items = self.weights.shape[1]
        self._model = self._getModel()
        self.modelSense = GRB.MAXIMIZE

    def _getModel(self):
        m = gp.Model("knapsack")
        m.setParam('OutputFlag', 0)
        # LP relaxation: continuous variables with upper bound 1
        x = m.addVars(self.num_items, name="x", lb=0, ub=1)
        m.modelSense = GRB.MAXIMIZE

        # Capacity constraints
        for i in range(len(self.capacities)):
            m.addConstr(
                gp.quicksum(self.weights[i, j] * x[j] for j in range(self.num_items)) <= self.capacities[i],
                name=f"cap_{i}"
            )

        m.update()
        self._vars = x
        return m

    def setObj(self, cost):
        obj = gp.quicksum(cost[j] * self._vars[j] for j in range(self.num_items))
        self._model.setObjective(obj, GRB.MAXIMIZE)

    def solve(self):
        self._model.optimize()
        sol = np.array([self._vars[j].x for j in range(self.num_items)])
        obj = self._model.objVal
        return sol, obj


# ============================================================================
# optDatasetAugmented - Matches original from solver-free/data/dataset_augmented.py
# ============================================================================

class optDatasetAugmented(torch.utils.data.Dataset):
    """
    Augmented dataset with adjacent vertices for LAVA.
    Exactly matches original implementation from solver-free.
    """

    def __init__(self, opt_model, feats, costs):
        """
        Args:
            opt_model: PyEPO optimization model
            feats: Feature vectors (n_samples, feat_dim)
            costs: Cost vectors (n_samples, n_edges)
        """
        self.opt_model = opt_model
        self.feats = feats
        self.costs = costs
        self.adjacent_vertices_cache = {}

        # Build dataset
        (self.sols, self.objs, self.relaxed_sols, self.relaxed_objs,
         self.ctrs, self.adjacent_verts, self.adj_vert_computation_time) = self._build()

    def _build(self):
        """Build dataset with solutions and adjacent vertices."""
        model = self.opt_model._model
        num_vars = len(model.getVars())

        sols, objs, relaxed_sols, relaxed_objs, ctrs, adjacent_verts = [], [], [], [], [], []

        # Convert to slack form once
        t0 = time.time()
        slack_model = convert_to_slack_form(model)
        A, _ = get_constraints_matrix(slack_model)
        adj_vert_computation_time = time.time() - t0

        # Optimize once to initialize
        slack_model.optimize()

        cache = {}

        for i in tqdm(range(len(self.costs)), desc="Computing solutions and adjacent vertices"):
            cost = self.costs[i]

            # Solve optimization problem
            self.opt_model.setObj(cost)
            sol, obj = self.opt_model.solve()

            # For LP, relaxed solution = solution
            relaxed_sol, relaxed_obj = sol, obj

            # Get binding constraints
            constrs = self._get_binding_constrs()

            # Compute adjacent vertices
            tic = time.time()
            cache_key = tuple(sol)

            if cache_key in cache:
                adj_v = cache[cache_key]
            else:
                # Set objective and solve slack model
                slack_model_obj = gp.quicksum(cost[k] * slack_model.getVars()[k] for k in range(num_vars))
                slack_model.setObjective(slack_model_obj, model.ModelSense)
                slack_model.optimize()

                if slack_model.Status == GRB.OPTIMAL:
                    adj_v = get_adjacent_vertices(slack_model, A)
                    adj_v = [np.array(av) for av in adj_v]

                    # Discard slack variables
                    for j in range(len(adj_v)):
                        adj_v[j] = adj_v[j][:num_vars]

                    # Remove duplicates
                    adj_v = list({tuple(arr.tolist()): arr for arr in adj_v}.values())

                    # Remove original solution if present
                    if len(adj_v) > 1:
                        sol_array = np.array(sol)
                        adj_v = [v for v in adj_v if not np.array_equal(v, sol_array)]
                else:
                    adj_v = []

                cache[cache_key] = adj_v

            adj_vert_computation_time += time.time() - tic

            # Store results
            sols.append(sol)
            objs.append([obj])
            relaxed_sols.append(relaxed_sol)
            relaxed_objs.append([relaxed_obj])
            ctrs.append(np.array(constrs) if constrs else np.zeros((1, num_vars)))
            adjacent_verts.append(np.array(adj_v) if len(adj_v) > 0 else np.zeros((1, num_vars)))

        return (np.array(sols), np.array(objs), np.array(relaxed_sols),
                np.array(relaxed_objs), ctrs, adjacent_verts, adj_vert_computation_time)

    def _get_binding_constrs(self):
        """Get binding constraints for current solution."""
        model = self.opt_model._model
        xs = model.getVars()
        constrs = []

        for constr in model.getConstrs():
            if abs(constr.Slack) < 1e-5:
                t_constr = [model.getCoeff(constr, x) for x in xs]
                if constr.sense == GRB.LESS_EQUAL:
                    constrs.append(t_constr)
                elif constr.sense == GRB.GREATER_EQUAL:
                    constrs.append([-c for c in t_constr])
                elif constr.sense == GRB.EQUAL:
                    constrs.append(t_constr)
                    constrs.append([-c for c in t_constr])

        for i, x in enumerate(xs):
            t_constr = [0] * len(xs)
            if x.x <= 1e-5:
                t_constr[i] = -1
                constrs.append(t_constr)
            elif x.ub != float('inf') and x.x >= x.ub - 1e-5:
                t_constr[i] = 1
                constrs.append(t_constr)

        return constrs

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
            torch.FloatTensor(self.relaxed_sols[index]),
            torch.FloatTensor(self.relaxed_objs[index]),
            torch.FloatTensor(self.ctrs[index]),
            torch.FloatTensor(self.adjacent_verts[index])
        )


def collate_fn_lava(batch):
    """Custom collate function for LAVA dataset."""
    x, c, w, z, w_rel, z_rel, ctrs, adj_verts = zip(*batch)

    x = torch.stack(x, dim=0)
    c = torch.stack(c, dim=0)
    w = torch.stack(w, dim=0)
    z = torch.stack(z, dim=0)
    w_rel = torch.stack(w_rel, dim=0)
    z_rel = torch.stack(z_rel, dim=0)

    # Pad constraints and adjacent vertices
    ctrs_padded = torch.nn.utils.rnn.pad_sequence(ctrs, batch_first=True, padding_value=0)
    adj_verts_padded = torch.nn.utils.rnn.pad_sequence(adj_verts, batch_first=True, padding_value=0)

    return x, c, w, z, w_rel, z_rel, ctrs_padded, adj_verts_padded
