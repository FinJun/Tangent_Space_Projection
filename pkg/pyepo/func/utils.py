#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.sparse as spa
import scipy.linalg as la
import gurobipy as gp


def gurobi_to_osqp_matrices(model, forward_smoothing=0.1, return_eq_mask=False):
    model.update()

    vars_ = model.getVars()
    n_vars = len(vars_)
    lbs = np.array([v.lb if v.lb > -1e20 else -np.inf for v in vars_])
    ubs = np.array([v.ub if v.ub < 1e20 else np.inf for v in vars_])

    A_constr = model.getA()
    rhs = np.array(model.getAttr("RHS", model.getConstrs()))
    senses = np.array(model.getAttr("Sense", model.getConstrs()))

    l_constr = np.full(len(rhs), -np.inf)
    u_constr = np.full(len(rhs), np.inf)

    eq_constr_mask = np.zeros(len(rhs), dtype=bool) if return_eq_mask else None

    for i, sense in enumerate(senses):
        if sense == gp.GRB.LESS_EQUAL:
            u_constr[i] = rhs[i]
        elif sense == gp.GRB.GREATER_EQUAL:
            l_constr[i] = rhs[i]
        elif sense == gp.GRB.EQUAL:
            l_constr[i] = rhs[i]
            u_constr[i] = rhs[i]
            if return_eq_mask:
                eq_constr_mask[i] = True

    I = spa.eye(n_vars, format='csr')
    A_osqp = spa.vstack([A_constr, I], format='csc')
    l_osqp = np.concatenate([l_constr, lbs])
    u_osqp = np.concatenate([u_constr, ubs])

    if forward_smoothing > 0:
        P_osqp = forward_smoothing * spa.eye(n_vars, format='csc')
    else:
        P_osqp = spa.csc_matrix((n_vars, n_vars))

    if return_eq_mask:
        eq_mask = np.concatenate([eq_constr_mask, np.zeros(n_vars, dtype=bool)])
        return P_osqp, A_osqp, l_osqp, u_osqp, eq_mask

    return P_osqp, A_osqp, l_osqp, u_osqp


def detect_active_constraints(res, A_dense, l, u, eq_mask=None):
    if res.x is None or res.y is None:
        if eq_mask is not None:
            return eq_mask.copy()
        return np.zeros(A_dense.shape[0], dtype=bool)

    z = A_dense @ res.x
    lower_active = (z - l) < -res.y
    upper_active = (u - z) < res.y
    active_mask = lower_active | upper_active

    if eq_mask is not None:
        active_mask = active_mask | eq_mask

    return active_mask


def cholesky_solve_with_fallback(S, rhs, regularization=1e-8):
    np.fill_diagonal(S, S.diagonal() + regularization)

    try:
        c_factor = la.cho_factor(S, lower=True, overwrite_a=True)
        w = la.cho_solve(c_factor, rhs, overwrite_b=False)
    except la.LinAlgError:
        w = np.linalg.lstsq(S, rhs, rcond=None)[0]

    if not np.all(np.isfinite(w)):
        w = np.zeros_like(rhs)

    return w


def compute_projection_gradient(
    error_vec, active_mask, A_dense, H_inv_scalar, epsilon,
    regularization=1e-8, ni_eta=0.0, ni_delta=1e-6
):
    n_active = active_mask.sum()

    if n_active == 0:
        proj_grad = H_inv_scalar * error_vec
        g = proj_grad + epsilon * error_vec
    else:
        J = A_dense[active_mask]
        rhs = J @ error_vec
        S = J @ J.T
        v = cholesky_solve_with_fallback(S, rhs, regularization=regularization)

        Jt_v = J.T @ v
        u = H_inv_scalar * (error_vec - Jt_v)

        if ni_eta > 0:
            n_normal = H_inv_scalar * Jt_v
            norm_u = np.linalg.norm(u)
            norm_n = np.linalg.norm(n_normal)
            eta_eff = ni_eta * (norm_u / (norm_n + ni_delta))
            g = u + eta_eff * n_normal
        else:
            g = u + epsilon * error_vec

    g = np.clip(g, -1e6, 1e6)
    if not np.all(np.isfinite(g)):
        g = epsilon * error_vec

    return g
