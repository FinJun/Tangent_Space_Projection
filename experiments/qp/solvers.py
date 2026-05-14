import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import osqp
from scipy import linalg as la, sparse

# Use local qpth with solve_qp_dual
_QPTH_DIR = os.path.join(os.path.dirname(__file__), "qpth")
if _QPTH_DIR not in sys.path:
    sys.path.insert(0, _QPTH_DIR)

from qpth.qp import QPFunction, solve_qp_dual


# ============================================================================
# OSQP Utilities
# ============================================================================

def osqp_interface(P, q, A, lb, ub):
    """OSQP interface for solving QP problems."""
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=False, eps_abs=1e-5, eps_rel=1e-5,
               eps_prim_inf=1e-5, eps_dual_inf=1e-5)
    res = prob.solve()
    return res.x, res.y


# ============================================================================
# Oracle Solver (Gurobi-based, no gradients, for consistent oracle computation)
# ============================================================================

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class OracleSolver(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for OracleSolver. Please install gurobipy.")

    def forward(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # Handle single sample case
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            L = L.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = mu.shape[0]
        n_assets = mu.shape[1]
        device = mu.device

        results = []
        for i in range(batch_size):
            # Construct covariance matrix
            cov = L[i] @ L[i].T
            cov_np = cov.detach().cpu().numpy()
            mu_np = mu[i].detach().cpu().numpy()

            # Solve with Gurobi
            try:
                with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag', 0)  # Suppress output
                    env.start()
                    with gp.Model(env=env) as model:
                        # Variables: w >= 0
                        w = model.addMVar(n_assets, lb=0.0, name="w")

                        # Q = (λ/2)Σ (solver computes x'Qx directly)
                        Q = 0.5 * self.risk_aversion * cov_np
                        model.setObjective(w @ Q @ w - mu_np @ w, GRB.MINIMIZE)

                        # Constraint: sum(w) = 1
                        model.addConstr(w.sum() == 1, "budget")

                        model.optimize()

                        if model.Status == GRB.OPTIMAL:
                            w_opt = torch.from_numpy(w.X).float().to(device)
                        else:
                            # Fallback to uniform weights
                            w_opt = torch.ones(n_assets, device=device) / n_assets
            except Exception:
                # Fallback to uniform weights
                w_opt = torch.ones(n_assets, device=device) / n_assets

            results.append(w_opt)

        w_batch = torch.stack(results, dim=0)

        if squeeze_output:
            w_batch = w_batch.squeeze(0)

        return w_batch


# ============================================================================
# BPQP Solver (from BPQP paper)
# ============================================================================


def qp_osqp_backward(x_value, y_value, P, G, A, grad_output):
    """Backward pass for BPQP using OSQP."""
    nineq, ndim = G.shape
    neq = A.shape[0]
    lambs = y_value[:nineq]
    active_set = np.concatenate([np.argwhere(lambs > 1e-4), np.argwhere(x_value <= 1e-4)])
    bG = G[active_set, :].squeeze()
    bb = np.zeros(neq)
    bh = np.zeros(len(active_set))
    bq = -grad_output.detach().cpu().numpy()
    osnewA = np.vstack([bG, A])
    osnewA = sparse.csc_matrix(osnewA)
    l_new = np.hstack([bh, bb])
    u_new = np.hstack([bh, bb])
    x_grad, y_grad = osqp_interface(P, bq, osnewA, l_new, u_new)
    return x_grad


def create_qp_instances_bpqp(P, q, G, h, A, b):
    """Create QP instances for BPQP."""
    P, q, G, h, A, b = [x.detach().cpu().numpy() for x in [P, q, G, h, A, b]]
    n_ineq = G.shape[0]
    P = sparse.csc_matrix(P)
    osA = np.vstack([G, A])
    osA = sparse.csc_matrix(osA)
    lb = np.hstack([np.zeros(n_ineq), 1.0])
    ub = np.hstack([np.ones(n_ineq), 1.0])
    return P, q, osA, lb, ub


class BPQPFunction(Function):

    @staticmethod
    def forward(ctx, P, q, sign=-1):
        device = P.device
        n_dim = P.shape[0]
        n_ineq = n_dim
        G = torch.diag(torch.ones(n_dim, device=device))
        h = torch.zeros(n_ineq, device=device)
        A = torch.ones(n_dim, device=device).unsqueeze(0)
        b = torch.tensor([1.0], device=device)

        _P, _q, _osA, _l, _u = create_qp_instances_bpqp(P, sign * q, G, h, A, b)
        x_value, y_value = osqp_interface(_P, _q, _osA, _l, _u)

        ctx.P = _P
        ctx.G = G.cpu().numpy()
        ctx.A = A.cpu().numpy()
        yy = torch.cat([
            torch.from_numpy(x_value).to(device).float(),
            torch.from_numpy(y_value).to(device).float(),
        ], dim=0)

        ctx.save_for_backward(yy)
        ctx.device = device
        return yy[:n_dim]

    @staticmethod
    def backward(ctx, grad_output):
        P, G, A = ctx.P, ctx.G, ctx.A
        device = ctx.device
        ndim = P.shape[0]
        nineq = G.shape[0]
        yy = ctx.saved_tensors[0]
        x_star = yy[:ndim]
        lambda_star = yy[ndim:(ndim + nineq)]

        x_grad = qp_osqp_backward(
            x_star.detach().cpu().numpy(),
            lambda_star.detach().cpu().numpy(),
            P, G, A, grad_output
        )

        try:
            x_grad = torch.from_numpy(x_grad).float().to(device)
        except TypeError:
            x_grad = None

        return None, x_grad, None


class BPQPSolver(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

    def forward(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:

        # Handle both batched and non-batched inputs
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            L = L.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = mu.shape[0]
        n_assets = mu.shape[1]
        device = mu.device

        results = []
        for i in range(batch_size):
            # P = λΣ (solver adds ½ internally)
            cov = L[i] @ L[i].T
            P = self.risk_aversion * cov
            q = mu[i]

            w = BPQPFunction.apply(P, q, -1)  # sign=-1 for maximization
            results.append(w)

        w_batch = torch.stack(results, dim=0)

        if squeeze_output:
            w_batch = w_batch.squeeze(0)

        return w_batch


# ============================================================================
# QPTH Solver (OptNet - Differentiable QP)
# ============================================================================

class QPTHBatchSolver(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

    def forward(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            L = L.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = mu.shape[0]
        n_assets = mu.shape[1]
        device = mu.device

        # P = λΣ (solver adds ½ internally)
        # L: (B, n, n) -> cov: (B, n, n)
        cov = torch.bmm(L, L.transpose(1, 2))
        P = self.risk_aversion * cov  # (B, n, n)

        # q = -mu for maximization
        q = -mu  # (B, n)

        # Constraints: Gw <= h (w >= 0 -> -w <= 0)
        G = -torch.eye(n_assets, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # (B, n, n)
        h = torch.zeros(batch_size, n_assets, device=device)  # (B, n)

        # Equality constraint: Aw = b (sum(w) = 1)
        A = torch.ones(1, n_assets, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, n)
        b = torch.ones(batch_size, 1, device=device)  # (B, 1)

        # Solve using QPTH with batch processing
        qpf = QPFunction(verbose=0, maxIter=20)
        try:
            w = qpf(P, q, G, h, A, b)  # (B, n)
        except Exception:
            # Fallback to softmax if solver fails
            w = F.softmax(mu, dim=1)

        if squeeze_output:
            w = w.squeeze(0)

        return w


class CVXPYLayersBatchSolver(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

        # Define CVXPY problem (QP form using Cholesky)
        w = cp.Variable(n_assets)
        mu = cp.Parameter(n_assets, name='mu')
        L = cp.Parameter((n_assets, n_assets), name='L')

        # Q = (λ/2)Σ (solver computes x'Qx directly)
        obj = cp.Minimize(
            0.5 * self.risk_aversion * cp.sum_squares(L.T @ w)
            - mu.T @ w
        )
        cons = [cp.sum(w) == 1, w >= 0]

        prob = cp.Problem(obj, cons)
        self.cvx_layer = CvxpyLayer(prob, parameters=[mu, L], variables=[w])

    def forward(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # Process entire batch at once
        w, = self.cvx_layer(mu, L, solver_args={'solve_method': 'SCS', 'eps': 1e-4})
        return w


# ============================================================================
# Sequential Solvers (solver_batch=1, for fair comparison with BPQP)
# ============================================================================

class QPTHSequentialSolver(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

    def forward(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            L = L.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = mu.shape[0]
        n_assets = mu.shape[1]
        device = mu.device

        # Process each sample individually (like BPQP)
        qpf = QPFunction(verbose=0, maxIter=20)
        results = []
        for i in range(batch_size):
            # Single sample
            cov_i = L[i] @ L[i].T
            # P = λΣ (solver adds ½ internally)
            P_i = (self.risk_aversion * cov_i).unsqueeze(0)  # (1, n, n)
            q_i = -mu[i].unsqueeze(0)  # (1, n)

            G_i = -torch.eye(n_assets, device=device).unsqueeze(0)  # (1, n, n)
            h_i = torch.zeros(1, n_assets, device=device)  # (1, n)
            A_i = torch.ones(1, 1, n_assets, device=device)  # (1, 1, n)
            b_i = torch.ones(1, 1, device=device)  # (1, 1)

            try:
                w_i = qpf(P_i, q_i, G_i, h_i, A_i, b_i)  # (1, n)
            except Exception:
                w_i = F.softmax(mu[i:i+1], dim=1)

            results.append(w_i)

        w = torch.cat(results, dim=0)  # (B, n)

        if squeeze_output:
            w = w.squeeze(0)

        return w


class CVXPYLayersSequentialSolver(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

        # Define CVXPY problem (QP form using Cholesky)
        w = cp.Variable(n_assets)
        mu = cp.Parameter(n_assets, name='mu')
        L = cp.Parameter((n_assets, n_assets), name='L')

        # Q = (λ/2)Σ (solver computes x'Qx directly)
        obj = cp.Minimize(
            0.5 * self.risk_aversion * cp.sum_squares(L.T @ w)
            - mu.T @ w
        )
        cons = [cp.sum(w) == 1, w >= 0]

        prob = cp.Problem(obj, cons)
        self.cvx_layer = CvxpyLayer(prob, parameters=[mu, L], variables=[w])

    def forward(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
            L = L.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = mu.shape[0]

        # Process each sample individually (like BPQP)
        results = []
        for i in range(batch_size):
            mu_i = mu[i:i+1]  # (1, n)
            L_i = L[i:i+1]    # (1, n, n)

            try:
                w_i, = self.cvx_layer(mu_i, L_i, solver_args={'solve_method': 'SCS', 'eps': 1e-4})
            except Exception:
                w_i = F.softmax(mu_i, dim=1)

            results.append(w_i)

        w = torch.cat(results, dim=0)  # (B, n)

        if squeeze_output:
            w = w.squeeze(0)

        return w



class ProjectionLoss(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0,
                 backend: str = 'qpth', max_iter: int = 50, schur_reg: float = 1e-8):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.backend = backend
        self.max_iter = max_iter
        self.schur_reg = schur_reg

    def forward(self, mu_pred: torch.Tensor, mu_true: torch.Tensor,
                L_pred: torch.Tensor) -> torch.Tensor:
        cov_pred = torch.bmm(L_pred, L_pred.transpose(1, 2))
        if self.backend == 'osqp':
            return ProjectionLossOSQPFunction.apply(
                mu_pred, mu_true, cov_pred,
                self.risk_aversion, self.n_assets, self.schur_reg
            )
        else:  # qpth
            return ProjectionLossFunction.apply(
                mu_pred, mu_true, cov_pred,
                self.risk_aversion, self.max_iter, self.schur_reg
            )


def _projection_backward(mu_pred, mu_true, cov_pred, active_mask, risk_aversion, schur_reg):
    """Shared backward pass for projection losses (Schur complement method)."""
    device = mu_pred.device
    batch_size = mu_pred.shape[0]
    n_assets = mu_pred.shape[1]
    dtype = mu_pred.dtype

    e = mu_pred - mu_true
    # H = λΣ
    Q = risk_aversion * cov_pred

    L = torch.linalg.cholesky(Q)
    H_inv_e = torch.cholesky_solve(e.unsqueeze(2), L).squeeze(2)
    eye = torch.eye(n_assets, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    H_inv = torch.cholesky_solve(eye, L)

    ones = torch.ones(n_assets, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    H_inv_1 = torch.bmm(H_inv, ones.unsqueeze(2)).squeeze(2)

    m = active_mask.to(dtype)
    dH_inv = m.unsqueeze(2) * H_inv
    dH_inv_1 = m * H_inv_1
    dH_inv_e = m * H_inv_e

    a = (H_inv_1 * ones).sum(dim=1)
    S22 = dH_inv * m.unsqueeze(1)

    S = torch.zeros(batch_size, n_assets + 1, n_assets + 1, device=device, dtype=dtype)
    S[:, 0, 0] = a
    S[:, 0, 1:] = dH_inv_1
    S[:, 1:, 0] = dH_inv_1
    S[:, 1:, 1:] = S22
    S = S + schur_reg * torch.eye(n_assets + 1, device=device, dtype=dtype).unsqueeze(0)

    r = torch.zeros(batch_size, n_assets + 1, device=device, dtype=dtype)
    r[:, 0] = (H_inv_e * ones).sum(dim=1)
    r[:, 1:] = dH_inv_e

    w = torch.linalg.solve(S, r.unsqueeze(2)).squeeze(2)
    w0 = w[:, 0].unsqueeze(1)
    w1 = w[:, 1:]
    v = w0 * ones + m * w1

    proj_error = H_inv_e - torch.bmm(H_inv, v.unsqueeze(2)).squeeze(2)
    return proj_error / batch_size


class ProjectionLossFunction(Function):

    @staticmethod
    def forward(ctx, mu_pred, mu_true, cov_pred,
                risk_aversion, max_iter, schur_reg):
        device = mu_pred.device
        batch_size = mu_pred.shape[0]
        n_assets = mu_pred.shape[1]
        dtype = mu_pred.dtype

        # P = λΣ (solver adds ½ internally)
        Q = risk_aversion * cov_pred
        q = -mu_pred

        G = -torch.eye(n_assets, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        h = torch.zeros(batch_size, n_assets, device=device, dtype=dtype)

        A = torch.ones(batch_size, 1, n_assets, device=device, dtype=dtype)
        b = torch.ones(batch_size, 1, device=device, dtype=dtype)

        try:
            with torch.no_grad():
                _, lams, _, slacks = solve_qp_dual(Q, q, G, h, A, b, maxIter=max_iter, verbose=-1)
            # OSQP-style active set detection:
            # For Gx <= h, slack = h - Gx, lam >= 0
            # Constraint is active if slack < lam (same as OSQP's upper_active = (u - z) < y)
            active_mask = slacks < lams
        except Exception:
            with torch.no_grad():
                w = F.softmax(mu_pred, dim=1)
            active_mask = w <= 1e-5

        ctx.save_for_backward(mu_pred, mu_true, cov_pred, active_mask)
        ctx.risk_aversion = risk_aversion
        ctx.schur_reg = schur_reg

        mse_loss = 0.5 * torch.sum((mu_pred - mu_true) ** 2, dim=1).mean()
        return mse_loss

    @staticmethod
    def backward(ctx, grad_output):
        mu_pred, mu_true, cov_pred, active_mask = ctx.saved_tensors
        proj = _projection_backward(
            mu_pred, mu_true, cov_pred, active_mask,
            ctx.risk_aversion, ctx.schur_reg
        )
        return proj * grad_output, None, None, None, None, None


class ProjectionLossOSQPFunction(Function):

    @staticmethod
    def forward(ctx, mu_pred, mu_true, cov_pred, risk_aversion, n_assets, schur_reg):
        device = mu_pred.device
        batch_size = mu_pred.shape[0]

        mu_np = mu_pred.detach().cpu().numpy()
        cov_np = cov_pred.detach().cpu().numpy()

        active_constraints_list = []

        # Build OSQP constraint matrix A: [sum(w)=1; w>=0]
        A_eq = np.ones((1, n_assets))
        A_ineq = np.eye(n_assets)
        A_dense = np.vstack([A_eq, A_ineq])
        A_sparse = sparse.csc_matrix(A_dense)

        # l = [1, 0, 0, ..., 0], u = [1, inf, inf, ..., inf]
        l = np.array([1.0] + [0.0] * n_assets)
        u = np.array([1.0] + [np.inf] * n_assets)

        for i in range(batch_size):
            # P = λΣ (solver adds ½ internally)
            P = risk_aversion * cov_np[i]
            P = 0.5 * (P + P.T)
            P_sparse = sparse.csc_matrix(P)

            # q = -μ
            q = -mu_np[i]

            # Solve with OSQP
            prob = osqp.OSQP()
            prob.setup(P_sparse, q, A_sparse, l, u,
                      verbose=False, eps_abs=1e-5, eps_rel=1e-5)
            res = prob.solve()

            if res.x is None or res.y is None:
                active_mask = np.zeros(A_dense.shape[0], dtype=bool)
            else:
                # OSQP-style active set detection (same as LP)
                z = A_dense @ res.x
                lower_active = (z - l) < -res.y
                upper_active = (u - z) < res.y
                active_mask = lower_active | upper_active

            active_constraints_list.append(active_mask)

        ctx.active_constraints_list = active_constraints_list
        ctx.A_dense = A_dense
        ctx.risk_aversion = risk_aversion
        ctx.n_assets = n_assets
        ctx.schur_reg = schur_reg
        ctx.save_for_backward(mu_pred, mu_true, cov_pred)

        # Return MSE loss
        mse_loss = 0.5 * torch.sum((mu_pred - mu_true)**2, dim=1).mean()
        return mse_loss

    @staticmethod
    def backward(ctx, grad_output):
        mu_pred, mu_true, cov_pred = ctx.saved_tensors
        active_constraints_list = ctx.active_constraints_list
        A_dense = ctx.A_dense
        risk_aversion = ctx.risk_aversion
        n_assets = ctx.n_assets
        schur_reg = ctx.schur_reg

        device = mu_pred.device
        batch_size = mu_pred.shape[0]

        e = (mu_pred - mu_true).detach().cpu().numpy()
        cov_np = cov_pred.detach().cpu().numpy()

        grads = np.zeros_like(e)

        for i in range(batch_size):
            error_vec = e[i]
            active_mask = active_constraints_list[i]
            n_active = active_mask.sum()

            # H = λΣ
            Q = risk_aversion * cov_np[i]

            if n_active == 0:
                # No active constraints: P_H @ e = Q^{-1} @ e
                try:
                    g = np.linalg.solve(Q, error_vec)
                except np.linalg.LinAlgError:
                    g = error_vec
            else:
                # Active constraints: J = A[active_mask]
                J = A_dense[active_mask]  # (n_active, n_assets)

                try:
                    # Q^{-1} @ e
                    Q_inv_e = np.linalg.solve(Q, error_vec)

                    # Q^{-1} @ J^T
                    Q_inv_Jt = np.linalg.solve(Q, J.T)

                    # Schur complement: S = J @ Q^{-1} @ J^T
                    S = J @ Q_inv_Jt

                    # Add regularization
                    np.fill_diagonal(S, S.diagonal() + schur_reg)

                    # Solve S @ w = J @ Q^{-1} @ e
                    rhs = J @ Q_inv_e

                    try:
                        c_factor = la.cho_factor(S, lower=True, overwrite_a=True)
                        w = la.cho_solve(c_factor, rhs, overwrite_b=False)
                    except la.LinAlgError:
                        w = np.linalg.lstsq(S, rhs, rcond=None)[0]

                    if not np.all(np.isfinite(w)):
                        w = np.zeros_like(rhs)

                    # g = P_H @ e = Q^{-1} @ (e - J^T @ w)
                    g = Q_inv_e - Q_inv_Jt @ w

                except np.linalg.LinAlgError:
                    g = error_vec

            # Clip gradient
            g = np.clip(g, -1e6, 1e6)
            if not np.all(np.isfinite(g)):
                g = np.zeros_like(error_vec)

            grads[i] = g

        grads = torch.tensor(grads, dtype=mu_pred.dtype, device=device)
        grad_mu_pred = grads * grad_output / batch_size

        return grad_mu_pred, None, None, None, None, None


class ProjectionLossTorch(nn.Module):

    def __init__(self, n_assets: int, risk_aversion: float = 1.0,
                 max_iter: int = 50, schur_reg: float = 1e-8,
                 osqp_workers: int = 0):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.max_iter = max_iter
        self.schur_reg = schur_reg
        self.osqp_workers = osqp_workers

    def forward(self, mu_pred: torch.Tensor, mu_true: torch.Tensor,
                L_pred: torch.Tensor) -> torch.Tensor:
        cov_pred = torch.bmm(L_pred, L_pred.transpose(1, 2))
        cov_pred = cov_pred + 1e-6 * torch.eye(self.n_assets, device=cov_pred.device, dtype=cov_pred.dtype).unsqueeze(0)
        return ProjectionLossTorchFunction.apply(
            mu_pred, mu_true, cov_pred,
            self.risk_aversion, self.max_iter, self.schur_reg, self.osqp_workers
        )


class ProjectionLossTorchFunction(Function):
    """
    Projection Loss with OSQP active set detection and batched torch backward.
    """

    @staticmethod
    def forward(ctx, mu_pred, mu_true, cov_pred,
                risk_aversion, max_iter, schur_reg, osqp_workers):
        from concurrent.futures import ThreadPoolExecutor

        device = mu_pred.device
        batch_size = mu_pred.shape[0]
        n_assets = mu_pred.shape[1]

        mu_pred_np = mu_pred.detach().cpu().numpy()
        cov_np = cov_pred.detach().cpu().numpy()

        # Constraints: sum(w) = 1, w >= 0
        A_eq = np.ones((1, n_assets))
        A_ineq = np.eye(n_assets)
        A = np.vstack([A_eq, A_ineq])
        l = np.hstack([1.0, np.zeros(n_assets)])
        u = np.hstack([1.0, np.inf * np.ones(n_assets)])
        A_sparse = sparse.csc_matrix(A)

        def solve_one(i):
            # P = λΣ (solver adds ½ internally)
            Q = risk_aversion * cov_np[i]
            Q = 0.5 * (Q + Q.T)
            q = -mu_pred_np[i]
            P = sparse.csc_matrix(Q)
            prob = osqp.OSQP()
            prob.setup(
                P, q, A_sparse, l, u,
                verbose=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=max_iter
            )
            res = prob.solve()
            if res.x is None or res.y is None:
                return np.zeros(n_assets, dtype=bool)
            z = A @ res.x
            lower_active = (z - l) < -res.y
            upper_active = (u - z) < res.y
            active = lower_active | upper_active
            return active[1:]  # Exclude equality constraint (always active)

        if osqp_workers is not None and osqp_workers > 1 and batch_size > 1:
            max_workers = min(int(osqp_workers), batch_size)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                active_list = list(executor.map(solve_one, range(batch_size)))
        else:
            active_list = [solve_one(i) for i in range(batch_size)]

        active_mask = torch.tensor(
            np.stack(active_list, axis=0),
            dtype=torch.bool,
            device=device
        )

        ctx.save_for_backward(mu_pred, mu_true, cov_pred, active_mask)
        ctx.risk_aversion = risk_aversion
        ctx.schur_reg = schur_reg

        # Return MSE loss
        mse_loss = 0.5 * torch.sum((mu_pred - mu_true) ** 2, dim=1).mean()
        return mse_loss

    @staticmethod
    def backward(ctx, grad_output):
        mu_pred, mu_true, cov_pred, active_mask = ctx.saved_tensors
        proj = _projection_backward(
            mu_pred, mu_true, cov_pred, active_mask,
            ctx.risk_aversion, ctx.schur_reg
        )
        return proj * grad_output, None, None, None, None, None, None


def create_solver(method: str, n_assets: int, risk_aversion: float = 1.0) -> nn.Module:

    if method == 'cvxpy':
        return CVXPYLayersBatchSolver(n_assets, risk_aversion)
    elif method == 'cvxpy_seq':
        # Sequential version for fair comparison with BPQP
        return CVXPYLayersSequentialSolver(n_assets, risk_aversion)
    elif method in ('projection', 'projection_batch'):
        # Projection uses ProjectionLoss for training, QPTHBatchSolver for backtest
        return QPTHBatchSolver(n_assets, risk_aversion)
    elif method == 'mse':
        return QPTHBatchSolver(n_assets, risk_aversion)
    elif method == 'bpqp':
        return BPQPSolver(n_assets, risk_aversion)
    elif method == 'qpth':
        return QPTHBatchSolver(n_assets, risk_aversion)
    elif method == 'qpth_seq':
        # Sequential version for fair comparison with BPQP
        return QPTHSequentialSolver(n_assets, risk_aversion)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from 'mse', 'projection', "
            "'projection_batch', 'bpqp', 'qpth', 'cvxpy', 'qpth_seq', 'cvxpy_seq'"
        )


def calculate_utility(w: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    risk = torch.bmm(w.unsqueeze(1), torch.bmm(cov, w.unsqueeze(2))).squeeze()
    ret = (w * mu).sum(dim=1)
    return ret - 0.5 * risk_aversion * risk
