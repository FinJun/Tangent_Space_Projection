#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import osqp

from pyepo.model.opt import optModel
from pyepo.func.utils import (
    gurobi_to_osqp_matrices,
    detect_active_constraints,
    compute_projection_gradient,
)


class BaseProjection(nn.Module):

    def __init__(self, optmodel, epsilon=0.1, forward_smoothing=0.1,
                 ni_eta=0.0, ni_reg=1e-8, ni_delta=1e-6):
        super().__init__()
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")

        self.optmodel = optmodel
        self.epsilon = epsilon
        self.forward_smoothing = forward_smoothing
        self.ni_eta = ni_eta
        self.ni_reg = ni_reg
        self.ni_delta = ni_delta
        self.model_sense = optmodel.modelSense

        self.P, self.A, self.l, self.u, self.eq_mask = gurobi_to_osqp_matrices(
            optmodel._model, self.forward_smoothing, return_eq_mask=True
        )
        self.n_vars = self.P.shape[0]
        self.A_dense = self.A.toarray()

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.P,
            q=np.zeros(self.n_vars),
            A=self.A,
            l=self.l,
            u=self.u,
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            warm_start=True,
            polish=False,
            max_iter=1000
        )

    def forward(self, pred_cost, true_cost):
        return ProjectionFuncUnified.apply(
            pred_cost, true_cost,
            self.prob, self.A_dense, self.l, self.u, self.eq_mask,
            self.epsilon, self.forward_smoothing,
            self.ni_eta, self.ni_reg, self.ni_delta,
            self.model_sense
        )


class ProjectionFuncUnified(Function):

    @staticmethod
    def forward(ctx, pred_cost, true_cost, prob, A_dense, l, u, eq_mask,
                epsilon, forward_smoothing, ni_eta, ni_reg, ni_delta,
                model_sense):
        cp = pred_cost.detach().cpu().numpy()
        batch_size = cp.shape[0]

        active_constraints_list = []
        for i in range(batch_size):
            prob.update(q=model_sense * cp[i])
            res = prob.solve()
            active_mask = detect_active_constraints(res, A_dense, l, u, eq_mask)
            active_constraints_list.append(active_mask)

        ctx.active_constraints_list = active_constraints_list
        ctx.A_dense = A_dense
        ctx.epsilon = epsilon
        ctx.forward_smoothing = forward_smoothing
        ctx.ni_eta = ni_eta
        ctx.ni_reg = ni_reg
        ctx.ni_delta = ni_delta
        ctx.save_for_backward(pred_cost, true_cost)

        loss = 0.5 * torch.sum((pred_cost - true_cost)**2, dim=1).mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        pred_cost, true_cost = ctx.saved_tensors
        active_constraints_list = ctx.active_constraints_list
        A_dense = ctx.A_dense
        epsilon = ctx.epsilon
        forward_smoothing = ctx.forward_smoothing
        ni_eta = ctx.ni_eta
        ni_reg = ctx.ni_reg
        ni_delta = ctx.ni_delta

        device = pred_cost.device
        cp = pred_cost.detach().cpu().numpy()
        c = true_cost.detach().cpu().numpy()
        e = cp - c
        batch_size = len(e)

        H_inv_scalar = 1.0 / forward_smoothing
        grads = np.zeros_like(e)

        for i in range(batch_size):
            grads[i] = compute_projection_gradient(
                error_vec=e[i],
                active_mask=active_constraints_list[i],
                A_dense=A_dense,
                H_inv_scalar=H_inv_scalar,
                epsilon=epsilon,
                regularization=ni_reg,
                ni_eta=ni_eta,
                ni_delta=ni_delta
            )

        grads = torch.tensor(grads, dtype=pred_cost.dtype, device=device)

        return grads * grad_output / batch_size, None, None, None, None, None, None, None, None, None, None, None, None
