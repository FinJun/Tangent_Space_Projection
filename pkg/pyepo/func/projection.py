#!/usr/bin/env python
# coding: utf-8


from pyepo.func.base import BaseProjection


class Projection(BaseProjection):

    def __init__(self, optmodel, epsilon=0.1, forward_smoothing=0.1):

        super().__init__(
            optmodel=optmodel,
            epsilon=epsilon,
            forward_smoothing=forward_smoothing,
            ni_eta=0.0,
            ni_reg=1e-8,
            ni_delta=1e-6
        )
