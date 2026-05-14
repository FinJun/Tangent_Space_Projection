import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pyepo

from .base import BaseTrainer
from .lava import LAVALoss


class MSETrainer(BaseTrainer):

    def _init_loss(self):
        return torch.nn.MSELoss()

    def _compute_loss(self, batch):
        x, c, w, z = batch
        c_hat = self.net(x.to(self.device))
        return self.loss_fn(c_hat, c.to(self.device))


class SPOTrainer(BaseTrainer):

    def __init__(self, *args, epsilon=0.0, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def _init_loss(self):
        return pyepo.func.SPOPlus(self.opt_model, processes=1)

    def _compute_loss(self, batch):
        x, c, w, z = batch
        c_hat = self.net(x.to(self.device))
        loss = self.loss_fn(c_hat, c.to(self.device),
                           w.to(self.device), z.to(self.device)).mean()
        if self.epsilon > 0:
            mse = 0.5 * ((c_hat - c.to(self.device))**2).sum(dim=1).mean()
            loss = loss + self.epsilon * mse
        return loss


class DBBTrainer(BaseTrainer):

    def __init__(self, *args, smoothing=10, epsilon=0.0, **kwargs):
        self.smoothing = smoothing
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def _init_loss(self):
        return pyepo.func.blackboxOpt(self.opt_model, lambd=self.smoothing, processes=1)

    def _compute_loss(self, batch):
        x, c, w, z = batch
        c_hat = self.net(x.to(self.device))
        w_hat = self.loss_fn(c_hat)
        z_hat = (w_hat * c.to(self.device)).sum(dim=1, keepdim=True)
        loss = F.l1_loss(z_hat, z.to(self.device))
        if self.epsilon > 0:
            mse = 0.5 * ((c_hat - c.to(self.device))**2).sum(dim=1).mean()
            loss = loss + self.epsilon * mse
        return loss


class PFYLTrainer(BaseTrainer):

    def __init__(self, *args, n_samples=1, sigma=1.0, epsilon=0.0, **kwargs):
        self.n_samples = n_samples
        self.sigma = sigma
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def _init_loss(self):
        return pyepo.func.perturbedFenchelYoung(self.opt_model, n_samples=self.n_samples,
                                                 sigma=self.sigma, processes=1)

    def _compute_loss(self, batch):
        x, c, w, z = batch
        c_hat = self.net(x.to(self.device))
        loss = self.loss_fn(c_hat, w.to(self.device)).mean()
        if self.epsilon > 0:
            mse = 0.5 * ((c_hat - c.to(self.device))**2).sum(dim=1).mean()
            loss = loss + self.epsilon * mse
        return loss


class NormalInjectionProjectionTrainer(BaseTrainer):

    def __init__(self, *args, epsilon=0.0, forward_smoothing=0.1,
                 ni_eta=0.1, ni_reg=1e-8, ni_delta=1e-6, **kwargs):
        self.epsilon = epsilon
        self.forward_smoothing = forward_smoothing
        self.ni_eta = ni_eta
        self.ni_reg = ni_reg
        self.ni_delta = ni_delta
        super().__init__(*args, **kwargs)

    def _init_loss(self):
        return pyepo.func.NormalInjectionProjection(
            self.opt_model,
            epsilon=self.epsilon,
            forward_smoothing=self.forward_smoothing,
            ni_eta=self.ni_eta,
            ni_reg=self.ni_reg,
            ni_delta=self.ni_delta
        )

    def _compute_loss(self, batch):
        x, c, w, z = batch
        c_hat = self.net(x.to(self.device))
        return self.loss_fn(c_hat, c.to(self.device))


class LAVATrainer(BaseTrainer):

    def __init__(self, *args, threshold=0.0, **kwargs):
        self.threshold = threshold
        self.adj_verts = None
        self.current_batch_idx = 0
        super().__init__(*args, **kwargs)
        self.mm = -1 * self.opt_model.modelSense

    def _init_loss(self):
        return LAVALoss(threshold=self.threshold)

    def set_adjacent_vertices(self, adj_verts_tensor):
        self.adj_verts = adj_verts_tensor.to(self.device)

    def _compute_loss(self, batch):
        if len(batch) == 8:
            x, c, w, z, w_rel, z_rel, bctr, adj_verts = batch
            adj_verts = adj_verts.to(self.device)
            w_for_loss = w_rel.to(self.device)
        elif len(batch) == 4:
            x, c, w, z = batch
            if self.adj_verts is None:
                raise ValueError("Adjacent vertices not set. Call set_adjacent_vertices() first.")
            batch_size = x.shape[0]
            start_idx = self.current_batch_idx
            end_idx = start_idx + batch_size
            adj_verts = self.adj_verts[start_idx:end_idx]
            self.current_batch_idx = end_idx
            w_for_loss = w.to(self.device)
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        c_hat = self.net(x.to(self.device))
        return self.loss_fn(c_hat, adj_verts, w_for_loss, self.mm)

    def train(self, trainloader, valloader=None):
        valloader = valloader or trainloader
        t0 = time.time()

        pbar = tqdm(range(self.epochs), desc="Training")
        for epoch in pbar:
            self.current_batch_idx = 0
            self.net.train()
            for batch in trainloader:
                loss = self._compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                val_regret = pyepo.metric.regret(self.net, self.opt_model, valloader)
                pbar.set_postfix({"val": f"{val_regret*100:.1f}%",
                                  "best": f"{self.early_stop.best_val*100:.1f}%"})

                if self.early_stop.step(val_regret, epoch, self.net):
                    print(f"\nEarly stop @ epoch {epoch}")
                    break

            if time.time() - t0 > self.max_time:
                print(f"\nTime limit @ epoch {epoch}")
                break

        self.early_stop.restore(self.net)

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.net.state_dict(), self.save_path)

        return self.net
