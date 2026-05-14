import os
import time
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

import pyepo


class EarlyStopping:

    def __init__(self, patience=3, min_improvement=0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_val = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.best_val_for_stopping = None
        self.counter = 0

    def step(self, val, epoch, model):
        if val < self.best_val:
            self.best_val = val
            self.best_epoch = epoch
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if self.best_val_for_stopping is None:
            self.best_val_for_stopping = val
            self.counter = 0
        else:
            improvement = (self.best_val_for_stopping - val) / (self.best_val_for_stopping + 1e-7)
            if improvement > self.min_improvement:
                self.counter = 0
                self.best_val_for_stopping = val
            else:
                self.counter += 1

        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


class BaseTrainer(ABC):

    def __init__(self, net, opt_model, optimizer, epochs=100,
                 patience=3, max_time=600, device=None, save_path=None):
        self.net = net
        self.opt_model = opt_model
        self.optimizer = optimizer
        self.epochs = epochs
        self.max_time = max_time
        self.save_path = save_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.early_stop = EarlyStopping(patience)
        self.loss_fn = self._init_loss()

    @abstractmethod
    def _init_loss(self):
        pass

    @abstractmethod
    def _compute_loss(self, batch):
        pass

    def train(self, trainloader, valloader=None):
        valloader = valloader or trainloader
        t0 = time.time()

        pbar = tqdm(range(self.epochs), desc="Training")
        for epoch in pbar:
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
