import torch
import torch.nn as nn


class RevIN(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.affine_weight)
            x = x * self.stdev + self.mean
        return x


class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        front = x_t[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x_t[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x_t, end], dim=2)
        moving_mean = self.avg(x_pad).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):

    def __init__(self, seq_len: int, pred_len: int, n_assets: int):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_assets = n_assets

        self.revin = RevIN(n_assets)
        self.decomposition = SeriesDecomp(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, return_normalized: bool = False) -> torch.Tensor:
        x = self.revin(x, 'norm')
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1))
        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1))
        x_norm = seasonal_output + trend_output
        x_norm = x_norm.permute(0, 2, 1)
        x_denorm = self.revin(x_norm, 'denorm')

        if return_normalized:
            return x_denorm, x_norm
        return x_denorm

    def normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.revin.mean) / self.revin.stdev
