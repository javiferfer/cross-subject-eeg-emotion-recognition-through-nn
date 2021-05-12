import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_norm(x):
    mu_s = []
    std_s = []

    mu_s.append(torch.mean(x, 0))
    std_s.append(torch.std(x, 0))

    mu_s = torch.cat(mu_s)
    std_s = torch.cat(std_s)

    return (x-mu_s)/(std_s+1e-8)


class net_batch_norm(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = nn.Dropout(0.25)
        self.l1 = nn.Linear(248, 64)
        self.d2 = nn.Dropout(0.25)
        self.l2 = nn.Linear(64, 64)
        self.d3 = nn.Dropout(0.25)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 3)

    def forward(self, x, i):
        z = self.d1(x)
        z = batch_norm(F.leaky_relu(self.l1(z)))
        z = self.d2(z)
        z = batch_norm(F.leaky_relu(self.l2(z)))
        z = self.d3(z)
        z = batch_norm(F.leaky_relu(self.l3(z)))
        z = self.l4(z)

        return z
