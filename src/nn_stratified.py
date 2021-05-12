import torch
import torch.nn as nn
import torch.nn.functional as F


def stratified_norm(x, s):
    mu_s = []
    std_s = []

    ns = torch.unique(s)
    maxs = torch.max(ns)

    for i in range(maxs.cpu().detach().item()+1):
        if i in ns:
            mu_s.append(torch.mean(x[s == i], 0).unsqueeze(0))
            std_s.append(torch.std(x[s == i], 0).unsqueeze(0))
        else:
            mu_s.append(torch.zeros(1, x.size(1)))
            std_s.append(torch.zeros(1, x.size(1)))
    mu_s = torch.cat(mu_s, 0)
    std_s = torch.cat(std_s, 0)

    mu = mu_s[s]
    std = std_s[s]

    return (x-mu)/(std+1e-8)


class net_stratified_norm(nn.Module):
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
        z = stratified_norm(F.leaky_relu(self.l1(z)), i)
        z = self.d2(z)
        z = stratified_norm(F.leaky_relu(self.l2(z)), i)
        z = self.d3(z)
        z = stratified_norm(F.leaky_relu(self.l3(z)), i)
        z = self.l4(z)

        return z
