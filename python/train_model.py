import torch
import torch.nn.functional as F
import time


def train_model_cross_subject(model, train_x, test_x, train_y, test_y, train_i, test_i, no_epochs, normalize):
    tr_err = []
    ts_err = []

    tr_acc = []
    ts_acc = []

    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[40], gamma=0.2)

    for epoch in range(no_epochs):

        # Forward
        model.train()

        idx = torch.arange(train_x.shape[0]).long()
        p = model.forward(torch.FloatTensor(train_x), torch.LongTensor(train_i))
        p = F.log_softmax(p, dim=1)
        y = torch.LongTensor(train_y)
        loss = -torch.mean(p[idx, y[idx]])
        tr_err.append(loss.detach().item())
        tr_acc.append(torch.mean((torch.argmax(p, 1) == y).float()).detach().item())

        # Backward
        optim.zero_grad()
        if normalize is True:
            reg = 0
            for p in model.parameters():
                reg += 10 * torch.mean(torch.abs(p))
            loss = loss + reg
        loss.backward()
        optim.step()

        # Validate acc
        model.eval()
        idx = torch.arange(test_x.shape[0]).long()
        y = torch.LongTensor(test_y)
        p = model.forward(torch.FloatTensor(test_x), torch.LongTensor(test_i))
        p = F.log_softmax(p, dim=1)
        loss = -torch.mean(p[idx, y[idx]])
        ts_err.append(loss.detach().item())
        ts_acc.append(torch.mean((torch.argmax(p, 1) == y).float()).detach().item())

        # Scheduler
        scheduler.step()

    return ts_acc[-1]
