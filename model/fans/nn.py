import torch.nn as nn

class MLP4(nn.Module):
    """ a simple 4-layer MLP """
    def __init__(self, nin, nout, nh, batch_norm=False):
        super().__init__()
        if batch_norm:
            self.net = nn.Sequential(
                nn.Linear(nin, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(0.2),
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(0.2),
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(0.2),
                nn.Linear(nh, nout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(nin, nh),
                nn.LeakyReLU(0.2),
                nn.Linear(nh, nh),
                nn.LeakyReLU(0.2),
                nn.Linear(nh, nh),
                nn.LeakyReLU(0.2),
                nn.Linear(nh, nout),
            )
    def forward(self, x):
        return self.net(x)


class MLP1layer(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)