#!/usr/bin/env python
__doc__ = """
An Approximation of Earth Mover's Distance

An implementation based on the code from
Achlioptas et al., 2017
"Learning Representations and Generative Models for 3D Point Clouds"
https://github.com/optas/latent_3d_points/blob/master/external/structural_losses/approxmatch.cpp
https://github.com/optas/latent_3d_points/blob/master/external/structural_losses/approxmatch.cu

Nicholas Turner <nturner@cs.princeton.edu>, 2018
"""

import itertools

import torch
from torch import nn
from torch.nn.modules import distance


class EMD(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, preds, targets):

        pwdist = self.pairwise_dist(preds, targets)
        match = self.optimal_match(pwdist)

        return self.match_cost(match, pwdist)

    def pairwise_dist(self, p1, p2):

        # largely taken from
        # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
        p1_norm = (p1**2).sum(-1).unsqueeze(-1)
        p2_norm = (p2**2).sum(-1).unsqueeze(-2)

        return p1_norm + p2_norm - 2. * torch.bmm(p1, torch.transpose(p2, -1, -2))

    def optimal_match(self, pwdist):
        """Brute force search"""

        sz = pwdist.size()
        assert sz[-1] == sz[-2], "need equal number of points"

        curr_match = torch.zeros(sz[-2:]) #current match for a single batch
        full_match = torch.zeros(sz) #best matches over batches

        if pwdist.is_cuda:
            curr_match = curr_match.cuda()
            full_match = full_match.cuda()

        num_points = pwdist.size()[-1]
        for b in range(sz[0]): #looping over batches
            best_cost = float("Inf")
            for js in itertools.permutations(range(num_points)):
                curr_match[:] = 0
                curr_match[range(num_points), js] = 1
                cost = torch.sum(curr_match * pwdist[b, ...]).item()

                if cost < best_cost:
                    best_cost = cost
                    full_match[b, ...] = curr_match

        return full_match

    def match_cost(self, match, pwdist):
        # return torch.sum(torch.pow(torch.sum(match * pwdist, -1), 2))
        return torch.sum(match * pwdist)


class ApproxEMD(EMD):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, preds, labels):
        pwdist = self.pairwise_dist(preds, labels)
        match = self.approx_match(pwdist)

        return self.match_cost(match, pwdist)

    def approx_match(self, pwdist):
        sz = pwdist.size()
        assert sz[-1] == sz[-2], f"need equal number of points, got {sz}"

        match = torch.zeros_like(pwdist)

        # amount each predicted point can spend to match
        # with a target
        currency = torch.ones(sz[0], sz[1], 1, dtype=pwdist.dtype)

        # total amount anyone is allowed to spend
        # in order to match with a given target
        cost = torch.ones(sz[0], 1, sz[2], dtype=pwdist.dtype)

        if pwdist.is_cuda:
            currency = currency.cuda()
            cost = cost.cuda()

        # Perform a soft assignment over a series of rounds
        # based on the e^(-pwdist)
        for i in range(4, -3, -1):
            exp_factor = -(4 ** i) if i != -2 else 0.

            # perform a softmax (with stability constant)
            # to determine the amount bid to each target
            # (weighted by the amount of currency possessed each pred)
            bids = torch.exp(exp_factor * pwdist) * cost

            # bids *= currency/ (torch.sum(bids, 1, keepdim=True) + 1e-9)
            bids = bids * currency / (torch.sum(bids, 2, keepdim=True) + 1e-9)

            # limit bids by amount of cost left to consume
            bid_wt = cost / (torch.sum(bids, 1, keepdim=True) + 1e-9)
            bid_wt[bid_wt > 1] = 1

            bids = bids * bid_wt

            # Finalize bids
            match = match + bids

            cost = cost - torch.sum(bids, 1, keepdim=True)
            currency = currency - torch.sum(bids, 2, keepdim=True)

            cost[cost < 0] = 0
            currency[currency < 0] = 0

        # match = match / torch.sum(match, -1, keepdim=True)

        return match
