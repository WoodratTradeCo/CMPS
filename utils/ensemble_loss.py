import torch.nn as nn
import torch
import torch.nn.functional as F


class EnsembleLoss(torch.nn.Module):

    def __init__(self):
        super(EnsembleLoss, self).__init__()

    def forward(self, output_i, output_s, output_c, target, kl_i, kl_s):

        loss_function = nn.CrossEntropyLoss()
        loss_i = loss_function(output_i, target)
        loss_s = loss_function(output_s, target)
        loss_c = loss_function(output_c, target)

        divergence = F.kl_div(kl_s.softmax(-1).log(), kl_i.softmax(-1), reduction='sum')
        loss = 2.5 * loss_i + loss_s + loss_c + 0.1 * divergence
        return loss
