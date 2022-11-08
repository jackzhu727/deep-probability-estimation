import torch
import torch.nn as nn
import torch.nn.functional as F


class MILAttention(nn.Module):
    def __init__(self,  in_size=1536, hidden_size=512, batch=True):
        super(MILAttention, self).__init__()
        self.L = in_size
        self.D = hidden_size
        self.K = 1
        self.batch = batch

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
        )

    def get_output(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        return Y_prob

    def forward(self, H):
        if self.batch:
            Y_prob = torch.stack([self.get_output(h) for h in H]).squeeze(1)
        else:
            Y_prob = self.get_output(H.squeeze(0))
        
        return Y_prob
