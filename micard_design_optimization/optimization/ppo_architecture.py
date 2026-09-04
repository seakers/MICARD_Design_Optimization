"""Generative transformer Actor/Critic for PPO design synthesis.

Reuses PositionalEncoding / CustomDecoderLayer / CustomTransformerDecoder from
the reference transformer architecture [6][7], but the Actor emits a full
fixed-length gene vector in one pass (generative) rather than a repair
trajectory with a stop token [7]. The design-space weight vector is fed into
the observation (the "informed" idea [4]) and used to scalarize reward.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        tgt2 = F.scaled_dot_product_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(d_model, dim_feedforward, dropout)
            for _ in range(num_layers)])

    def forward(self, tgt, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask)
        return tgt


class Actor(nn.Module):
    """Generative actor: emits a full fixed-length gene vector.

    One output head per gene. Discrete genes use a Categorical over their
    options; continuous genes use a Beta distribution scaled to their range
    (matching the Beta parameterization in the reference repair actor [7]).
    The sampled weight vector is concatenated to the observation [4].
    """

    def __init__(self, device, params, gene_space, num_objectives):
        super().__init__()
        self.device = device
        self.gene_space = gene_space
        self.num_objectives = num_objectives
        self.clip_ratio = params["clip_ratio"]
        self.dense_dim = params.get("dense_dim", 64)

        # Observation = weight vector (num_objectives), broadcast to gene tokens.
        self.encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim, num_layers=2,
            dim_feedforward=self.dense_dim, dropout=0.1)

        # One output head per gene. Discrete -> logits over options; continuous -> Beta(a,b).
        self.heads = nn.ModuleList()
        self.head_types = []
        for var in gene_space:
            if var["type"] == "discrete":
                self.heads.append(nn.Linear(self.dense_dim, len(var["range"])))
                self.head_types.append(("discrete", var["range"]))
            else:  # continuous
                self.heads.append(nn.Linear(self.dense_dim, 2))  # Beta alpha, beta
                self.head_types.append(("continuous", var["range"]))

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=params["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9)

    def _tokens(self, weights):
        """Broadcast the weight vector into num_genes tokens [4]."""
        n = len(self.gene_space)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        # Repeat the weight summary across gene positions as the sequence input.
        seq = w.mean().repeat(n).unsqueeze(0).unsqueeze(-1)  # [1, n, 1]
        return seq

    def forward(self, weights):
        x = self._tokens(weights)
        x = self.encoder(x)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_decoder(x)          # [1, n, dense_dim]
        return x.squeeze(0)                      # [n, dense_dim]

    def sample_action(self, weights):
        """Sample a full gene vector; return genes, summed log-prob, per-gene info."""
        feats = self.forward(weights)
        genes, log_prob = [], 0.0
        for i, (kind, rng) in enumerate(self.head_types):
            out = self.heads[i](feats[i])
            if kind == "discrete":
                dist = torch.distributions.Categorical(logits=out)
                idx = dist.sample()
                log_prob = log_prob + dist.log_prob(idx)
                genes.append(float(rng[int(idx)]))
            else:
                lo, hi = rng
                alpha = F.softplus(out[0]) + 1.0
                beta = F.softplus(out[1]) + 1.0
                dist = torch.distributions.Beta(alpha, beta)
                z = dist.sample()
                log_prob = log_prob + dist.log_prob(z)
                genes.append(float(lo + z.item() * (hi - lo)))
        return np.array(genes, dtype=float), log_prob

    def ppo_update(self, weights_batch, genes_batch, old_logprobs, advantages):
        """PPO-clipped policy update [6][7]."""
        self.optimizer.zero_grad()
        new_logprobs = []
        for weights, genes in zip(weights_batch, genes_batch):
            feats = self.forward(weights)
            lp = 0.0
            for i, (kind, rng) in enumerate(self.head_types):
                out = self.heads[i](feats[i])
                if kind == "discrete":
                    dist = torch.distributions.Categorical(logits=out)
                    target = torch.tensor(rng.index(int(round(genes[i])))
                                          if int(round(genes[i])) in rng else 0,
                                          device=self.device)
                    lp = lp + dist.log_prob(target)
                else:
                    lo, hi = rng
                    alpha = F.softplus(out[0]) + 1.0
                    beta = F.softplus(out[1]) + 1.0
                    dist = torch.distributions.Beta(alpha, beta)
                    z = torch.clamp(torch.tensor((genes[i] - lo) / (hi - lo + 1e-9),
                                                 device=self.device), 1e-4, 1 - 1e-4)
                    lp = lp + dist.log_prob(z)
            new_logprobs.append(lp)

        new_logprobs = torch.stack(new_logprobs)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32,
                                    device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32,
                                  device=self.device)

        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped = torch.where(advantages > 0,
                              (1 + self.clip_ratio) * advantages,
                              (1 - self.clip_ratio) * advantages)
        policy_loss = -torch.mean(torch.min(ratio * advantages, clipped))  # [6][7]
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        kl = torch.mean(old_logprobs - new_logprobs)
        return policy_loss.item(), kl.item()


class Critic(nn.Module):
    """Value network conditioned on the weight vector (scalar value) [6][7]."""

    def __init__(self, device, params, num_objectives):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(num_objectives, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=params["learning_rate"])

    def value(self, weights):
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return self.net(w).squeeze(-1)

    def ppo_update(self, weights_batch, returns):
        self.optimizer.zero_grad()
        w = torch.tensor(np.array(weights_batch), dtype=torch.float32,
                         device=self.device)
        r = torch.tensor(returns, dtype=torch.float32, device=self.device)
        loss = F.mse_loss(self.net(w).squeeze(-1), r)
        loss.backward()
        self.optimizer.step()
        return loss.item()