"""
Restricted Boltzmann Machine (RBM) Implementation
A generative stochastic neural network for collaborative filtering.

Architecture:
  - Visible layer: movie ratings (one node per movie)
  - Hidden layer: learned latent features
  - Contrastive Divergence (CD-k) for training
"""

import torch
import torch.nn as nn


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine for collaborative filtering.

    The RBM learns latent user preferences from binary rating patterns,
    then reconstructs ratings for unrated movies to generate recommendations.
    """

    def __init__(self, n_visible, n_hidden, k=1, learning_rate=0.01,
                 momentum=0.5, weight_decay=0.001):
        """
        Args:
            n_visible: Number of visible units (movies).
            n_hidden:  Number of hidden units (latent features).
            k:         Number of Gibbs sampling steps (CD-k).
            learning_rate: Learning rate for weight updates.
            momentum:  Momentum coefficient.
            weight_decay: L2 regularization strength.
        """
        super().__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Parameters
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

        # Momentum buffers (not model parameters)
        self.register_buffer("W_momentum", torch.zeros(n_visible, n_hidden))
        self.register_buffer("v_bias_momentum", torch.zeros(n_visible))
        self.register_buffer("h_bias_momentum", torch.zeros(n_hidden))

    def sample_hidden(self, v):
        """
        Sample hidden units given visible units.
        P(h=1|v) = sigmoid(v @ W + h_bias)
        """
        activation = torch.mm(v, self.W) + self.h_bias
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_visible(self, h):
        """
        Sample visible units given hidden units.
        P(v=1|h) = sigmoid(h @ W^T + v_bias)
        """
        activation = torch.mm(h, self.W.t()) + self.v_bias
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def contrastive_divergence(self, v0, mask=None):
        """
        Perform CD-k learning step.

        Args:
            v0:   Input visible layer (batch of user ratings).
            mask: Binary mask (1 where user rated, 0 otherwise).

        Returns:
            Reconstruction error (MSE on rated items only).
        """
        # Positive phase
        p_h0, h0 = self.sample_hidden(v0)

        # Negative phase (k steps of Gibbs sampling)
        h_k = h0
        for _ in range(self.k):
            p_v_k, v_k = self.sample_visible(h_k)
            # Clamp unrated movies to original values
            if mask is not None:
                v_k = v_k * mask + v0 * (1 - mask)
            p_h_k, h_k = self.sample_hidden(v_k)

        # Compute gradients
        positive_grad = torch.mm(v0.t(), p_h0)
        negative_grad = torch.mm(v_k.t(), p_h_k)

        batch_size = v0.size(0)

        # Update with momentum
        self.W_momentum = (
            self.momentum * self.W_momentum
            + self.lr * (
                (positive_grad - negative_grad) / batch_size
                - self.weight_decay * self.W
            )
        )
        self.v_bias_momentum = (
            self.momentum * self.v_bias_momentum
            + self.lr * (v0 - v_k).mean(dim=0)
        )
        self.h_bias_momentum = (
            self.momentum * self.h_bias_momentum
            + self.lr * (p_h0 - p_h_k).mean(dim=0)
        )

        # Apply updates
        self.W.data += self.W_momentum
        self.v_bias.data += self.v_bias_momentum
        self.h_bias.data += self.h_bias_momentum

        # Compute reconstruction error (only on rated items)
        if mask is not None:
            error = torch.sum((v0 - p_v_k) ** 2 * mask) / torch.sum(mask)
        else:
            error = torch.mean((v0 - p_v_k) ** 2)

        return error.item()

    def predict(self, v, mask=None):
        """
        Generate predictions for all movies given partial ratings.
        Passes through hidden layer and reconstructs.
        """
        with torch.no_grad():
            p_h, _ = self.sample_hidden(v)
            p_v, _ = self.sample_visible(p_h)

            # Average over multiple passes for stability
            for _ in range(9):
                p_h, _ = self.sample_hidden(p_v)
                p_v_new, _ = self.sample_visible(p_h)
                p_v = (p_v + p_v_new) / 2

        return p_v

    def free_energy(self, v):
        """Compute the free energy of visible vector v."""
        vbias_term = torch.mv(v, self.v_bias)
        wx_b = torch.mm(v, self.W) + self.h_bias
        hidden_term = wx_b.exp().add(1).log().sum(dim=1)
        return (-vbias_term - hidden_term).mean()
