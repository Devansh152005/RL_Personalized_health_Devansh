"""
Thompson Sampling Agent for Contextual Bandits

Implements the Bayesian Thompson Sampling algorithm as described in Appendix A.2
of the paper, with update equations (6) and (7).

The agent uses a linear reward model: r ~ N(θ_a^T v_t, σ²_Ya)
where v_t is the state vector and θ_a are per-action weight vectors.
"""

import numpy as np


class ThompsonSamplingAgent:
    """
    Contextual Thompson Sampling agent.

    Parameters (from Appendix C.2):
        n_actions: number of actions (4: {0, 1, 2, 3})
        state_dim: dimension of state vector v_t (2: [1, l_t])
        mu_0: prior mean for θ_a (0 for all)
        sigma_0: prior covariance scalar (Σ_0a = sigma_0 * I)
        sigma_y: reward noise std (σ_Ya = 25, so σ²_Ya = 625)
    """

    def __init__(
        self,
        n_actions=4,
        state_dim=2,
        mu_0=0.0,
        sigma_0=100.0,
        sigma_y=25.0,
        seed=None,
    ):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.sigma_y_sq = sigma_y ** 2  # σ²_Ya = 625

        self.rng = np.random.RandomState(seed)

        # Initialize per-action posterior parameters
        # μ_0a = 0 vector, Σ_0a = 100 * I  (from Appendix C.2)
        self.mu = {}  # μ_ta for each action a
        self.Sigma = {}  # Σ_ta for each action a

        for a in range(n_actions):
            self.mu[a] = np.full(state_dim, mu_0)
            self.Sigma[a] = sigma_0 * np.eye(state_dim)

    def select_action(self, v_t):
        """
        Select action using Thompson Sampling.

        1. For each action a, sample θ̂_a ~ N(μ_a, Σ_a)
        2. Compute expected reward θ̂_a^T v_t
        3. Choose action with largest expected reward

        Args:
            v_t: state vector [1, l_t] of shape (state_dim,)

        Returns:
            action: selected action (0, 1, 2, or 3)
        """
        best_action = 0
        best_value = -np.inf

        for a in range(self.n_actions):
            # Sample θ̂_a from posterior N(μ_a, Σ_a)
            theta_hat = self.rng.multivariate_normal(self.mu[a], self.Sigma[a])
            # Compute expected reward
            value = theta_hat @ v_t

            if value > best_value:
                best_value = value
                best_action = a

        return best_action

    def update(self, v_t, action, reward):
        """
        Update posterior for the selected action using Bayesian inference.
        Equations (6) and (7) from the paper.

        Σ_{t+1,a} = σ²_Ya * (v_t v_t^T + σ²_Ya Σ_ta^{-1})^{-1}
        μ_{t+1,a} = Σ_{t+1,a} * ((σ²_Ya)^{-1} r_t v_t + Σ_ta^{-1} μ_ta)

        Args:
            v_t: state vector used at decision time
            action: the action that was taken (ã_t, after LLM filtering)
            reward: observed reward r_t
        """
        a = action

        # Current posterior
        Sigma_a = self.Sigma[a]
        mu_a = self.mu[a]

        # Compute new covariance: Eq. (6)
        # Σ_{t+1,a} = σ²_Ya * (v_t v_t^T + σ²_Ya Σ_ta^{-1})^{-1}
        Sigma_a_inv = np.linalg.inv(Sigma_a)
        vvT = np.outer(v_t, v_t)
        new_Sigma = self.sigma_y_sq * np.linalg.inv(vvT + self.sigma_y_sq * Sigma_a_inv)

        # Compute new mean: Eq. (7)
        # μ_{t+1,a} = Σ_{t+1,a} * ((σ²_Ya)^{-1} r_t v_t + Σ_ta^{-1} μ_ta)
        new_mu = new_Sigma @ ((1.0 / self.sigma_y_sq) * reward * v_t + Sigma_a_inv @ mu_a)

        self.Sigma[a] = new_Sigma
        self.mu[a] = new_mu

    def reset(self, mu_0=0.0, sigma_0=100.0):
        """Reset the agent's posterior to the prior."""
        for a in range(self.n_actions):
            self.mu[a] = np.full(self.state_dim, mu_0)
            self.Sigma[a] = sigma_0 * np.eye(self.state_dim)
