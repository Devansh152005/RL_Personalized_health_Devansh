import os
import sys
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from simulator import StepCountJITAI
from thompson_sampling import ThompsonSamplingAgent

class LinUCBAgent:
    """
    Contextual LinUCB agent.
    Shares the exact same tracking mechanics as Thompson Sampling
    but uses upper confidence bounds for action selection instead of posterior sampling.
    """
    def __init__(
        self,
        n_actions=4,
        state_dim=2,
        mu_0=0.0,
        sigma_0=100.0,
        sigma_y=25.0,
        alpha=2.0,  # Exploration parameter (commonly 1, sqrt(2), or tuned)
        seed=None,
    ):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.sigma_y_sq = sigma_y ** 2
        self.alpha = alpha

        self.rng = np.random.RandomState(seed)

        self.mu = {}
        self.Sigma = {}

        for a in range(n_actions):
            self.mu[a] = np.full(state_dim, mu_0)
            self.Sigma[a] = sigma_0 * np.eye(state_dim)

    def select_action(self, v_t):
        best_action = 0
        best_value = -np.inf

        for a in range(self.n_actions):
            expected_reward = self.mu[a] @ v_t
            # Matrix-vector multiplication for variance
            confidence_bound = np.sqrt(v_t @ self.Sigma[a] @ v_t)
            
            value = expected_reward + self.alpha * confidence_bound

            if value > best_value:
                best_value = value
                best_action = a

        return best_action
        
    def update(self, v_t, action, reward):
        a = action
        Sigma_a = self.Sigma[a]
        mu_a = self.mu[a]

        # Update covariance
        Sigma_a_inv = np.linalg.inv(Sigma_a)
        vvT = np.outer(v_t, v_t)
        new_Sigma = self.sigma_y_sq * np.linalg.inv(vvT + self.sigma_y_sq * Sigma_a_inv)

        # Update mean
        new_mu = new_Sigma @ ((1.0 / self.sigma_y_sq) * reward * v_t + Sigma_a_inv @ mu_a)

        self.Sigma[a] = new_Sigma
        self.mu[a] = new_mu

    def reset(self, mu_0=0.0, sigma_0=100.0):
        for a in range(self.n_actions):
            self.mu[a] = np.full(self.state_dim, mu_0)
            self.Sigma[a] = sigma_0 * np.eye(self.state_dim)

def run_single_trial(env, agent):
    v_t, user_preference = env.reset()
    total_reward = 0.0

    while not env.done:
        action = agent.select_action(v_t)
        v_next, reward, done, info = env.step(action)
        agent.update(v_t, action, reward)
        total_reward += reward
        v_t = v_next

    return total_reward

def run_isolated_comparison():
    print("="*60)
    print("  ISOLATED EXPERIMENT: Thompson Sampling vs LinUCB")
    print("="*60)
    
    scenarios = [
        {"pw11": 0.7,  "pw00": 0.1, "epsilon_d": 0.05, "eta_d": 0.4},
        {"pw11": 0.7,  "pw00": 0.5, "epsilon_d": 0.05, "eta_d": 0.4},
        {"pw11": 0.95, "pw00": 0.1, "epsilon_d": 0.05, "eta_d": 0.4},
        {"pw11": 0.95, "pw00": 0.5, "epsilon_d": 0.05, "eta_d": 0.4},
    ]
    
    n_trials = 5
    max_steps = 50
    seed_base = 42

    for scenario in scenarios:
        pw11 = scenario["pw11"]
        pw00 = scenario["pw00"]
        print(f"\n  Scenario: pw11={pw11}, pw00={pw00}")
        
        ts_rewards = []
        ucb_rewards = []
        
        for trial in range(n_trials):
            seed = seed_base + trial
            
            # Thompson Sampling
            env_ts = StepCountJITAI(**scenario, max_steps=max_steps, seed=seed)
            agent_ts = ThompsonSamplingAgent(seed=seed)
            ts_rewards.append(run_single_trial(env_ts, agent_ts))
            
            # LinUCB
            env_ucb = StepCountJITAI(**scenario, max_steps=max_steps, seed=seed)
            # using alpha=1.0 for standard UCB confidence bound
            agent_ucb = LinUCBAgent(seed=seed, alpha=1.0)
            ucb_rewards.append(run_single_trial(env_ucb, agent_ucb))
            
        print(f"    Standard TS median reward: {np.median(ts_rewards):.1f}")
        print(f"    LinUCB median reward:      {np.median(ucb_rewards):.1f}")

if __name__ == "__main__":
    run_isolated_comparison()
