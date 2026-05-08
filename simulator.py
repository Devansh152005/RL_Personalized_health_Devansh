"""
StepCountJITAI Simulation Environment (Extended with LLM support)

Replicates the simulation environment from:
  "Using LLMs to improve RL policies in personalized health adaptive interventions"
  Karine & Marlin, CL4Health Workshop @ ACL 2025

Implements:
  - Base behavioral dynamics (Equations 1-5)
  - Extended dynamics with walk-state constraints (Equations 10-14)
  - Markov chain for walk state W (Figure 4, Table 3)
"""

import numpy as np
from preference_generator import PreferenceGenerator


class StepCountJITAI:
    """
    StepCountJITAI simulation environment with LLM support.

    State variables:
        c_t: true context {0, 1}
        p_t: probability of context 1 [0, 1]
        l_t: inferred context {0, 1}
        h_t: habituation level [0, 1]
        d_t: disengagement risk level [0, 1]
        s_t: step count (reward)
        w_t: walk state {0="cannot walk", 1="can walk"}

    Actions:
        0: do not send a message
        1: send a generic message
        2: send a message tailored to context 0
        3: send a message tailored to context 1

    Parameters (from paper Appendix A.1.1):
        sigma:  context uncertainty = 0.4
        delta_h: habituation decay = 0.1
        epsilon_h: habituation increment = 0.05
        delta_d: disengagement decay = 0.1
        epsilon_d: disengagement increment = 0.4
        m_s: baseline step count = 0.1
        rho_1: generic message reward boost = 50
        rho_2: tailored message reward boost = 200
        D_threshold: disengagement threshold = 0.99
        eta_d: "cannot walk" constraint penalty ∈ [0, 1]

    Walk state Markov chain parameters:
        pw11: P(w_{t+1}=1 | w_t=1) - probability of staying in "can walk"
        pw01: P(w_{t+1}=1 | w_t=0) - probability of transitioning to "can walk"
        pw00: 1 - pw01 - probability of staying in "cannot walk"
    """

    def __init__(
        self,
        # Base simulator parameters (Appendix A.1.1)
        sigma=0.4,
        delta_h=0.1,
        epsilon_h=0.05,
        delta_d=0.1,
        epsilon_d=0.4,
        m_s=0.1,
        rho_1=50,
        rho_2=200,
        D_threshold=0.99,
        # Walk state constraint parameter
        eta_d=0.4,
        # Walk state Markov chain parameters
        pw11=0.7,
        pw00=0.1,
        # Experiment settings
        max_steps=50,
        p_other_preference=0.3,
        seed=None,
    ):
        # Base parameters
        self.sigma = sigma
        self.delta_h = delta_h
        self.epsilon_h = epsilon_h
        self.delta_d = delta_d
        self.epsilon_d = epsilon_d
        self.m_s = m_s
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.D_threshold = D_threshold

        # Walk state parameters
        self.eta_d = eta_d
        self.pw11 = pw11
        self.pw00 = pw00
        self.pw01 = 1.0 - pw00  # P(w_{t+1}=1 | w_t=0)

        # Experiment settings
        self.max_steps = max_steps
        self.p_other_preference = p_other_preference

        # Preference generator
        self.pref_gen = PreferenceGenerator()

        # Random state
        self.rng = np.random.RandomState(seed)

        # State variables (initialized in reset)
        self.c_t = None  # true context
        self.p_t = None  # probability of context 1
        self.l_t = None  # inferred context
        self.h_t = None  # habituation
        self.d_t = None  # disengagement risk
        self.s_t = None  # step count
        self.w_t = None  # walk state
        self.t = None  # current time step
        self.done = None  # episode done flag
        self.user_preference = None  # current text preference

    def reset(self):
        """Reset environment to initial state. Returns (state_vector, user_preference)."""
        self.t = 0
        self.h_t = 0.0
        self.d_t = 0.0
        self.w_t = 1  # start in "can walk" state
        self.done = False
        self.user_preference = None

        # Sample initial context (Eq. 1)
        self.c_t = self.rng.binomial(1, 0.5)
        x_t = self.rng.normal(self.c_t, self.sigma)

        # Infer context (Eq. 2) - using Bayesian inference with Gaussian likelihood
        # P(C=1|x) ∝ P(x|C=1) P(C=1) = N(x;1,σ²) * 0.5
        # P(C=0|x) ∝ P(x|C=0) P(C=0) = N(x;0,σ²) * 0.5
        # P(C=1|x) = sigmoid((2x - 1) / (2σ²))
        # Simplified: p_t = P(C=1|x_t) using Bayes' rule
        from scipy.stats import norm
        likelihood_1 = norm.pdf(x_t, loc=1, scale=self.sigma)
        likelihood_0 = norm.pdf(x_t, loc=0, scale=self.sigma)
        self.p_t = likelihood_1 / (likelihood_1 + likelihood_0)
        self.l_t = int(self.p_t > 0.5)

        # Initial step count = 0
        self.s_t = 0.0

        return self._get_state_vector(), self.user_preference

    def _get_state_vector(self):
        """
        Return the state vector v_t used by the TS agent.
        v_t = [1, l_t] (intercept + inferred context)
        Note: The RL agent does NOT observe w_t.
        """
        return np.array([1.0, float(self.l_t)])

    def _update_walk_state(self):
        """
        Update walk state w_t using Markov chain (Figure 4, Table 3).
        Returns the previous walk state for preference generation logic.
        """
        w_prev = self.w_t
        if self.w_t == 1:
            # Currently "can walk"
            self.w_t = self.rng.binomial(1, self.pw11)
        else:
            # Currently "cannot walk"
            self.w_t = self.rng.binomial(1, self.pw01)
        return w_prev

    def _generate_preference(self, w_prev):
        """
        Generate text-based user preference based on walk state transitions.
        See Appendix B.3.
        """
        if w_prev == 1 and self.w_t == 0:
            # Transition to "cannot walk": emit "cannot walk" preference
            self.user_preference = self.pref_gen.get_cannot_walk_preference(self.rng)
        elif w_prev == 0 and self.w_t == 1:
            # Transition to "can walk": emit "other" preference
            self.user_preference = self.pref_gen.get_other_preference(self.rng)
        elif self.w_t == 1:
            # Staying in "can walk": emit "other" with prob 0.3
            if self.rng.random() < self.p_other_preference:
                self.user_preference = self.pref_gen.get_other_preference(self.rng)
            else:
                self.user_preference = None
        else:
            # Staying in "cannot walk": keep the previous preference or generate new one
            self.user_preference = self.pref_gen.get_cannot_walk_preference(self.rng)

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: the final action ã_t (after LLM filtering if applicable)

        Returns:
            state_vector: v_{t+1} for the TS agent
            reward: r_t (step count)
            done: whether the episode ended
            info: dict with additional info (true_context, walk_state, preference, etc.)
        """
        assert not self.done, "Episode is done. Call reset()."
        assert action in [0, 1, 2, 3], f"Invalid action: {action}"

        a_t = action
        c_t = self.c_t
        w_t = self.w_t

        # --- Update habituation (Eq. 12) ---
        if a_t == 0:
            h_new = (1 - self.delta_h) * self.h_t
        else:
            h_new = min(1.0, self.h_t + self.epsilon_h)

        # --- Update disengagement risk (Eq. 13) ---
        if a_t == 0:
            # No message sent: disengagement stays the same
            d_new = self.d_t
        elif a_t in [1, c_t + 2]:
            # Correct message (generic or correctly tailored)
            if w_t == 1:
                # Can walk: decrement disengagement
                d_new = (1 - self.delta_d) * self.d_t
            else:
                # Cannot walk: penalty even for "correct" message
                d_new = min(1.0, self.d_t + self.eta_d)
        else:
            # Incorrect message (wrong tailoring)
            d_new = min(1.0, self.d_t + self.epsilon_d + (1 - w_t) * self.eta_d)

        # --- Compute reward / step count (Eq. 14) ---
        if a_t == 1 and w_t == 1:
            # Generic message, can walk
            reward = self.m_s + (1 - h_new) * self.rho_1
        elif a_t == c_t + 2 and w_t == 1:
            # Correctly tailored message, can walk
            reward = self.m_s + (1 - h_new) * self.rho_2
        else:
            # No message, wrong message, or cannot walk
            reward = self.m_s * w_t  # m_s * w_t: 0 if cannot walk, m_s if can walk but no/wrong msg

        # --- Update context for next time step (Eq. 10-11) ---
        c_new = self.rng.binomial(1, 0.5)
        x_new = self.rng.normal(c_new, self.sigma)
        from scipy.stats import norm
        likelihood_1 = norm.pdf(x_new, loc=1, scale=self.sigma)
        likelihood_0 = norm.pdf(x_new, loc=0, scale=self.sigma)
        p_new = likelihood_1 / (likelihood_1 + likelihood_0)
        l_new = int(p_new > 0.5)

        # --- Update walk state and generate preference ---
        w_prev = self.w_t
        self._update_walk_state()  # updates self.w_t
        self._generate_preference(w_prev)

        # --- Check disengagement threshold ---
        self.t += 1
        if d_new >= self.D_threshold or self.t >= self.max_steps:
            self.done = True

        # --- Update state ---
        self.h_t = h_new
        self.d_t = d_new
        self.c_t = c_new
        self.p_t = p_new
        self.l_t = l_new

        info = {
            "true_context": c_t,
            "walk_state": w_t,
            "new_walk_state": self.w_t,
            "disengagement": d_new,
            "habituation": h_new,
            "time_step": self.t,
            "user_preference": self.user_preference,
            "disengaged": d_new >= self.D_threshold,
        }

        return self._get_state_vector(), reward, self.done, info
