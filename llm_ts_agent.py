"""
LLM+TS Hybrid Agent

Combines Thompson Sampling with LLM-based action filtering.
Implements the method described in Section 3.1 of the paper.

The hybrid action selection works as follows:
1. TS proposes candidate action a_t
2. If a_t = 0: use ã_t = 0 (no message, no LLM needed)
3. If a_t > 0 AND user preference exists: query LLM
   - If LLM says "not send": ã_t = 0
   - If LLM says "send": ã_t = a_t
4. Update TS with (v_t, ã_t, r_t)
"""

from thompson_sampling import ThompsonSamplingAgent
from llm_judge import LLMJudge


class LLMTSAgent:
    """
    LLM+TS hybrid agent.

    Args:
        ts_kwargs: keyword arguments for ThompsonSamplingAgent
        llm_kwargs: keyword arguments for LLMJudge
    """

    def __init__(self, ts_kwargs=None, llm_kwargs=None):
        ts_kwargs = ts_kwargs or {}
        llm_kwargs = llm_kwargs or {}

        self.ts_agent = ThompsonSamplingAgent(**ts_kwargs)
        self.llm_judge = LLMJudge(**llm_kwargs)

        # Tracking
        self.llm_calls = 0
        self.llm_overrides = 0  # times LLM changed action from >0 to 0

    def select_action(self, v_t, user_preference=None):
        """
        Select action using LLM+TS method.

        Args:
            v_t: state vector for TS
            user_preference: text-based user preference (or None)

        Returns:
            action: final action ã_t
            info: dict with TS candidate action, LLM decision, etc.
        """
        # Step 1: TS proposes candidate action
        candidate_action = self.ts_agent.select_action(v_t)

        info = {
            "candidate_action": candidate_action,
            "llm_called": False,
            "llm_decision": None,
            "llm_reason": None,
        }

        # Step 2: If candidate is "no message" or no preference, skip LLM
        if candidate_action == 0 or user_preference is None:
            return candidate_action, info

        # Step 3: Query LLM
        self.llm_calls += 1
        info["llm_called"] = True

        decision, reason = self.llm_judge.decide(user_preference)
        info["llm_decision"] = decision
        info["llm_reason"] = reason

        # Step 4: Apply LLM filter
        if decision == "not send":
            self.llm_overrides += 1
            return 0, info  # Override to "no message"
        else:
            return candidate_action, info  # Keep TS action

    def update(self, v_t, action, reward):
        """Update the TS posterior with the executed action."""
        self.ts_agent.update(v_t, action, reward)

    def reset(self):
        """Reset the agent for a new trial."""
        self.ts_agent.reset()
        self.llm_calls = 0
        self.llm_overrides = 0
