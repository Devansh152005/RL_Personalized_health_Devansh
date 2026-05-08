"""
Interactive LLM + Thompson Sampling Action Recommender

Run this script to interactively input a user preference text and get the
recommended action from the LLM+TS hybrid agent.

Usage:
    python interactive_llm_ts.py                      # Interactive mode (Groq)
    python interactive_llm_ts.py --backend simulated   # Use simulated LLM
    python interactive_llm_ts.py --backend ollama      # Use local Ollama

Actions:
    0: Do not send a message
    1: Send a generic walking message
    2: Send a message tailored to context 0
    3: Send a message tailored to context 1
"""

import sys
import os
import argparse
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from llm_ts_agent import LLMTSAgent
from thompson_sampling import ThompsonSamplingAgent
from llm_judge import LLMJudge

# ── Action descriptions ──────────────────────────────────────────────────────

ACTION_NAMES = {
    0: "Do NOT send a message",
    1: "Send a GENERIC walking encouragement message",
    2: "Send a message tailored to Context 0 (e.g., sedentary state)",
    3: "Send a message tailored to Context 1 (e.g., active state)",
}


def print_banner():
    """Print a nice startup banner."""
    print()
    print("=" * 65)
    print("   LLM + Thompson Sampling  —  Action Recommender")
    print("=" * 65)
    print()
    print("  Type a user preference / health state and the agent will")
    print("  decide what action to take (send or not send a message).")
    print()
    print("  Actions:")
    for a_id, desc in ACTION_NAMES.items():
        print(f"    {a_id}: {desc}")
    print()
    print("  Commands:  'quit' or 'exit' to stop,  'reset' to reset TS")
    print("=" * 65)
    print()


def build_agent(args):
    """Build the LLM+TS agent from CLI arguments."""
    ts_kwargs = {
        "n_actions": 4,
        "state_dim": 2,
        "mu_0": 0.0,
        "sigma_0": 100.0,
        "sigma_y": 25.0,
        "seed": args.seed,
    }

    llm_kwargs = {
        "backend": args.backend,
        "model": args.model,
        "temperature": args.temperature,
    }

    if args.backend == "groq":
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "groq_key.txt")
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                llm_kwargs["api_key"] = f.read().strip()
        else:
            print("[ERROR] groq_key.txt not found. Provide a Groq API key or use --backend simulated")
            sys.exit(1)

    elif args.backend == "ollama":
        llm_kwargs["ollama_url"] = args.ollama_url

    agent = LLMTSAgent(ts_kwargs=ts_kwargs, llm_kwargs=llm_kwargs)
    print(f"[✓] Agent initialized  (backend={args.backend}, model={args.model})")
    return agent


def query_agent(agent, preference_text, inferred_context=None, rng=None):
    """
    Query the LLM+TS agent with a preference text.

    Args:
        agent: LLMTSAgent instance
        preference_text: user health/preference string
        inferred_context: override inferred context l_t (0 or 1). If None, sampled randomly.
        rng: numpy RandomState for context sampling

    Returns:
        dict with action, action_name, candidate_action, llm_decision, llm_reason, state_vector
    """
    if rng is None:
        rng = np.random.RandomState()

    # Build state vector v_t = [1, l_t]
    if inferred_context is not None:
        l_t = int(inferred_context)
    else:
        l_t = rng.randint(0, 2)  # random context for demo

    v_t = np.array([1.0, float(l_t)])

    # Query agent
    final_action, info = agent.select_action(v_t, user_preference=preference_text)

    result = {
        "action": final_action,
        "action_name": ACTION_NAMES[final_action],
        "inferred_context": l_t,
        "state_vector": v_t.tolist(),
        "candidate_action_from_ts": info["candidate_action"],
        "candidate_action_name": ACTION_NAMES[info["candidate_action"]],
        "llm_called": info["llm_called"],
        "llm_decision": info["llm_decision"],
        "llm_reason": info["llm_reason"],
    }

    return result


def print_result(result):
    """Pretty-print the query result."""
    print()
    print("─" * 55)
    print(f"  Inferred Context (l_t):  {result['inferred_context']}")
    print(f"  State Vector (v_t):      {result['state_vector']}")
    print()

    # TS candidate
    ts_act = result["candidate_action_from_ts"]
    print(f"  TS Candidate Action:     {ts_act} → {result['candidate_action_name']}")

    # LLM filter
    if result["llm_called"]:
        print(f"  LLM Called:              Yes")
        print(f"  LLM Decision:            {result['llm_decision']}")
        print(f"  LLM Reason:")
        # Indent the reason text
        for line in result["llm_reason"].split("\n"):
            print(f"    | {line}")
    else:
        reason = "TS chose 'no message' (action 0) — LLM not needed"
        if result["candidate_action_from_ts"] == 0:
            reason = "TS chose 'no message' (action 0) — LLM not needed"
        else:
            reason = "No preference text provided — LLM skipped"
        print(f"  LLM Called:              No  ({reason})")

    # Final action
    print()
    final = result["action"]
    overridden = (ts_act != final)
    override_str = "  ⚠ OVERRIDDEN by LLM!" if overridden else ""
    print(f"  ══ FINAL ACTION: {final} → {result['action_name']}{override_str}")
    print("─" * 55)
    print()


def interactive_loop(agent, args):
    """Run the interactive input loop."""
    rng = np.random.RandomState(args.seed)
    context = args.context  # fixed context or None for random

    print_banner()

    step_count = 0

    while True:
        try:
            user_input = input("📝 Enter preference text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            print("  (empty input — type something or 'quit' to exit)\n")
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("  [✓] Agent reset. TS posteriors are back to priors.\n")
            continue

        if user_input.lower() == "stats":
            print(f"  LLM calls:     {agent.llm_calls}")
            print(f"  LLM overrides: {agent.llm_overrides}")
            print(f"  Steps:         {step_count}\n")
            continue

        # Query the agent
        result = query_agent(agent, user_input, inferred_context=context, rng=rng)
        print_result(result)

        # Optionally simulate a reward update (so TS learns over time)
        # Using a simple heuristic reward for demo purposes
        if args.learn:
            # Simulate reward: action 0 gets small reward, action 1-3 get moderate reward
            if result["action"] == 0:
                reward = 0.1
            else:
                reward = 50.0 + rng.normal(0, 10)
            v_t = np.array(result["state_vector"])
            agent.update(v_t, result["action"], reward)
            print(f"  [TS updated with reward={reward:.1f}]")

        step_count += 1


def single_query(agent, preference_text, args):
    """Run a single query (non-interactive mode)."""
    rng = np.random.RandomState(args.seed)
    result = query_agent(agent, preference_text, inferred_context=args.context, rng=rng)
    print_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive LLM+TS Action Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_llm_ts.py
  python interactive_llm_ts.py --backend simulated
  python interactive_llm_ts.py --query "I am feeling tired"
  python interactive_llm_ts.py --query "I feel great" --context 1
  python interactive_llm_ts.py --backend groq --model llama-3.1-70b-versatile
  python interactive_llm_ts.py --learn  # TS updates after each query
        """
    )

    parser.add_argument("--backend", choices=["groq", "ollama", "simulated"],
                        default="groq", help="LLM backend to use (default: groq)")
    parser.add_argument("--model", default="llama-3.1-8b-instant",
                        help="Model name (default: llama-3.1-8b-instant)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM temperature (default: 0.2)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for TS (default: 42)")
    parser.add_argument("--context", type=int, choices=[0, 1], default=None,
                        help="Fix inferred context l_t (0 or 1). If omitted, sampled randomly.")
    parser.add_argument("--learn", action="store_true",
                        help="Update TS posteriors after each query (agent learns over time)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Single preference text to query (non-interactive mode)")

    args = parser.parse_args()

    # Build agent
    agent = build_agent(args)

    if args.query:
        # Single query mode
        single_query(agent, args.query, args)
    else:
        # Interactive mode
        interactive_loop(agent, args)


if __name__ == "__main__":
    main()
