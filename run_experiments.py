"""
python src/run_experiments.py --backend groq --model llama-3.1-8b-instant
Main Experiment Runner

Reproduces the core experiments from the paper:
1. LLM Validation (Section C.1): Tests LLM accuracy on classifying preferences
2. LLM+TS vs Standard TS (Figures 2, 3): Compares total rewards and action distributions

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --validate-only     # Only run LLM validation
    python run_experiments.py --simulated         # Use simulated LLM (no API calls)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from simulator import StepCountJITAI
from thompson_sampling import ThompsonSamplingAgent
from llm_judge import LLMJudge
from llm_ts_agent import LLMTSAgent
from preference_generator import PreferenceGenerator
from plotting import plot_figure2, plot_figure3, plot_llm_validation


BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_api_key():
    """Load Groq API key from file."""
    key_path = os.path.join(BASE_DIR, "groq_key.txt")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()
    return None


# ─── Experiment 1: LLM Validation (Section C.1) ─────────────────────────────

def run_llm_validation(model="llama-3.1-8b-instant", backend="groq", api_key=None):
    """
    Test LLM accuracy on classifying user preferences.
    Uses 50 "cannot walk" + 50 "other" preferences.
    
    Expected results from paper:
        Gemma 2: 0.86, Llama 3 8B: 0.87, Llama 3 70B: 0.98
    """
    print(f"\n{'='*60}")
    print(f"  Experiment 1: LLM Validation ({model})")
    print(f"{'='*60}")

    pref_gen = PreferenceGenerator()
    judge = LLMJudge(backend=backend, model=model, api_key=api_key, temperature=0.2)

    # Get preferences
    cannot_walk_prefs = pref_gen.get_all_cannot_walk()[:50]
    other_prefs = pref_gen.get_all_other()[:50]

    # Pad to 50 if needed
    rng = np.random.RandomState(42)
    while len(cannot_walk_prefs) < 50:
        cannot_walk_prefs.append(cannot_walk_prefs[rng.randint(0, len(cannot_walk_prefs))])
    while len(other_prefs) < 50:
        other_prefs.append(other_prefs[rng.randint(0, len(other_prefs))])

    correct = 0
    total = 0

    # Test "cannot walk" preferences (should return "not send")
    print("\nTesting 'cannot walk' preferences...")
    for i, pref in enumerate(cannot_walk_prefs):
        decision, reason = judge.decide(pref)
        is_correct = (decision == "not send")
        correct += int(is_correct)
        total += 1
        status = "OK" if is_correct else "FAIL"
        if not is_correct:
            print(f"  {status} [{i+1}] '{pref}' -> {decision}")

    # Test "other" preferences (should return "send")
    print("\nTesting 'other/healthy' preferences...")
    for i, pref in enumerate(other_prefs):
        decision, reason = judge.decide(pref)
        is_correct = (decision == "send")
        correct += int(is_correct)
        total += 1
        status = "OK" if is_correct else "FAIL"
        if not is_correct:
            print(f"  {status} [{i+1}] '{pref}' -> {decision}")

    accuracy = correct / total
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.2f}")
    print(f"  Paper reference ({model}): ~0.87 (Llama 3 8B)")
    return accuracy


# ─── Experiment 2: LLM+TS vs Standard TS (Figures 2, 3) ─────────────────────

def run_single_trial(env, agent, is_llm_ts=False):
    """
    Run a single trial (one simulated participant study).
    
    Returns:
        total_reward: sum of rewards over the trial
        trial_data: dict with "actions", "rewards", and per-step info
    """
    v_t, user_preference = env.reset()
    total_reward = 0.0
    actions = []
    rewards = []
    step_infos = []

    while not env.done:
        if is_llm_ts:
            action, action_info = agent.select_action(v_t, user_preference)
        else:
            action = agent.select_action(v_t)
            action_info = {"candidate_action": action}

        v_next, reward, done, info = env.step(action)

        # Update agent
        if is_llm_ts:
            agent.update(v_t, action, reward)
        else:
            agent.update(v_t, action, reward)

        actions.append(action)
        rewards.append(reward)
        step_infos.append({**info, **action_info})

        total_reward += reward
        v_t = v_next
        user_preference = info.get("user_preference", None)

    return total_reward, {
        "actions": actions,
        "rewards": rewards,
        "step_infos": step_infos,
        "episode_length": len(actions),
    }


def run_comparison_experiment(
    pw11, pw00, epsilon_d, eta_d,
    n_trials=5, max_steps=50,
    llm_model="llama-3.1-8b-instant", llm_backend="groq", api_key=None,
    seed_base=42,
):
    """
    Run LLM+TS vs Standard TS comparison for a given scenario.

    Args:
        pw11: probability of staying in "can walk"
        pw00: probability of staying in "cannot walk"  
        epsilon_d: disengagement increment for incorrect message
        eta_d: "cannot walk" constraint penalty
        n_trials: number of trials (default 5, as in paper)
        max_steps: maximum study length (default 50 days)
        llm_model: LLM model name
        llm_backend: "groq" or "simulated"
        api_key: Groq API key

    Returns:
        result: dict with rewards and trial data for both methods
    """
    print(f"\n  Scenario: pw11={pw11}, pw00={pw00}, eps_d={epsilon_d}, eta_d={eta_d}")
    print(f"  Running {n_trials} trials...")

    llm_ts_rewards = []
    ts_rewards = []
    llm_ts_trial_data = []
    ts_trial_data = []

    for trial in range(n_trials):
        seed = seed_base + trial
        print(f"    Trial {trial+1}/{n_trials}", end="")

        # --- LLM+TS ---
        env = StepCountJITAI(
            pw11=pw11, pw00=pw00, epsilon_d=epsilon_d, eta_d=eta_d,
            max_steps=max_steps, seed=seed,
        )
        agent = LLMTSAgent(
            ts_kwargs={"seed": seed},
            llm_kwargs={"backend": llm_backend, "model": llm_model, "api_key": api_key},
        )
        total_reward_llm, trial_data = run_single_trial(env, agent, is_llm_ts=True)
        llm_ts_rewards.append(total_reward_llm)
        llm_ts_trial_data.append(trial_data)

        # --- Standard TS ---
        env = StepCountJITAI(
            pw11=pw11, pw00=pw00, epsilon_d=epsilon_d, eta_d=eta_d,
            max_steps=max_steps, seed=seed,
        )
        agent = ThompsonSamplingAgent(seed=seed)
        total_reward_ts, trial_data = run_single_trial(env, agent, is_llm_ts=False)
        ts_rewards.append(total_reward_ts)
        ts_trial_data.append(trial_data)

        print(f"  ->  LLM+TS: {total_reward_llm:.1f}  |  TS: {total_reward_ts:.1f}")

    print(f"\n  Summary:")
    print(f"    LLM+TS median: {np.median(llm_ts_rewards):.1f} "
          f"(Q1={np.percentile(llm_ts_rewards, 25):.1f}, Q3={np.percentile(llm_ts_rewards, 75):.1f})")
    print(f"    Std TS median: {np.median(ts_rewards):.1f} "
          f"(Q1={np.percentile(ts_rewards, 25):.1f}, Q3={np.percentile(ts_rewards, 75):.1f})")

    return {
        "pw11": pw11,
        "pw00": pw00,
        "epsilon_d": epsilon_d,
        "eta_d": eta_d,
        "D_threshold": 0.99,
        "llm_ts_rewards": llm_ts_rewards,
        "ts_rewards": ts_rewards,
        "llm_ts_trial_data": llm_ts_trial_data,
        "ts_trial_data": ts_trial_data,
    }


def run_main_experiments(llm_model, llm_backend, api_key):
    """
    Run the core experiments from the paper (Figures 2 & 3).

    Scenario settings from paper (Section 4, Appendix C.2):
    - Scenario 1: pw11=0.7,  pw00 ∈ {0.1, 0.5}
    - Scenario 2: pw11=0.95, pw00 ∈ {0.1, 0.5}
    - εd=0.05, ηd=0.4, D_threshold=0.99
    """
    print(f"\n{'='*60}")
    print(f"  Experiment 2: LLM+TS vs Standard TS")
    print(f"  Model: {llm_model} ({llm_backend})")
    print(f"{'='*60}")

    # Core scenarios matching Figure 2
    scenarios = [
        {"pw11": 0.7,  "pw00": 0.1, "epsilon_d": 0.05, "eta_d": 0.4},
        {"pw11": 0.7,  "pw00": 0.5, "epsilon_d": 0.05, "eta_d": 0.4},
        {"pw11": 0.95, "pw00": 0.1, "epsilon_d": 0.05, "eta_d": 0.4},
        {"pw11": 0.95, "pw00": 0.5, "epsilon_d": 0.05, "eta_d": 0.4},
    ]

    all_results = []
    for scenario in scenarios:
        result = run_comparison_experiment(
            **scenario,
            n_trials=5,
            llm_model=llm_model,
            llm_backend=llm_backend,
            api_key=api_key,
        )
        all_results.append(result)

    # --- Generate Figure 2: Total reward box plots ---
    print(f"\n{'='*60}")
    print("  Generating plots...")
    print(f"{'='*60}")

    plot_figure2(all_results, BASE_DIR)

    # --- Generate Figure 3: Action histograms + cumulative rewards ---
    # Use scenario (pw11=0.7, pw00=0.5) as in paper's Figure 3
    for result in all_results:
        plot_figure3(
            result["llm_ts_trial_data"],
            result["ts_trial_data"],
            result["pw11"], result["pw00"],
            result["epsilon_d"], result["eta_d"],
            BASE_DIR,
        )

    # Save numerical results
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary = []
    for r in all_results:
        summary.append({
            "pw11": r["pw11"],
            "pw00": r["pw00"],
            "epsilon_d": r["epsilon_d"],
            "eta_d": r["eta_d"],
            "llm_ts_rewards": r["llm_ts_rewards"],
            "ts_rewards": r["ts_rewards"],
            "llm_ts_median": float(np.median(r["llm_ts_rewards"])),
            "ts_median": float(np.median(r["ts_rewards"])),
            "llm_ts_episode_lengths": [d["episode_length"] for d in r["llm_ts_trial_data"]],
            "ts_episode_lengths": [d["episode_length"] for d in r["ts_trial_data"]],
        })

    with open(os.path.join(results_dir, "experiment_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved numerical results -> results/experiment_results.json")

    return all_results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replicate experiments: Using LLMs to improve RL policies"
    )
    parser.add_argument(
        "--backend", type=str, default="ollama",
        choices=["groq", "ollama", "simulated"],
        help="LLM backend (default: ollama)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model name (default: gemma2:9b for ollama, llama-3.1-8b-instant for groq)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run LLM validation experiment"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip LLM validation, only run main experiments"
    )
    args = parser.parse_args()

    # Set default model based on backend
    if args.model is None:
        if args.backend == "ollama":
            args.model = "gemma2:9b"
        else:
            args.model = "llama-3.1-8b-instant"

    api_key = load_api_key() if args.backend == "groq" else None

    print("=" * 60)
    print("  Replication: Using LLMs to improve RL policies")
    print("    in personalized health adaptive interventions")
    print("  Karine & Marlin (CL4Health @ ACL 2025)")
    print("=" * 60)
    print(f"  Backend: {args.backend}")
    print(f"  Model: {args.model}")
    if args.backend == "groq":
        print(f"  API Key: {'loaded' if api_key else 'not found'}")

    start_time = time.time()

    # Experiment 1: LLM Validation
    if not args.skip_validation:
        accuracy = run_llm_validation(
            model=args.model, backend=args.backend, api_key=api_key
        )
        plot_llm_validation({args.model: accuracy}, BASE_DIR)

    if args.validate_only:
        print(f"\n  Done (validation only). Time: {time.time() - start_time:.1f}s")
        return

    # Experiment 2: Main comparison
    all_results = run_main_experiments(args.model, args.backend, api_key)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  All experiments complete. Total time: {elapsed:.1f}s")
    print(f"  Results saved in: {os.path.join(BASE_DIR, 'results')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
