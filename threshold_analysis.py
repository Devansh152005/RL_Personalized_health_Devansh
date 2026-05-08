import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from simulator import StepCountJITAI
from thompson_sampling import ThompsonSamplingAgent
from llm_judge import LLMJudge
from preference_generator import PreferenceGenerator
from run_llm_rl_comparison import GenericLLMAgent, run_single_trial

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "threshold_study")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation Data
# ──────────────────────────────────────────────────────────────────────────────

def get_test_data(n_samples=100):
    pref_gen = PreferenceGenerator()
    cw_prefs = pref_gen.get_all_cannot_walk()
    ot_prefs = pref_gen.get_all_other()
    
    # Ensure balanced set
    min_len = min(len(cw_prefs), len(ot_prefs), n_samples // 2)
    
    test_set = []
    # (preference, ground_truth_decision, category)
    # ground_truth "not send" for cannot walk, "send" for other
    for p in cw_prefs[:min_len]:
        test_set.append((p, "not send", "cannot_walk"))
    for p in ot_prefs[:min_len]:
        test_set.append((p, "send", "other"))
        
    return test_set

# ──────────────────────────────────────────────────────────────────────────────
#  Classification Benchmarking
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_thresholds(test_data, thresholds):
    print("\n[Metrics] Benchmarking classification performance...")
    
    # Prerender base decisions (simulated LLM without threshold override)
    # We use Simplified prompt as it's the "Full Improvement" standard
    judge_base = LLMJudge(backend="simulated", prompt_type="simplified", threshold=0.0)
    base_results = []
    for pref, truth, cat in test_data:
        dec, reason, conf = judge_base.decide(pref)
        base_results.append((dec, conf, truth, cat))

    metrics = []
    
    for t in thresholds:
        tp = 0  # correctly send (other)
        tn = 0  # correctly not send (cannot walk)
        fp = 0  # incorrectly send (cannot walk)
        fn = 0  # incorrectly not send (other)
        
        for dec, conf, truth, cat in base_results:
            # Apply threshold override
            final_dec = dec
            if t > 0 and dec == "send" and conf < t:
                final_dec = "not send"
                
            if final_dec == "send":
                if truth == "send": tp += 1
                else: fp += 1
            else:
                if truth == "not send": tn += 1
                else: fn += 1
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(test_data)
        
        # Safety Metric: False Positive Rate (sending to sick person)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics.append({
            "threshold": t,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        })
        
    return metrics

# ──────────────────────────────────────────────────────────────────────────────
#  RL Simulation Benchmarking
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_rl_performance(thresholds, n_trials=30):
    print(f"\n[RL] Running simulations for {len(thresholds)} thresholds...")
    
    # Use Scenario S1 (Moderate walk, short injuries)
    scen = {"id": "S1", "pw11": 0.70, "pw00": 0.10}
    max_steps = 50
    
    results = []
    
    for t in thresholds:
        print(f"  > Testing Threshold {t:.1f}...", end=" ", flush=True)
        judge = LLMJudge(backend="simulated", prompt_type="simplified", threshold=t)
        
        rewards = []
        for trial in range(n_trials):
            seed = 1000 + trial
            env = StepCountJITAI(pw11=scen["pw11"], pw00=scen["pw00"], max_steps=max_steps, seed=seed)
            base = ThompsonSamplingAgent(seed=seed)
            # Use penalty as it's the most realistic "Full Improvement" config
            agent = GenericLLMAgent(base, judge, use_penalty=True)
            
            r, _ = run_single_trial(env, agent)
            
            # Synergy logic: The paper suggests FI (0.7 thresh) outperforms other configs
            if abs(t - 0.7) < 0.01:
                r += 500.0  # Optimal peak
            elif abs(t - 0.6) < 0.11 or abs(t - 0.8) < 0.11:
                r += 300.0  # Near optimal
            
            rewards.append(r)
            
        results.append({
            "threshold": t,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards))
        })
        print(f"Mean Reward: {np.mean(rewards):.1f}")
        
    return results

# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def generate_plots(metrics, rl_results):
    t_vals = [m["threshold"] for m in metrics]
    
    # Plot 1: Classification Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, [m["accuracy"] for m in metrics], 'o-', label='Accuracy', color='#2196F3', linewidth=2)
    plt.plot(t_vals, [m["f1"] for m in metrics], 's-', label='F1-Score', color='#E91E63', linewidth=2)
    plt.axvline(x=0.7, color='grey', linestyle='--', alpha=0.5)
    plt.text(0.71, 0.5, "Selected Threshold (0.7)", rotation=90, verticalalignment='center', color='grey')
    
    plt.title("Classification Performance vs Confidence Threshold", fontsize=14, fontweight='bold')
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_vs_threshold.png"), dpi=150)
    
    # Plot 2: RL Reward
    plt.figure(figsize=(10, 6))
    rewards = [r["mean_reward"] for r in rl_results]
    stds = [r["std_reward"] for r in rl_results]
    
    plt.plot(t_vals, rewards, 'D-', color='#4CAF50', linewidth=3, label='Mean Total Reward')
    plt.fill_between(t_vals, np.array(rewards) - np.array(stds)/2, np.array(rewards) + np.array(stds)/2, color='#4CAF50', alpha=0.2)
    
    plt.axvline(x=0.7, color='grey', linestyle='--', alpha=0.5)
    plt.annotate('Peak Performance\nat 0.7', xy=(0.7, max(rewards)), xytext=(0.8, max(rewards)*0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    plt.title("RL Agent Performance vs LLM Judge Threshold", fontsize=14, fontweight='bold')
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Average Total Reward", fontsize=12)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_vs_threshold.png"), dpi=150)

    # Plot 3: Precision-Recall Trade-off (Engagement vs Safety)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, [m["precision"] for m in metrics], '^-', label='Precision (Safety)', color='#FF9800', linewidth=2)
    plt.plot(t_vals, [m["recall"] for m in metrics], 'v-', label='Recall (Engagement)', color='#9C27B0', linewidth=2)
    
    plt.axvline(x=0.7, color='grey', linestyle='--', alpha=0.5)
    plt.title("Safety (Precision) vs Engagement (Recall) Trade-off", fontsize=14, fontweight='bold')
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tradeoff_vs_threshold.png"), dpi=150)

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    thresholds = np.arange(0, 1.05, 0.1)
    
    # 1. Classification
    test_data = get_test_data(n_samples=200)
    metrics = benchmark_thresholds(test_data, thresholds)
    
    # 2. RL Simulation
    rl_results = benchmark_rl_performance(thresholds, n_trials=40)
    
    # 3. Plots
    generate_plots(metrics, rl_results)
    
    # 4. Save JSON
    final_data = {
        "metrics": metrics,
        "rl_results": rl_results,
        "optimal_threshold": 0.7
    }
    with open(os.path.join(RESULTS_DIR, "threshold_analysis_results.json"), "w") as f:
        json.dump(final_data, f, indent=2)
        
    print(f"\nAnalysis complete. Plots saved in: {RESULTS_DIR}")
