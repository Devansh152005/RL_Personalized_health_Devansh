"""
Plotting utilities for reproducing paper figures.

Generates:
- Figure 2: Box plots of total reward (LLM+TS vs standard TS)
- Figure 3: Action histograms + cumulative reward plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11


def ensure_results_dir(base_dir):
    """Create results directory if it doesn't exist."""
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def plot_figure2(results_list, base_dir):
    """
    Reproduce Figure 2: Box plots of total reward.

    Args:
        results_list: list of dicts, each with keys:
            - "pw11", "pw00": scenario parameters
            - "epsilon_d", "eta_d": disengagement parameters
            - "llm_ts_rewards": list of total rewards for LLM+TS (over 5 trials)
            - "ts_rewards": list of total rewards for standard TS (over 5 trials)
        base_dir: project base directory
    """
    results_dir = ensure_results_dir(base_dir)

    # Group by scenario
    scenarios = {}
    for r in results_list:
        key = (r["pw11"], r["pw00"])
        if key not in scenarios:
            scenarios[key] = []
        scenarios[key].append(r)

    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), squeeze=False)

    for idx, ((pw11, pw00), scenario_results) in enumerate(sorted(scenarios.items())):
        ax = axes[0, idx]

        # Use the first result set for this scenario
        r = scenario_results[0]
        llm_ts = r["llm_ts_rewards"]
        ts = r["ts_rewards"]

        # Box plot
        positions = [0, 1]
        bp = ax.boxplot(
            [llm_ts, ts],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='black', markersize=5),
        )

        # Color the boxes
        colors = ['#4CAF50', '#FF7043']  # green for LLM+TS, orange for TS
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(['LLM+TS', 'standard TS'], fontsize=10)
        ax.set_ylabel('Total reward', fontsize=11)
        ax.set_title(
            f'$(p_{{w11}}, p_{{w00}})$ = ({pw11}, {pw00})\n'
            f'D={r.get("D_threshold", 0.99)} '
            f'$\\epsilon_d$={r["epsilon_d"]} '
            f'$\\eta_d$={r["eta_d"]}',
            fontsize=10,
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Figure 2: LLM+TS vs Standard TS - Total Reward Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(results_dir, "figure2_total_reward.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved Figure 2 -> {path}")
    return path


def plot_figure3(trial_data_llm_ts, trial_data_ts, pw11, pw00, epsilon_d, eta_d, base_dir):
    """
    Reproduce Figure 3: Action histograms (top) + cumulative reward plots (bottom).

    Args:
        trial_data_llm_ts: list of dicts per trial, each with "actions" and "rewards" lists
        trial_data_ts: same for standard TS
        pw11, pw00: scenario parameters
        epsilon_d, eta_d: disengagement parameters
        base_dir: project base directory
    """
    results_dir = ensure_results_dir(base_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Top row: Action histograms ---
    # Aggregate actions across all trials
    all_actions_llm_ts = []
    for trial in trial_data_llm_ts:
        all_actions_llm_ts.extend(trial["actions"])

    all_actions_ts = []
    for trial in trial_data_ts:
        all_actions_ts.extend(trial["actions"])

    n_trials = len(trial_data_llm_ts)

    # LLM+TS histogram
    ax = axes[0, 0]
    action_counts = [all_actions_llm_ts.count(a) for a in range(4)]
    bars = ax.bar(range(4), action_counts, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'], alpha=0.8)
    ax.set_xlabel('Selected action', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'LLM+TS\nAction histogram ({n_trials} trials)', fontsize=10)
    ax.set_xticks(range(4))

    # Standard TS histogram
    ax = axes[0, 1]
    action_counts = [all_actions_ts.count(a) for a in range(4)]
    bars = ax.bar(range(4), action_counts, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'], alpha=0.8)
    ax.set_xlabel('Selected action', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'standard TS\nAction histogram ({n_trials} trials)', fontsize=10)
    ax.set_xticks(range(4))

    # --- Bottom row: Cumulative reward per episode ---
    # LLM+TS cumulative rewards
    ax = axes[1, 0]
    for i, trial in enumerate(trial_data_llm_ts):
        cum_rewards = np.cumsum(trial["rewards"])
        ax.plot(range(len(cum_rewards)), cum_rewards, alpha=0.5, linewidth=1)
    # Plot mean
    max_len = max(len(trial["rewards"]) for trial in trial_data_llm_ts)
    padded = np.zeros((n_trials, max_len))
    for i, trial in enumerate(trial_data_llm_ts):
        padded[i, :len(trial["rewards"])] = np.cumsum(trial["rewards"])
        if len(trial["rewards"]) < max_len:
            padded[i, len(trial["rewards"]):] = padded[i, len(trial["rewards"]) - 1]
    mean_cum = padded.mean(axis=0)
    ax.plot(range(max_len), mean_cum, color='black', linewidth=2, label='Mean')
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel('Cumulative reward', fontsize=10)
    ax.set_title('LLM+TS\nCumulative reward', fontsize=10)
    ax.legend()

    # Standard TS cumulative rewards
    ax = axes[1, 1]
    for i, trial in enumerate(trial_data_ts):
        cum_rewards = np.cumsum(trial["rewards"])
        ax.plot(range(len(cum_rewards)), cum_rewards, alpha=0.5, linewidth=1)
    max_len = max(len(trial["rewards"]) for trial in trial_data_ts)
    padded = np.zeros((n_trials, max_len))
    for i, trial in enumerate(trial_data_ts):
        padded[i, :len(trial["rewards"])] = np.cumsum(trial["rewards"])
        if len(trial["rewards"]) < max_len:
            padded[i, len(trial["rewards"]):] = padded[i, len(trial["rewards"]) - 1]
    mean_cum = padded.mean(axis=0)
    ax.plot(range(max_len), mean_cum, color='black', linewidth=2, label='Mean')
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel('Cumulative reward', fontsize=10)
    ax.set_title('standard TS\nCumulative reward', fontsize=10)
    ax.legend()

    plt.suptitle(
        f'Figure 3: LLM+TS vs standard TS - '
        f'$(p_{{w11}}, p_{{w00}})$ = ({pw11}, {pw00}), '
        f'$\\epsilon_d$={epsilon_d}, $\\eta_d$={eta_d}',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    path = os.path.join(results_dir, f"figure3_pw11={pw11}_pw00={pw00}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved Figure 3 -> {path}")
    return path


def plot_llm_validation(accuracies, base_dir):
    """
    Plot LLM validation accuracy results.

    Args:
        accuracies: dict mapping model_name -> accuracy
        base_dir: project base directory
    """
    results_dir = ensure_results_dir(base_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(accuracies.keys())
    accs = list(accuracies.values())

    bars = ax.bar(models, accs, color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{acc:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
        )

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('LLM Validation: Accuracy on "cannot walk" / "other" classification', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "llm_validation_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved LLM validation -> {path}")
    return path
