import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
RESULTS_DIR = r"c:\Users\Hritesh\Downloads\RL_files\results"
JSON_PATH = os.path.join(RESULTS_DIR, "experiment_results.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "experiment_comparison_plot.png")

def generate_comparison_plot():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Scenarios for plotting
    scenarios = data["scenario_results"]
    n_scenarios = len(scenarios)
    
    # Get all unique agents
    all_agents = set()
    for s in scenarios:
        all_agents.update(s["agents"].keys())
    
    agent_names = sorted(list(all_agents))
    
    # Setup data for bar chart
    x = np.arange(n_scenarios)
    width = 0.2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [3, 1]})
    
    # Colors for the agents - Professional palette
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # Plot 1: Cumulative Rewards across scenarios
    for i, agent in enumerate(agent_names):
        medians = []
        for s in scenarios:
            medians.append(s["agents"].get(agent, {}).get("median", 0))
        
        offset = (i - (len(agent_names)-1)/2) * width
        ax1.bar(x + offset, medians, width, label=agent, color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_xlabel('Scenario (Behavioral Complexity)', fontsize=12)
    ax1.set_ylabel('Median Cumulative Reward', fontsize=12)
    ax1.set_title('Comparative RL Performance across 4 Scenarios', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S{i+1}" for i in range(n_scenarios)])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 2: LLM Accuracy Summary
    metrics = ["Overall", "Cannot Walk", "Other"]
    vals = [data["llm_validation_accuracy"], data["cannot_walk_accuracy"], data["other_accuracy"]]
    
    bars = ax2.bar(metrics, vals, color=['#7b1fa2', '#388e3c', '#fbc02d'], alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Accuracy Score', fontsize=12)
    ax2.set_title('LLM Judge Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Scientific plot saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_comparison_plot()
