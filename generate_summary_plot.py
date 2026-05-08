import json
import matplotlib.pyplot as plt
import numpy as np

def generate_impact_plot():
    # Load the results
    with open('results/ablation_full/full_ablation_results.json', 'r') as f:
        data = json.load(f)
    
    agent_names = ["Baseline", "SimpPrompt", "PenaltyOnly", "ThreshOnly", "Simp+Pen", "Simp+Thresh", "Pen+Thresh", "Full Improvement"]
    colors = ['#888888', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#9467bd', '#8c564b', '#2ca02c']
    
    # Calculate mean reward across all scenarios for each agent
    avg_rewards = {name: [] for name in agent_names}
    
    for scenario in data:
        for name in agent_names:
            if name in scenario['agents']:
                # Back to Mean as requested
                avg_rewards[name].append(scenario['agents'][name]['mean'])
    
    final_means = [np.mean(avg_rewards[name]) for name in agent_names]
    # Calculate % improvement over baseline
    baseline_mean = final_means[0]
    improvements = [((m - baseline_mean) / baseline_mean) * 100 for m in final_means]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.barh(agent_names, improvements, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    
    # Add labels and style
    plt.xlabel('Average Reward Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    plt.title('Consolidated Impact of LLM-RL Architectural Enhancements', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Highlight the winner
    for i, bar in enumerate(bars):
        if agent_names[i] == "Full Improvement":
            bar.set_alpha(1.0)
            bar.set_linewidth(2)
        
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2., f'+{width:.1f}%', 
                 va='center', fontweight='bold', fontsize=11, color='darkgreen' if width > 0 else 'red')

    plt.tight_layout()
    plt.savefig('results/ablation_full/final_impact_summary.png', dpi=300)
    print("Consolidated impact plot generated: results/ablation_full/final_impact_summary.png")

if __name__ == "__main__":
    generate_impact_plot()
