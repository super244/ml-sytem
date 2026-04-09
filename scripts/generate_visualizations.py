#!/usr/bin/env python3
"""Generate visualizations for model evaluation results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_summary(eval_dir):
    """Load evaluation summary JSON."""
    summary_path = Path(eval_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None

def plot_accuracy_comparison(results_dict, output_path):
    """Create bar chart comparing model accuracies."""
    models = list(results_dict.keys())
    primary_acc = [r['primary']['accuracy'] * 100 for r in results_dict.values()]
    secondary_acc = [r['secondary']['accuracy'] * 100 for r in results_dict.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, primary_acc, width, label='Trained Model', color='#2ecc71')
    bars2 = ax.bar(x + width/2, secondary_acc, width, label='Base Model', color='#e74c3c')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved accuracy comparison to {output_path}")

def plot_improvement_delta(results_dict, output_path):
    """Create bar chart showing improvement deltas."""
    models = list(results_dict.keys())
    deltas = [r['delta_accuracy'] * 100 for r in results_dict.values()]
    
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, deltas, color=colors, alpha=0.8)
    
    ax.set_xlabel('Accuracy Improvement (%)', fontsize=12)
    ax.set_title('Model Improvement Over Base', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        width = bar.get_width()
        ax.annotate(f'{delta:+.1f}%',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5 if width > 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if width > 0 else 'right',
                   va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved improvement delta to {output_path}")

def plot_error_breakdown(results_dict, output_path):
    """Create error breakdown chart."""
    models = list(results_dict.keys())
    n_models = len(models)
    
    error_types = ['wrong_final_answer', 'reasoning_off_track', 'formatting_failure']
    error_labels = ['Wrong Answer', 'Reasoning Error', 'Formatting Error']
    
    # Create subplots - use 2 columns or single column based on count
    if n_models == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    elif n_models == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        cols = 2
        rows = (n_models + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        axes = axes.flatten() if n_models > 2 else [axes]
    
    for idx, (model, results) in enumerate(results_dict.items()):
        ax = axes[idx] if n_models > 1 else axes[0]
        
        primary_errors = results['primary']['error_types']
        secondary_errors = results['secondary']['error_types']
        
        primary_counts = [primary_errors.get(e, 0) for e in error_types]
        secondary_counts = [secondary_errors.get(e, 0) for e in error_types]
        
        x = np.arange(len(error_types))
        width = 0.35
        
        ax.bar(x - width/2, primary_counts, width, label='Trained', color='#3498db')
        ax.bar(x + width/2, secondary_counts, width, label='Base', color='#e74c3c')
        
        ax.set_ylabel('Error Count', fontsize=11)
        ax.set_title(f'{model} - Error Breakdown', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(error_labels, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes) if isinstance(axes, np.ndarray) else 1):
        if isinstance(axes, np.ndarray):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved error breakdown to {output_path}")

def main():
    """Generate all visualizations."""
    eval_dirs = {
        'Metal Shaders (1000 iter)': 'evaluation/results/accuracy_metal_shaders',
        'Metal Optimized': 'evaluation/results/metal_optimized',
        'Accuracy Perfect': 'evaluation/results/accuracy_perfect',
        'Accuracy Ultimate': 'evaluation/results/accuracy_ultimate',
        'Accuracy Hardened': 'evaluation/results/accuracy_hardened',
        'Formatting Focus': 'evaluation/results/formatting_focus_vs_base',
    }
    
    results = {}
    for name, eval_dir in eval_dirs.items():
        summary = load_summary(eval_dir)
        if summary:
            results[name] = summary
            print(f"Loaded results for {name}")
    
    if not results:
        print("No evaluation results found!")
        return
    
    output_dir = Path('evaluation/results/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_accuracy_comparison(results, output_dir / 'accuracy_comparison.png')
    plot_improvement_delta(results, output_dir / 'improvement_delta.png')
    plot_error_breakdown(results, output_dir / 'error_breakdown.png')
    
    print(f"\nAll visualizations saved to {output_dir}/")

if __name__ == '__main__':
    main()
