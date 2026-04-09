#!/usr/bin/env python3
"""
Generate 7 different types of graphs for training run analysis.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

RUN_DIR = Path("/Users/luca/Projects/ai-factory/artifacts/runs/accuracy_ultimate_95_plus-20260408-174421")
OUTPUT_DIR = RUN_DIR / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
with open(RUN_DIR / "metrics/dataset_report.json") as f:
    dataset_data = json.load(f)

with open(RUN_DIR / "metrics/model_report.json") as f:
    model_data = json.load(f)

with open(RUN_DIR / "metrics/metrics.json") as f:
    final_metrics = json.load(f)

# Load training metrics
training_metrics = []
with open(RUN_DIR / "logs/training_metrics.jsonl") as f:
    for line in f:
        training_metrics.append(json.loads(line.strip()))

df = pd.DataFrame(training_metrics)

print(f"Loaded {len(df)} training steps")
print(f"Final eval loss: {final_metrics['eval_loss']:.6f}")
print(f"Total parameters: {model_data['total_parameters']:,}")
print(f"Trainable parameters: {model_data['trainable_parameters']:,}")

# GRAPH 1: Line Graph - Training Loss Over Epochs
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['epoch'], df['loss'], alpha=0.6, linewidth=1, color='#2E86AB', label='Training Loss')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_training_loss_epochs.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 2: Bar Graph - Dataset Difficulty Distribution (Train vs Eval)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

train_difficulty = dataset_data['train']['stats']['difficulty_counts']
eval_difficulty = dataset_data['eval']['stats']['difficulty_counts']

difficulties = ['easy', 'medium', 'hard', 'olympiad']
train_counts = [train_difficulty.get(d, 0) for d in difficulties]
eval_counts = [eval_difficulty.get(d, 0) for d in difficulties]

x = np.arange(len(difficulties))
width = 0.35

ax1.bar(x - width/2, train_counts, width, label='Train', color='#2E86AB', alpha=0.8)
ax1.bar(x + width/2, eval_counts, width, label='Eval', color='#F25F5C', alpha=0.8)
ax1.set_xlabel('Difficulty Level', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Dataset Difficulty Distribution', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(difficulties)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Percentage view
train_total = sum(train_counts)
eval_total = sum(eval_counts)
train_pct = [c/train_total*100 for c in train_counts]
eval_pct = [c/eval_total*100 for c in eval_counts]

ax2.bar(x - width/2, train_pct, width, label='Train', color='#2E86AB', alpha=0.8)
ax2.bar(x + width/2, eval_pct, width, label='Eval', color='#F25F5C', alpha=0.8)
ax2.set_xlabel('Difficulty Level', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Dataset Difficulty Distribution (Percentage)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(difficulties)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2_difficulty_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 3: Heat Map - Topic vs Difficulty Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Prepare data for heatmaps
topics = list(dataset_data['train']['stats']['topic_counts'].keys())
difficulties = ['easy', 'medium', 'hard', 'olympiad']

# Create topic-difficulty matrix for train
train_matrix = np.zeros((len(topics), len(difficulties)))
for i, topic in enumerate(topics):
    for j, diff in enumerate(difficulties):
        # Estimate distribution based on overall proportions
        train_matrix[i, j] = dataset_data['train']['stats']['topic_counts'][topic] * \
                           (dataset_data['train']['stats']['difficulty_counts'][diff] / dataset_data['train']['stats']['num_records'])

# Create topic-difficulty matrix for eval
eval_matrix = np.zeros((len(topics), len(difficulties)))
for i, topic in enumerate(topics):
    for j, diff in enumerate(difficulties):
        eval_matrix[i, j] = dataset_data['eval']['stats']['topic_counts'][topic] * \
                          (dataset_data['eval']['stats']['difficulty_counts'][diff] / dataset_data['eval']['stats']['num_records'])

# Train heatmap
im1 = ax1.imshow(train_matrix, cmap='Blues', aspect='auto')
ax1.set_xticks(np.arange(len(difficulties)))
ax1.set_yticks(np.arange(len(topics)))
ax1.set_xticklabels(difficulties)
ax1.set_yticklabels(topics)
ax1.set_xlabel('Difficulty', fontsize=12)
ax1.set_ylabel('Topic', fontsize=12)
ax1.set_title('Train: Topic × Difficulty Distribution', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(topics)):
    for j in range(len(difficulties)):
        text = ax1.text(j, i, f'{train_matrix[i, j]:.0f}',
                       ha="center", va="center", color="white", fontsize=8)

plt.colorbar(im1, ax=ax1, label='Estimated Count')

# Eval heatmap
im2 = ax2.imshow(eval_matrix, cmap='Reds', aspect='auto')
ax2.set_xticks(np.arange(len(difficulties)))
ax2.set_yticks(np.arange(len(topics)))
ax2.set_xticklabels(difficulties)
ax2.set_yticklabels(topics)
ax2.set_xlabel('Difficulty', fontsize=12)
ax2.set_ylabel('Topic', fontsize=12)
ax2.set_title('Eval: Topic × Difficulty Distribution', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(topics)):
    for j in range(len(difficulties)):
        text = ax2.text(j, i, f'{eval_matrix[i, j]:.0f}',
                       ha="center", va="center", color="white", fontsize=8)

plt.colorbar(im2, ax=ax2, label='Estimated Count')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3_topic_difficulty_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 4: Quadrant - Learning Rate vs Gradient Norm
fig, ax = plt.subplots(figsize=(12, 10))

# Create quadrants based on median values
median_lr = df['learning_rate'].median()
median_grad = df['grad_norm'].median()

# Color by epoch
scatter = ax.scatter(df['learning_rate'], df['grad_norm'], 
                    c=df['epoch'], cmap='viridis', alpha=0.6, s=20)
ax.set_xlabel('Learning Rate', fontsize=12)
ax.set_ylabel('Gradient Norm', fontsize=12)
ax.set_title('Learning Rate vs Gradient Norm (colored by Epoch)', fontsize=14, fontweight='bold')

# Add quadrant lines
ax.axvline(median_lr, color='gray', linestyle='--', alpha=0.5)
ax.axhline(median_grad, color='gray', linestyle='--', alpha=0.5)

# Add quadrant labels
ax.text(median_lr * 0.5, median_grad * 2, 'Low LR\nHigh Grad', 
        ha='center', va='center', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(median_lr * 1.5, median_grad * 2, 'High LR\nHigh Grad', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(median_lr * 0.5, median_grad * 0.3, 'Low LR\nLow Grad', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.text(median_lr * 1.5, median_grad * 0.3, 'High LR\nLow Grad', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Epoch', fontsize=11)

ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '4_lr_grad_quadrant.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 5: Creative - Parameter Efficiency Donut Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Donut chart
total_params = model_data['total_parameters']
trainable_params = model_data['trainable_parameters']
frozen_params = total_params - trainable_params

sizes = [trainable_params, frozen_params]
labels = [f'Trainable\n{trainable_params:,}\n({trainable_params/total_params*100:.2f}%)', 
          f'Frozen\n{frozen_params:,}\n({frozen_params/total_params*100:.2f}%)']
colors = ['#2E86AB', '#E0E0E0']
explode = (0.05, 0)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.1f%%', shadow=True, startangle=90,
                                   wedgeprops=dict(width=0.5, edgecolor='white'))

ax1.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')

# Bar chart comparison
categories = ['Total Parameters', 'Trainable Parameters', 'Frozen Parameters']
values = [total_params, trainable_params, frozen_params]
colors = ['#7FB3D5', '#2E86AB', '#E0E0E0']

bars = ax2.bar(categories, values, color=colors, alpha=0.8)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Parameter Count Breakdown', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,}',
             ha='center', va='bottom', fontsize=9)

ax2.set_yscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '5_parameter_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 6: Bar Graph - Reasoning Style Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

train_reasoning = dataset_data['train']['stats']['reasoning_style_counts']
eval_reasoning = dataset_data['eval']['stats']['reasoning_style_counts']

styles = list(train_reasoning.keys())
train_style_counts = [train_reasoning[s] for s in styles]
eval_style_counts = [eval_reasoning[s] for s in styles]

x = np.arange(len(styles))
width = 0.35

ax1.bar(x - width/2, train_style_counts, width, label='Train', color='#2E86AB', alpha=0.8)
ax1.bar(x + width/2, eval_style_counts, width, label='Eval', color='#F25F5C', alpha=0.8)
ax1.set_xlabel('Reasoning Style', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Reasoning Style Distribution', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(styles)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Percentage view
train_total = sum(train_style_counts)
eval_total = sum(eval_style_counts)
train_pct = [c/train_total*100 for c in train_style_counts]
eval_pct = [c/eval_total*100 for c in eval_style_counts]

ax2.bar(x - width/2, train_pct, width, label='Train', color='#2E86AB', alpha=0.8)
ax2.bar(x + width/2, eval_pct, width, label='Eval', color='#F25F5C', alpha=0.8)
ax2.set_xlabel('Reasoning Style', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Reasoning Style Distribution (Percentage)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(styles)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '6_reasoning_style_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 7: Line Graph - Gradient Norm Evolution with Learning Rate
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot gradient norm
color1 = '#2E86AB'
ax1.plot(df['epoch'], df['grad_norm'], color=color1, alpha=0.7, linewidth=1.5, label='Gradient Norm')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Gradient Norm', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Create twin axis for learning rate
ax2 = ax1.twinx()
color2 = '#F25F5C'
ax2.plot(df['epoch'], df['learning_rate'], color=color2, alpha=0.7, linewidth=1.5, label='Learning Rate')
ax2.set_ylabel('Learning Rate', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.set_title('Gradient Norm and Learning Rate Evolution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '7_grad_norm_lr_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll graphs saved to: {OUTPUT_DIR}")
print("\nGenerated graphs:")
print("1. Training Loss Over Epochs (line graph)")
print("2. Dataset Difficulty Distribution (bar graph)")
print("3. Topic × Difficulty Heatmap (heat map)")
print("4. Learning Rate vs Gradient Norm Quadrant (quadrant plot)")
print("5. Parameter Efficiency (creative donut + bar chart)")
print("6. Reasoning Style Distribution (bar graph)")
print("7. Gradient Norm and Learning Rate Evolution (dual line graph)")
