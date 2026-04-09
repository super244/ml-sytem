#!/usr/bin/env python3
"""Combine all custom datasets into a single training file."""

import json
from pathlib import Path

def combine_custom_datasets(output_path):
    """Combine all custom datasets from the catalog."""
    catalog_path = Path('data/catalog.json')
    with open(catalog_path) as f:
        catalog = json.load(f)
    
    combined_examples = []
    seen_ids = set()
    
    for dataset in catalog['datasets']:
        if dataset['kind'] == 'custom' and dataset['num_rows'] > 0:
            dataset_path = Path(dataset['path'])
            if dataset_path.exists():
                print(f"Loading {dataset['title']} ({dataset['num_rows']} rows)...")
                with open(dataset_path) as f:
                    for line in f:
                        example = json.loads(line)
                        # Deduplicate by ID
                        if example.get('id') not in seen_ids:
                            # Add metadata
                            example['source_dataset'] = dataset['id']
                            example['difficulty'] = example.get('difficulty', 'medium')
                            example['topic'] = example.get('topic', 'calculus')
                            combined_examples.append(example)
                            seen_ids.add(example.get('id'))
    
    print(f"Combined {len(combined_examples)} unique examples from all custom datasets")
    
    # Shuffle to mix different topics
    import random
    random.shuffle(combined_examples)
    
    # Split into train/eval (80/20)
    split_idx = int(len(combined_examples) * 0.8)
    train_examples = combined_examples[:split_idx]
    eval_examples = combined_examples[split_idx:]
    
    # Write training set
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = Path(output_path)
    with open(train_path, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Write eval set
    eval_path = train_path.parent / train_path.name.replace('train', 'eval')
    with open(eval_path, 'w') as f:
        for example in eval_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Training set: {len(train_examples)} examples -> {train_path}")
    print(f"Eval set: {len(eval_examples)} examples -> {eval_path}")
    
    return len(combined_examples)

if __name__ == '__main__':
    combine_custom_datasets('data/processed/train_combined.jsonl')
