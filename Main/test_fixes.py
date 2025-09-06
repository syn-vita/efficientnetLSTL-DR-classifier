#!/usr/bin/env python3
import pandas as pd
import torch
from collections import Counter

# Test the class weight calculation
def test_class_weights():
    # Load dataset
    df = pd.read_csv("dataset.csv")
    
    # Simulate what happens in fold 1
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df['id_code']
    y = df['diagnosis']
    
    train_index, val_index = next(skf.split(X, y))
    train_df = df.iloc[train_index].reset_index(drop=True)
    
    print("Original class distribution in training fold:")
    class_counts = train_df['diagnosis'].value_counts().sort_index()
    print(class_counts.to_dict())
    
    # Calculate new class weights
    total_samples = len(train_df)
    NUM_CLASSES = 5
    
    class_weights = {}
    for class_id in range(NUM_CLASSES):
        if class_id in class_counts:
            class_weights[class_id] = total_samples / (NUM_CLASSES * class_counts[class_id])
        else:
            class_weights[class_id] = 1.0
    
    print(f"\nClass weights: {class_weights}")
    
    # Test sample weights
    sample_weights = [class_weights[label] for label in train_df['diagnosis']]
    print(f"\nSample weight range: {min(sample_weights):.4f} - {max(sample_weights):.4f}")
    
    # Simulate sampling
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Sample a batch to see distribution
    sample_indices = list(sampler)[:1000]  # Sample 1000 indices
    sampled_labels = [train_df.iloc[idx]['diagnosis'] for idx in sample_indices]
    sampled_counts = Counter(sampled_labels)
    
    print(f"\nSampled distribution from 1000 samples:")
    print(dict(sorted(sampled_counts.items())))
    
    expected_per_class = 1000 / NUM_CLASSES
    print(f"Expected per class (if balanced): {expected_per_class:.1f}")

if __name__ == "__main__":
    test_class_weights()
