"""
Detailed Attention Pattern Analysis

Analyzes specific patterns in windowed model attention:
1. Attention locality (local vs global patterns)
2. Sleep stage-specific attention behavior
3. Modality weight dynamics
4. Prediction accuracy by attention characteristics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats

def load_results(json_path):
    """Load attention analysis results"""
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_modality_weights(results):
    """Analyze modality weight patterns"""
    print("\n" + "="*70)
    print("MODALITY WEIGHT ANALYSIS")
    print("="*70)
    
    ppg_weights = []
    ecg_weights = []
    stage_distributions = defaultdict(lambda: {'ppg': [], 'ecg': []})
    
    for window in results['windows']:
        ppg_w = window['modality_weights']['ppg']
        ecg_w = window['modality_weights']['ecg']
        
        ppg_weights.append(ppg_w)
        ecg_weights.append(ecg_w)
        
        # Group by dominant sleep stage in window
        labels = np.array(window['labels'])
        if len(labels) > 0:
            dominant_stage = stats.mode(labels[labels != -1], keepdims=False)[0] if np.any(labels != -1) else -1
            if dominant_stage != -1:
                stage_distributions[int(dominant_stage)]['ppg'].append(ppg_w)
                stage_distributions[int(dominant_stage)]['ecg'].append(ecg_w)
    
    print(f"\nOverall Statistics:")
    print(f"  PPG Weight: {np.mean(ppg_weights):.4f} ± {np.std(ppg_weights):.4f}")
    print(f"    Range: [{np.min(ppg_weights):.4f}, {np.max(ppg_weights):.4f}]")
    print(f"  ECG Weight: {np.mean(ecg_weights):.4f} ± {np.std(ecg_weights):.4f}")
    print(f"    Range: [{np.min(ecg_weights):.4f}, {np.max(ecg_weights):.4f}]")
    
    # Analyze temporal trend
    if len(ppg_weights) > 1:
        correlation = np.corrcoef(range(len(ppg_weights)), ppg_weights)[0, 1]
        print(f"\n  PPG weight temporal correlation: {correlation:.4f}")
        if abs(correlation) > 0.5:
            trend = "increasing" if correlation > 0 else "decreasing"
            print(f"  → Strong {trend} trend detected!")
    
    # Stage-specific weights
    print(f"\nModality Weights by Dominant Sleep Stage:")
    stage_names = {0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
    for stage in sorted(stage_distributions.keys()):
        if stage_distributions[stage]['ppg']:
            ppg_avg = np.mean(stage_distributions[stage]['ppg'])
            ecg_avg = np.mean(stage_distributions[stage]['ecg'])
            print(f"  {stage_names.get(stage, f'Stage {stage}'):5s}: PPG={ppg_avg:.4f}, ECG={ecg_avg:.4f}")
    
    return {
        'ppg_weights': ppg_weights,
        'ecg_weights': ecg_weights,
        'stage_distributions': stage_distributions
    }

def analyze_prediction_accuracy(results):
    """Analyze prediction accuracy patterns"""
    print("\n" + "="*70)
    print("PREDICTION ACCURACY ANALYSIS")
    print("="*70)
    
    stage_names = {0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
    
    # Overall accuracy
    all_preds = []
    all_labels = []
    
    for window in results['windows']:
        preds = np.array(window['predictions'])
        labels = np.array(window['labels'])
        valid_mask = labels != -1
        all_preds.extend(preds[valid_mask].tolist())
        all_labels.extend(labels[valid_mask].tolist())
    
    overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    # Per-stage accuracy
    print(f"\nPer-Stage Accuracy:")
    stage_accs = {}
    for stage in range(4):
        stage_mask = np.array(all_labels) == stage
        if np.any(stage_mask):
            stage_preds = np.array(all_preds)[stage_mask]
            stage_labels = np.array(all_labels)[stage_mask]
            acc = np.mean(stage_preds == stage_labels)
            stage_accs[stage] = acc
            print(f"  {stage_names[stage]:5s}: {acc:.4f} ({acc*100:.2f}%) - {np.sum(stage_mask)} samples")
        else:
            stage_accs[stage] = 0.0
            print(f"  {stage_names[stage]:5s}: No samples")
    
    # Confusion analysis
    print(f"\nCommon Confusions:")
    confusion_pairs = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        if pred != label:
            pair = (label, pred)
            confusion_pairs[pair] += 1
    
    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for (true_stage, pred_stage), count in sorted_confusions[:5]:
        percent = count / len(all_labels) * 100
        print(f"  {stage_names[true_stage]:5s} → {stage_names[pred_stage]:5s}: {count:4d} times ({percent:.2f}%)")
    
    return {
        'overall_accuracy': overall_acc,
        'stage_accuracies': stage_accs,
        'confusion_pairs': confusion_pairs
    }

def analyze_window_transitions(results):
    """Analyze prediction consistency at window boundaries"""
    print("\n" + "="*70)
    print("WINDOW BOUNDARY ANALYSIS")
    print("="*70)
    
    # Check prediction agreement in overlapping regions
    window_epochs = results['config']['windowing']['window_epochs']
    overlap_percent = results['config']['windowing']['overlap_percent']
    overlap_epochs = int(window_epochs * overlap_percent / 100)
    
    print(f"\nWindow Configuration:")
    print(f"  Window size: {window_epochs} epochs")
    print(f"  Overlap: {overlap_percent}% ({overlap_epochs} epochs)")
    
    # Compare predictions in overlapping regions
    disagreements = []
    agreements = []
    
    for i in range(len(results['windows']) - 1):
        window1 = results['windows'][i]
        window2 = results['windows'][i + 1]
        
        # Get predictions from overlap region
        # Window 1 end overlaps with Window 2 start
        preds1_overlap = np.array(window1['predictions'][-overlap_epochs:])
        preds2_overlap = np.array(window2['predictions'][:overlap_epochs])
        
        agreement = np.mean(preds1_overlap == preds2_overlap)
        agreements.append(agreement)
        disagreements.append(1 - agreement)
    
    if agreements:
        print(f"\nOverlap Region Agreement:")
        print(f"  Average agreement: {np.mean(agreements):.4f} ({np.mean(agreements)*100:.2f}%)")
        print(f"  Min agreement: {np.min(agreements):.4f}")
        print(f"  Max agreement: {np.max(agreements):.4f}")
        print(f"  Std: {np.std(agreements):.4f}")
        
        if np.mean(agreements) > 0.85:
            print(f"  → High consistency across window boundaries! ✓")
        elif np.mean(agreements) < 0.70:
            print(f"  → Low consistency - potential boundary artifacts! ⚠")
    
    return {
        'agreements': agreements,
        'avg_agreement': np.mean(agreements) if agreements else 0
    }

def analyze_stage_transitions(results):
    """Analyze sleep stage transition patterns"""
    print("\n" + "="*70)
    print("SLEEP STAGE TRANSITION ANALYSIS")
    print("="*70)
    
    stage_names = {0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
    
    # Collect all predictions
    all_preds = []
    all_labels = []
    
    for window in results['windows']:
        preds = window['predictions']
        labels = window['labels']
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # Count transitions
    pred_transitions = defaultdict(int)
    label_transitions = defaultdict(int)
    
    for i in range(len(all_preds) - 1):
        if all_labels[i] != -1 and all_labels[i+1] != -1:
            label_trans = (all_labels[i], all_labels[i+1])
            label_transitions[label_trans] += 1
            
            pred_trans = (all_preds[i], all_preds[i+1])
            pred_transitions[pred_trans] += 1
    
    print(f"\nMost Common Predicted Transitions:")
    sorted_pred = sorted(pred_transitions.items(), key=lambda x: x[1], reverse=True)
    for (from_stage, to_stage), count in sorted_pred[:8]:
        if from_stage != to_stage:  # Only show actual transitions
            percent = count / sum(pred_transitions.values()) * 100
            print(f"  {stage_names[from_stage]:5s} → {stage_names[to_stage]:5s}: {count:4d} ({percent:.2f}%)")
    
    print(f"\nMost Common Ground Truth Transitions:")
    sorted_label = sorted(label_transitions.items(), key=lambda x: x[1], reverse=True)
    for (from_stage, to_stage), count in sorted_label[:8]:
        if from_stage != to_stage:
            percent = count / sum(label_transitions.values()) * 100
            print(f"  {stage_names[from_stage]:5s} → {stage_names[to_stage]:5s}: {count:4d} ({percent:.2f}%)")
    
    return {
        'pred_transitions': pred_transitions,
        'label_transitions': label_transitions
    }

def analyze_modality_correlation(weight_data):
    """Analyze correlation between modality weights and accuracy"""
    print("\n" + "="*70)
    print("MODALITY WEIGHT vs ACCURACY CORRELATION")
    print("="*70)
    
    ppg_weights = weight_data['ppg_weights']
    
    # Check for patterns
    print(f"\nKey Observations:")
    
    # Variance
    variance = np.var(ppg_weights)
    print(f"  PPG weight variance: {variance:.6f}")
    
    if variance < 0.001:
        print(f"  → Very stable weights - model has consistent modality preferences")
    elif variance > 0.005:
        print(f"  → High variance - model adapts weights significantly across windows")
    
    # Extremes
    max_ppg = max(ppg_weights)
    min_ppg = min(ppg_weights)
    range_ppg = max_ppg - min_ppg
    
    print(f"  Weight range: {range_ppg:.4f}")
    if range_ppg > 0.15:
        print(f"  → Large range indicates context-dependent modality weighting")
    
    return {
        'variance': variance,
        'range': range_ppg
    }

def main():
    """Run comprehensive pattern analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze attention patterns in detail')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to attention_analysis_results.json')
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE ATTENTION PATTERN ANALYSIS")
    print("="*70)
    
    # Load results
    results = load_results(args.results)
    print(f"\nAnalyzing {results['n_windows']} windows")
    print(f"Model: {results['config']['model_type']}")
    
    # Run analyses
    weight_data = analyze_modality_weights(results)
    accuracy_data = analyze_prediction_accuracy(results)
    boundary_data = analyze_window_transitions(results)
    transition_data = analyze_stage_transitions(results)
    correlation_data = analyze_modality_correlation(weight_data)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF KEY FINDINGS")
    print("="*70)
    
    print(f"\n1. Modality Weighting:")
    print(f"   - PPG dominance: {np.mean(weight_data['ppg_weights']):.2%} average")
    print(f"   - Weight stability: {'High' if correlation_data['variance'] < 0.001 else 'Moderate'}")
    
    print(f"\n2. Prediction Performance:")
    print(f"   - Overall accuracy: {accuracy_data['overall_accuracy']:.2%}")
    print(f"   - Best stage: {max(accuracy_data['stage_accuracies'].items(), key=lambda x: x[1])[0]} "
          f"({max(accuracy_data['stage_accuracies'].values()):.2%})")
    print(f"   - Worst stage: {min(accuracy_data['stage_accuracies'].items(), key=lambda x: x[1])[0]} "
          f"({min(accuracy_data['stage_accuracies'].values()):.2%})")
    
    print(f"\n3. Window Boundary Behavior:")
    print(f"   - Overlap consistency: {boundary_data['avg_agreement']:.2%}")
    print(f"   - Boundary quality: {'Excellent' if boundary_data['avg_agreement'] > 0.85 else 'Good' if boundary_data['avg_agreement'] > 0.75 else 'Needs improvement'}")
    
    print(f"\n4. Sleep Stage Transitions:")
    most_common_pred = max(transition_data['pred_transitions'].items(), key=lambda x: x[1])
    print(f"   - Most common predicted: {most_common_pred[0]} ({most_common_pred[1]} times)")
    most_common_true = max(transition_data['label_transitions'].items(), key=lambda x: x[1])
    print(f"   - Most common true: {most_common_true[0]} ({most_common_true[1]} times)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
