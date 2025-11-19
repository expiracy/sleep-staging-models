"""
Tune windowing parameters to minimize accuracy loss.

Tests different window sizes and overlap ratios to find optimal configuration
that balances memory savings with prediction accuracy.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse

def run_windowed_inference(checkpoint, model_type, subject_id, window_minutes, overlap, 
                           ppg_file, ecg_file, index_file, device='cuda'):
    """Run windowed inference and extract metrics"""
    
    cmd = [
        'python', 'windowed_inference.py',
        '--checkpoint', checkpoint,
        '--model_type', model_type,
        '--subject_id', str(subject_id),
        '--window_minutes', str(window_minutes),
        '--overlap', str(overlap),
        '--device', device,
        '--ppg_file', ppg_file,
        '--ecg_file', ecg_file,
        '--index_file', index_file
    ]
    
    print(f"\nTesting: Window={window_minutes}min, Overlap={overlap:.0%}")
    print("=" * 60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output for metrics
    lines = result.stdout.split('\n')
    metrics = {}
    
    for line in lines:
        if 'Cohen\'s Kappa:' in line:
            metrics['kappa'] = float(line.split(':')[1].strip())
        elif 'Accuracy:' in line:
            metrics['accuracy'] = float(line.split(':')[1].strip())
        elif 'F1-Score (Macro):' in line:
            metrics['f1_macro'] = float(line.split(':')[1].strip())
        elif 'Mean confidence:' in line:
            metrics['confidence'] = float(line.split(':')[1].strip())
        elif 'Created' in line and 'windows' in line:
            metrics['num_windows'] = int(line.split()[1])
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Tune windowing parameters')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='crossattn_ecg')
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--ppg_file', type=str, 
                        default='..\\..\\data\\mesa_processed\\mesa_ppg_with_labels.h5')
    parser.add_argument('--ecg_file', type=str,
                        default='..\\..\\data\\mesa_processed\\mesa_real_ecg.h5')
    parser.add_argument('--index_file', type=str,
                        default='..\\..\\data\\mesa_processed\\mesa_subject_index.h5')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--baseline_kappa', type=float, default=0.6799,
                        help='Baseline kappa from full-sequence inference')
    
    args = parser.parse_args()
    
    # Test configurations
    # Start with larger windows and increase overlap
    configs = [
        # Large windows (should be closer to baseline)
        (120, 0.3),  # 2 hours, 30% overlap
        (90, 0.3),   # 1.5 hours, 30% overlap
        (60, 0.3),   # 1 hour, 30% overlap
        (60, 0.4),   # 1 hour, 40% overlap
        (60, 0.5),   # 1 hour, 50% overlap
        
        # Medium-large windows
        (45, 0.4),   # 45 min, 40% overlap
        (45, 0.5),   # 45 min, 50% overlap
        (30, 0.5),   # 30 min, 50% overlap
        
        # Medium windows with heavy overlap
        (20, 0.5),   # 20 min, 50% overlap
        (15, 0.5),   # 15 min, 50% overlap
    ]
    
    results = []
    baseline_kappa = args.baseline_kappa
    
    print("\n" + "=" * 80)
    print(f"WINDOWING PARAMETER TUNING")
    print(f"Baseline (Full Sequence) Kappa: {baseline_kappa:.4f}")
    print(f"Target: Within 5-10% of baseline ({baseline_kappa*0.90:.4f} - {baseline_kappa*0.95:.4f})")
    print("=" * 80)
    
    for window_minutes, overlap in configs:
        try:
            metrics = run_windowed_inference(
                checkpoint=args.checkpoint,
                model_type=args.model_type,
                subject_id=args.subject_id,
                window_minutes=window_minutes,
                overlap=overlap,
                ppg_file=args.ppg_file,
                ecg_file=args.ecg_file,
                index_file=args.index_file,
                device=args.device
            )
            
            # Calculate degradation
            kappa = metrics.get('kappa', 0)
            degradation = ((baseline_kappa - kappa) / baseline_kappa) * 100
            
            results.append({
                'window_minutes': window_minutes,
                'overlap_ratio': overlap,
                'kappa': kappa,
                'accuracy': metrics.get('accuracy', 0),
                'f1_macro': metrics.get('f1_macro', 0),
                'confidence': metrics.get('confidence', 0),
                'num_windows': metrics.get('num_windows', 0),
                'degradation_pct': degradation,
                'within_target': degradation <= 10
            })
            
            print(f"Kappa: {kappa:.4f} ({degradation:+.1f}% vs baseline)")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('kappa', ascending=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (sorted by kappa)")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Highlight best configurations
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 80)
    
    target_configs = df[df['within_target']]
    if len(target_configs) > 0:
        print("\n✓ Configurations within 10% degradation:")
        for _, row in target_configs.iterrows():
            memory_reduction = 600 / row['window_minutes']  # Approximate
            print(f"  {row['window_minutes']}min window, {row['overlap_ratio']:.0%} overlap:")
            print(f"    Kappa: {row['kappa']:.4f} ({row['degradation_pct']:+.1f}%)")
            print(f"    Windows: {row['num_windows']}")
            print(f"    Memory reduction: ~{memory_reduction:.0f}x")
    else:
        print("\nNo configurations met the 10% threshold.")
        print("Best configuration:")
        best = df.iloc[0]
        print(f"  {best['window_minutes']}min window, {best['overlap_ratio']:.0%} overlap:")
        print(f"    Kappa: {best['kappa']:.4f} ({best['degradation_pct']:+.1f}%)")
        print(f"    Windows: {best['num_windows']}")
    
    # Save results
    output_file = '../../outputs/windowing_tuning_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
