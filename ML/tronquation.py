"""
Fix Negative Predictions and Truncate Low Predictions
Replace all negative predictions with 0 and all predictions < 50 with 0
"""
import pandas as pd
import numpy as np

def fix_negative_and_low_predictions(input_file='submission.csv',
                                     output_file='submission_fixed.csv',
                                     threshold=2):
    """
    Replace all negative predictions with 0 and all predictions below threshold with 0
    """
    print("="*80)
    print("FIXING NEGATIVE AND LOW PREDICTIONS")
    print("="*80)
    
    # Load submission
    print(f"\nLoading {input_file}...")
    submission = pd.read_csv(input_file, index_col=0)
    
    print(f"✓ Loaded: {submission.shape}")
    print(f"  Columns: {list(submission.columns)}")
    
    # Get prediction column (usually first column)
    pred_col = submission.columns[0]
    
    # Statistics before
    print(f"\nStatistics BEFORE:")
    print(f"  Mean:     {submission[pred_col].mean():.2f}")
    print(f"  Min:      {submission[pred_col].min():.2f}")
    print(f"  Max:      {submission[pred_col].max():.2f}")
    print(f"  Negative: {(submission[pred_col] < 0).sum()}")
    print(f"  Below {threshold}: {(submission[pred_col] < threshold).sum()}")
    
    # Fix negatives and truncate low values
    mask_low = submission[pred_col] < threshold
    n_low = mask_low.sum()
    
    if n_low > 0:
        print(f"\n  Found {n_low} predictions below {threshold} ({n_low/len(submission)*100:.2f}%)")
        print(f"  Setting these values to 0...")
        submission.loc[mask_low, pred_col] = 0
        print("✓ Fixed!")
    else:
        print(f"\n✓ No predictions below {threshold} found!")
    
    # Statistics after
    print(f"\nStatistics AFTER:")
    print(f"  Mean:     {submission[pred_col].mean():.2f}")
    print(f"  Min:      {submission[pred_col].min():.2f}")
    print(f"  Max:      {submission[pred_col].max():.2f}")
    print(f"  Negative: {(submission[pred_col] < 0).sum()}")
    print(f"  Below {threshold}: {(submission[pred_col] < threshold).sum()}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    submission.to_csv(output_file)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Fixed submission saved to: {output_file}")
    
    return submission


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'submission.csv'
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'submission_fixed.csv'
    
    print(f"""

Input:  {input_file}
Output: {output_file}
""")
    
    fix_negative_and_low_predictions(input_file, output_file)
