import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*80)
print("SUBMISSION QUALITY ANALYSIS - COMPARISON WITH TRAINING DATA")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

try:
    submission = pd.read_csv('submission_fixed.csv')
    print(f"✓ Submission loaded: {len(submission)} predictions")
except FileNotFoundError:
    print("✗ Error: submission.csv not found!")
    exit(1)

try:
    y_train = pd.read_csv('data/y_train.csv')
    # Handle y_train structure
    if y_train.shape[1] > 1:
        possible_target_names = ['math_score', 'target', 'y', 'score']
        target_col = None
        for col in y_train.columns:
            if col.lower() in possible_target_names or 'math' in col.lower() or 'score' in col.lower():
                target_col = col
                break
        if target_col is None:
            target_col = y_train.columns[-1]
        y_train = y_train[target_col]
    else:
        y_train = y_train.iloc[:, 0]
    print(f"✓ Training targets loaded: {len(y_train)} samples")
except FileNotFoundError:
    print("✗ Error: y_train.csv not found!")
    exit(1)

# ============================================================================
# STEP 2: BASIC STATISTICS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("[2/7] BASIC STATISTICS COMPARISON")
print("="*80)

predictions = submission['math_score'].values
train_scores = y_train.values

# Calculate statistics for both
stats_comparison = pd.DataFrame({
    'Training Data': [
        len(train_scores),
        np.mean(train_scores),
        np.median(train_scores),
        np.std(train_scores),
        np.var(train_scores),
        np.min(train_scores),
        np.max(train_scores),
        np.max(train_scores) - np.min(train_scores),
        np.percentile(train_scores, 25),
        np.percentile(train_scores, 75),
        np.percentile(train_scores, 75) - np.percentile(train_scores, 25),
        stats.skew(train_scores),
        stats.kurtosis(train_scores)
    ],
    'Your Predictions': [
        len(predictions),
        np.mean(predictions),
        np.median(predictions),
        np.std(predictions),
        np.var(predictions),
        np.min(predictions),
        np.max(predictions),
        np.max(predictions) - np.min(predictions),
        np.percentile(predictions, 25),
        np.percentile(predictions, 75),
        np.percentile(predictions, 75) - np.percentile(predictions, 25),
        stats.skew(predictions),
        stats.kurtosis(predictions)
    ]
}, index=[
    'Count',
    'Mean',
    'Median',
    'Std Dev',
    'Variance',
    'Min',
    'Max',
    'Range',
    'Q1 (25%)',
    'Q3 (75%)',
    'IQR',
    'Skewness',
    'Kurtosis'
])

# Calculate differences
stats_comparison['Difference'] = stats_comparison['Your Predictions'] - stats_comparison['Training Data']
stats_comparison['Difference %'] = (stats_comparison['Difference'] / stats_comparison['Training Data'] * 100).round(2)

print("\nSTATISTICS COMPARISON:")
print("-" * 100)
print(stats_comparison.to_string())

# ============================================================================
# STEP 3: DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[3/7] DISTRIBUTION ANALYSIS")
print("="*80)

# Percentile comparison
percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
percentile_comparison = pd.DataFrame({
    'Training Data': [np.percentile(train_scores, p) for p in percentiles],
    'Your Predictions': [np.percentile(predictions, p) for p in percentiles]
}, index=[f'{p}th percentile' for p in percentiles])

percentile_comparison['Difference'] = percentile_comparison['Your Predictions'] - percentile_comparison['Training Data']

print("\nPERCENTILE COMPARISON:")
print("-" * 80)
print(percentile_comparison.to_string())

# ============================================================================
# STEP 4: QUALITY CHECKS
# ============================================================================
print("\n" + "="*80)
print("[4/7] QUALITY CHECKS")
print("="*80)

# Check for issues
issues = []
warnings_list = []

# Check for NaN or infinite values
if np.isnan(predictions).any():
    issues.append(f"⚠ WARNING: {np.isnan(predictions).sum()} NaN values detected!")
if np.isinf(predictions).any():
    issues.append(f"⚠ WARNING: {np.isinf(predictions).sum()} infinite values detected!")

# Check for negative scores (if math scores shouldn't be negative)
if (predictions < 0).any():
    warnings_list.append(f"⚠ Note: {(predictions < 0).sum()} negative predictions ({(predictions < 0).sum()/len(predictions)*100:.2f}%)")

# Check if predictions are within reasonable range of training data
train_range = (train_scores.min(), train_scores.max())
pred_range = (predictions.min(), predictions.max())

if predictions.min() < train_scores.min():
    warnings_list.append(f"⚠ Note: Min prediction ({predictions.min():.2f}) is below training min ({train_scores.min():.2f})")
if predictions.max() > train_scores.max():
    warnings_list.append(f"⚠ Note: Max prediction ({predictions.max():.2f}) is above training max ({train_scores.max():.2f})")

# Check if standard deviation is similar
std_ratio = np.std(predictions) / np.std(train_scores)
if std_ratio < 0.5:
    warnings_list.append(f"⚠ Note: Predictions have much lower variance than training data (ratio: {std_ratio:.2f})")
elif std_ratio > 2.0:
    warnings_list.append(f"⚠ Note: Predictions have much higher variance than training data (ratio: {std_ratio:.2f})")
else:
    warnings_list.append(f"✓ Good: Variance ratio is reasonable ({std_ratio:.2f})")

# Check if mean is similar
mean_diff_pct = abs(np.mean(predictions) - np.mean(train_scores)) / np.mean(train_scores) * 100
if mean_diff_pct > 10:
    warnings_list.append(f"⚠ Note: Mean prediction differs by {mean_diff_pct:.2f}% from training mean")
else:
    warnings_list.append(f"✓ Good: Mean prediction is close to training mean (diff: {mean_diff_pct:.2f}%)")

# Check distribution shape (skewness and kurtosis)
skew_diff = abs(stats.skew(predictions) - stats.skew(train_scores))
if skew_diff > 1.0:
    warnings_list.append(f"⚠ Note: Distribution skewness differs significantly (diff: {skew_diff:.2f})")
else:
    warnings_list.append(f"✓ Good: Distribution skewness is similar (diff: {skew_diff:.2f})")

if len(issues) > 0:
    print("\n⚠ CRITICAL ISSUES FOUND:")
    for issue in issues:
        print(issue)
else:
    print("\n✓ No critical issues found")

print("\nQUALITY ASSESSMENT:")
for warning in warnings_list:
    print(warning)

# ============================================================================
# STEP 5: SCORE DISTRIBUTION BY BINS
# ============================================================================
print("\n" + "="*80)
print("[5/7] SCORE DISTRIBUTION BY BINS")
print("="*80)

# Create bins for comparison
bins = np.linspace(min(train_scores.min(), predictions.min()), 
                   max(train_scores.max(), predictions.max()), 
                   11)

train_hist, _ = np.histogram(train_scores, bins=bins)
pred_hist, _ = np.histogram(predictions, bins=bins)

bin_comparison = pd.DataFrame({
    'Bin Range': [f'{bins[i]:.0f}-{bins[i+1]:.0f}' for i in range(len(bins)-1)],
    'Training Count': train_hist,
    'Training %': (train_hist / len(train_scores) * 100).round(2),
    'Prediction Count': pred_hist,
    'Prediction %': (pred_hist / len(predictions) * 100).round(2)
})

bin_comparison['Difference %'] = bin_comparison['Prediction %'] - bin_comparison['Training %']

print("\nDISTRIBUTION BY SCORE RANGES:")
print("-" * 100)
print(bin_comparison.to_string(index=False))

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("[6/7] GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Submission Quality Analysis - Comparison with Training Data', fontsize=16, fontweight='bold')

# Plot 1: Distribution comparison (Histogram)
axes[0, 0].hist(train_scores, bins=50, alpha=0.6, label='Training Data', color='blue', density=True)
axes[0, 0].hist(predictions, bins=50, alpha=0.6, label='Your Predictions', color='red', density=True)
axes[0, 0].axvline(train_scores.mean(), color='blue', linestyle='--', linewidth=2, label=f'Train Mean: {train_scores.mean():.2f}')
axes[0, 0].axvline(predictions.mean(), color='red', linestyle='--', linewidth=2, label=f'Pred Mean: {predictions.mean():.2f}')
axes[0, 0].set_xlabel('Math Score', fontsize=12)
axes[0, 0].set_ylabel('Density', fontsize=12)
axes[0, 0].set_title('Distribution Comparison (Histogram)', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: KDE (Kernel Density Estimation)
axes[0, 1].hist(train_scores, bins=50, alpha=0.3, label='Training Data', color='blue', density=True)
axes[0, 1].hist(predictions, bins=50, alpha=0.3, label='Your Predictions', color='red', density=True)
from scipy.stats import gaussian_kde
kde_train = gaussian_kde(train_scores)
kde_pred = gaussian_kde(predictions)
x_range = np.linspace(min(train_scores.min(), predictions.min()), 
                      max(train_scores.max(), predictions.max()), 1000)
axes[0, 1].plot(x_range, kde_train(x_range), 'b-', linewidth=2, label='Training KDE')
axes[0, 1].plot(x_range, kde_pred(x_range), 'r-', linewidth=2, label='Prediction KDE')
axes[0, 1].set_xlabel('Math Score', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].set_title('Distribution Comparison (KDE)', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Box plots
box_data = [train_scores, predictions]
bp = axes[1, 0].boxplot(box_data, labels=['Training Data', 'Your Predictions'], 
                        patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
axes[1, 0].set_ylabel('Math Score', fontsize=12)
axes[1, 0].set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Q-Q Plot (using quantiles since sizes differ)
# Create common quantiles
n_quantiles = min(len(train_scores), len(predictions), 1000)
quantile_levels = np.linspace(0, 1, n_quantiles)
train_quantiles = np.quantile(train_scores, quantile_levels)
pred_quantiles = np.quantile(predictions, quantile_levels)

axes[1, 1].scatter(train_quantiles, pred_quantiles, alpha=0.5, s=1)
min_val = min(train_quantiles.min(), pred_quantiles.min())
max_val = max(train_quantiles.max(), pred_quantiles.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
axes[1, 1].set_xlabel('Training Data Quantiles', fontsize=12)
axes[1, 1].set_ylabel('Prediction Quantiles', fontsize=12)
axes[1, 1].set_title('Q-Q Plot (Quantile-Quantile)', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Cumulative Distribution
train_sorted = np.sort(train_scores)
pred_sorted = np.sort(predictions)
train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)
pred_cdf = np.arange(1, len(pred_sorted)+1) / len(pred_sorted)
axes[2, 0].plot(train_sorted, train_cdf, label='Training Data', linewidth=2, color='blue')
axes[2, 0].plot(pred_sorted, pred_cdf, label='Your Predictions', linewidth=2, color='red')
axes[2, 0].set_xlabel('Math Score', fontsize=12)
axes[2, 0].set_ylabel('Cumulative Probability', fontsize=12)
axes[2, 0].set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: Statistics comparison bar chart
metrics = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
train_values = [train_scores.mean(), np.median(train_scores), train_scores.std(), 
                train_scores.min(), train_scores.max()]
pred_values = [predictions.mean(), np.median(predictions), predictions.std(), 
               predictions.min(), predictions.max()]

x = np.arange(len(metrics))
width = 0.35
axes[2, 1].bar(x - width/2, train_values, width, label='Training Data', color='lightblue')
axes[2, 1].bar(x + width/2, pred_values, width, label='Your Predictions', color='lightcoral')
axes[2, 1].set_ylabel('Value', fontsize=12)
axes[2, 1].set_title('Key Statistics Comparison', fontsize=13, fontweight='bold')
axes[2, 1].set_xticks(x)
axes[2, 1].set_xticklabels(metrics, rotation=45)
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('submission_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to 'submission_analysis.png'")

# ============================================================================
# STEP 7: FINAL ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("[7/7] FINAL ASSESSMENT")
print("="*80)

score = 0
max_score = 100

# Score based on various criteria
# 1. Mean similarity (20 points)
mean_similarity = max(0, 20 - abs(mean_diff_pct))
score += mean_similarity

# 2. Std similarity (20 points)
std_similarity = max(0, 20 - abs(std_ratio - 1) * 40)
score += std_similarity

# 3. Range appropriateness (20 points)
range_penalty = 0
if predictions.min() < train_scores.min():
    range_penalty += abs(predictions.min() - train_scores.min()) / train_scores.std()
if predictions.max() > train_scores.max():
    range_penalty += abs(predictions.max() - train_scores.max()) / train_scores.std()
range_score = max(0, 20 - range_penalty * 5)
score += range_score

# 4. Distribution shape similarity (20 points)
shape_score = max(0, 20 - skew_diff * 10)
score += shape_score

# 5. No critical issues (20 points)
if len(issues) == 0:
    score += 20
else:
    score += max(0, 20 - len(issues) * 10)

print("\nQUALITY SCORE BREAKDOWN:")
print("-" * 60)
print(f"Mean Similarity:        {mean_similarity:.1f}/20 points")
print(f"Std Dev Similarity:     {std_similarity:.1f}/20 points")
print(f"Range Appropriateness:  {range_score:.1f}/20 points")
print(f"Distribution Shape:     {shape_score:.1f}/20 points")
print(f"No Critical Issues:     {(20 if len(issues) == 0 else max(0, 20 - len(issues) * 10)):.1f}/20 points")
print("-" * 60)
print(f"TOTAL QUALITY SCORE:    {score:.1f}/100")
print("-" * 60)

if score >= 80:
    grade = "EXCELLENT"
    color = "🟢"
    comment = "Your predictions look very similar to the training data distribution!"
elif score >= 60:
    grade = "GOOD"
    color = "🟡"
    comment = "Your predictions are reasonable but could be improved."
elif score >= 40:
    grade = "FAIR"
    color = "🟠"
    comment = "Your predictions deviate notably from training data. Consider model improvements."
else:
    grade = "NEEDS IMPROVEMENT"
    color = "🔴"
    comment = "Your predictions differ significantly from training data. Review your model."

print(f"\n{color} GRADE: {grade}")
print(f"Comment: {comment}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("Review the generated plot 'submission_analysis.png' for visual insights.")
print("\nKey Recommendations:")
if mean_diff_pct > 10:
    print("  • Your predictions' mean differs significantly from training. Check for bias.")
if std_ratio < 0.7 or std_ratio > 1.3:
    print("  • Your predictions' variance is off. Model might be over/under-fitting.")
if skew_diff > 1.0:
    print("  • Distribution shape differs. Consider feature engineering or different model.")
if len(issues) > 0:
    print("  • Fix critical issues (NaN, inf values) before submission!")
if score >= 70:
    print("  • Your submission looks solid! Consider minor tweaks for improvement.")