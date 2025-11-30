import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
print("="*80)
print("GPU CONFIGURATION CHECK")
print("="*80)

# Check if GPU is available
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ GPU detected!")
        print("\nGPU Info:")
        print(result.stdout.split('\n')[8:12])  # Print relevant GPU info lines
        USE_GPU = True
    else:
        print("⚠ No GPU detected. Will use CPU.")
        USE_GPU = False
except:
    print("⚠ Could not check GPU status. Will attempt GPU training anyway.")
    USE_GPU = True

print("="*80)
print("DETAILED R² ANALYSIS - TRAIN vs VALIDATION (GPU-ACCELERATED)")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================
print("\n[1/5] Loading data...")
X_train_full = pd.read_csv('data/X_train.csv')
y_train_full = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

print(f"X_train shape: {X_train_full.shape}")
print(f"y_train shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}")

# Handle y_train
print(f"\ny_train columns: {y_train_full.columns.tolist()}")
if y_train_full.shape[1] > 1:
    possible_target_names = ['math_score', 'target', 'y', 'score']
    target_col = None
    for col in y_train_full.columns:
        if col.lower() in possible_target_names or 'math' in col.lower() or 'score' in col.lower():
            target_col = col
            break
    if target_col is None:
        target_col = y_train_full.columns[-1]
    print(f"Using '{target_col}' as target variable")
    y_train_full = y_train_full[target_col]
else:
    y_train_full = y_train_full.iloc[:, 0]

if len(y_train_full) != len(X_train_full):
    min_len = min(len(X_train_full), len(y_train_full))
    X_train_full = X_train_full.iloc[:min_len]
    y_train_full = y_train_full.iloc[:min_len]

print(f"\nFinal shapes - X_train: {X_train_full.shape}, y_train: {y_train_full.shape}")

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================
print("\n[2/5] Preprocessing...")

# Remove empty columns from X_test
empty_cols_test = X_test.columns[X_test.isnull().all()].tolist()
X_train_full = X_train_full.drop(columns=empty_cols_test, errors='ignore')
X_test = X_test.drop(columns=empty_cols_test, errors='ignore')

# Identify column types
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object']).columns.tolist()

# Remove high missing columns
missing_threshold = 0.999
missing_pct = X_train_full.isnull().sum() / len(X_train_full)
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
X_train_full = X_train_full.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

numeric_cols = [col for col in numeric_cols if col in X_train_full.columns]
categorical_cols = [col for col in categorical_cols if col in X_train_full.columns]

# Encode categorical
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Impute
numeric_imputer = SimpleImputer(strategy='median')
X_train_full[numeric_cols] = numeric_imputer.fit_transform(X_train_full[numeric_cols])
X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_full[categorical_cols] = cat_imputer.fit_transform(X_train_full[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
else:
    cat_imputer = None

# Variance filtering
selector = VarianceThreshold(threshold=0.0)
X_train_selected = selector.fit_transform(X_train_full)
X_test_selected = selector.transform(X_test)
selected_features = X_train_full.columns[selector.get_support()].tolist()
X_train_full = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)

print(f"Features after preprocessing: {len(selected_features)}")

# ============================================================================
# STEP 3: TRAIN/VALIDATION SPLIT
# ============================================================================
print("\n[3/5] Creating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, 
    random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"\nTraining target stats:")
print(f"  Mean: {y_train.mean():.2f}")
print(f"  Std: {y_train.std():.2f}")
print(f"  Min: {y_train.min():.2f}")
print(f"  Max: {y_train.max():.2f}")
print(f"\nValidation target stats:")
print(f"  Mean: {y_val.mean():.2f}")
print(f"  Std: {y_val.std():.2f}")
print(f"  Min: {y_val.min():.2f}")
print(f"  Max: {y_val.max():.2f}")

# ============================================================================
# STEP 4: FEATURE SELECTION (GPU-ACCELERATED)
# ============================================================================
print("\n[4/5] Selecting important features...")
print("🚀 Training quick model with GPU acceleration...")

# GPU-accelerated quick model for feature selection
quick_model = xgb.XGBRegressor(
    n_estimators=400, 
    random_state=42,
    tree_method='gpu_hist' if USE_GPU else 'hist',  # GPU acceleration
    device='cuda' if USE_GPU else 'cpu',  # Specify device
    predictor='gpu_predictor' if USE_GPU else 'auto'  # GPU predictor
)
quick_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': quick_model.feature_importances_
}).sort_values('importance', ascending=False)

top_n_features = min(200, len(X_train.columns))
top_features = feature_importance.head(top_n_features)['feature'].tolist()

X_train = X_train[top_features]
X_val = X_val[top_features]
X_train_full = X_train_full[top_features]
X_test = X_test[top_features]

print(f"Using top {len(top_features)} features")

# ============================================================================
# STEP 5: TRAIN MODEL AND ANALYZE R² (GPU-ACCELERATED)
# ============================================================================
print("\n[5/5] Training model and analyzing R²...")
print("\n" + "="*80)
print("TRAINING DEEP TREES MODEL WITH GPU ACCELERATION")
print("="*80)

# GPU-ACCELERATED MODEL
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=12,
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.5,
    reg_alpha=0.5,
    reg_lambda=2,
    random_state=42,
    # GPU-SPECIFIC PARAMETERS
    tree_method='gpu_hist' if USE_GPU else 'hist',  # Use GPU histogram algorithm
    device='cuda' if USE_GPU else 'cpu',  # Specify CUDA device
    predictor='gpu_predictor' if USE_GPU else 'auto',  # Use GPU for prediction
    # Additional GPU optimization
    max_bin=256,  # GPU works well with this bin size
    early_stopping_rounds=50
)

if USE_GPU:
    print("✓ GPU training enabled!")
    print("  - tree_method: gpu_hist")
    print("  - device: cuda")
    print("  - predictor: gpu_predictor")
else:
    print("⚠ GPU not available, using CPU")

import time
start_time = time.time()

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

training_time = time.time() - start_time
print(f"\n Training completed in {training_time:.2f} seconds")

# ============================================================================
# GET PREDICTIONS ON BOTH SETS
# ============================================================================
print("\n" + "="*80)
print("PREDICTIONS AND METRICS")
print("="*80)

# Predictions on training set
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

# Predictions on validation set
y_val_pred = model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)

print("\n📊 TRAINING SET METRICS:")
print("-" * 60)
print(f"R² Score:  {train_r2:.6f}")
print(f"RMSE:      {train_rmse:.4f}")
print(f"MAE:       {train_mae:.4f}")

print("\n📊 VALIDATION SET METRICS:")
print("-" * 60)
print(f"R² Score:  {val_r2:.6f}")
print(f"RMSE:      {val_rmse:.4f}")
print(f"MAE:       {val_mae:.4f}")

print("\n📊 OVERFITTING CHECK:")
print("-" * 60)
r2_diff = train_r2 - val_r2
print(f"R² Difference (Train - Val): {r2_diff:.6f}")
if r2_diff < 0.01:
    print("✓ Minimal overfitting - model generalizes well")
elif r2_diff < 0.05:
    print("✓ Slight overfitting - acceptable")
elif r2_diff < 0.10:
    print("⚠ Moderate overfitting - consider more regularization")
else:
    print("⚠ Significant overfitting - model needs regularization")

# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS")
print("="*80)

# Training set distribution
print("\n📊 TRAINING SET:")
print("-" * 60)
print(f"Actual Mean:      {y_train.mean():.2f}")
print(f"Predicted Mean:   {y_train_pred.mean():.2f}")
print(f"Actual Std:       {y_train.std():.2f}")
print(f"Predicted Std:    {y_train_pred.std():.2f}")
print(f"Actual Range:     [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"Predicted Range:  [{y_train_pred.min():.2f}, {y_train_pred.max():.2f}]")

# Validation set distribution
print("\n📊 VALIDATION SET:")
print("-" * 60)
print(f"Actual Mean:      {y_val.mean():.2f}")
print(f"Predicted Mean:   {y_val_pred.mean():.2f}")
print(f"Actual Std:       {y_val.std():.2f}")
print(f"Predicted Std:    {y_val_pred.std():.2f}")
print(f"Actual Range:     [{y_val.min():.2f}, {y_val.max():.2f}]")
print(f"Predicted Range:  [{y_val_pred.min():.2f}, {y_val_pred.max():.2f}]")

# Variance ratio check
print("\n📊 VARIANCE ANALYSIS:")
print("-" * 60)
train_var_ratio = y_train_pred.std() / y_train.std()
val_var_ratio = y_val_pred.std() / y_val.std()
print(f"Training Variance Ratio:   {train_var_ratio:.4f}")
print(f"Validation Variance Ratio: {val_var_ratio:.4f}")
if train_var_ratio < 0.7 or val_var_ratio < 0.7:
    print("⚠ WARNING: Model is compressing predictions (underfitting variance)")
    print("   → Predictions are too conservative")
    print("   → Consider reducing regularization")
elif train_var_ratio > 1.3 or val_var_ratio > 1.3:
    print("⚠ WARNING: Model is expanding predictions (overfitting variance)")
    print("   → Consider increasing regularization")
else:
    print("✓ Variance ratios look good")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('R² Analysis - Training vs Validation (GPU-Accelerated)', fontsize=16, fontweight='bold')

# Plot 1: Training Set - Actual vs Predicted
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.3, s=1)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Math Score', fontsize=11)
axes[0, 0].set_ylabel('Predicted Math Score', fontsize=11)
axes[0, 0].set_title(f'Training Set: R² = {train_r2:.4f}', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Validation Set - Actual vs Predicted
axes[0, 1].scatter(y_val, y_val_pred, alpha=0.3, s=1)
axes[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Math Score', fontsize=11)
axes[0, 1].set_ylabel('Predicted Math Score', fontsize=11)
axes[0, 1].set_title(f'Validation Set: R² = {val_r2:.4f}', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: R² Comparison Bar Chart
metrics_names = ['R² Score', 'RMSE', 'MAE']
train_metrics = [train_r2, train_rmse/100, train_mae/100]
val_metrics = [val_r2, val_rmse/100, val_mae/100]
x = np.arange(len(metrics_names))
width = 0.35
axes[0, 2].bar(x - width/2, train_metrics, width, label='Training', color='lightblue')
axes[0, 2].bar(x + width/2, val_metrics, width, label='Validation', color='lightcoral')
axes[0, 2].set_ylabel('Value', fontsize=11)
axes[0, 2].set_title('Metrics Comparison', fontsize=12, fontweight='bold')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(metrics_names)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')
axes[0, 2].text(0.5, 0.95, f'R²(Train)={train_r2:.4f}\nR²(Val)={val_r2:.4f}\nDiff={r2_diff:.4f}\nTime={training_time:.1f}s', 
                transform=axes[0, 2].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Training Set - Distribution Comparison
axes[1, 0].hist(y_train, bins=50, alpha=0.5, label='Actual', color='blue', density=True)
axes[1, 0].hist(y_train_pred, bins=50, alpha=0.5, label='Predicted', color='red', density=True)
axes[1, 0].axvline(y_train.mean(), color='blue', linestyle='--', linewidth=2)
axes[1, 0].axvline(y_train_pred.mean(), color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Math Score', fontsize=11)
axes[1, 0].set_ylabel('Density', fontsize=11)
axes[1, 0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Validation Set - Distribution Comparison
axes[1, 1].hist(y_val, bins=50, alpha=0.5, label='Actual', color='blue', density=True)
axes[1, 1].hist(y_val_pred, bins=50, alpha=0.5, label='Predicted', color='red', density=True)
axes[1, 1].axvline(y_val.mean(), color='blue', linestyle='--', linewidth=2)
axes[1, 1].axvline(y_val_pred.mean(), color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Math Score', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].set_title('Validation Set Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Residuals Analysis
train_residuals = y_train - y_train_pred
val_residuals = y_val - y_val_pred
axes[1, 2].scatter(y_train_pred, train_residuals, alpha=0.3, s=1, label='Training', color='blue')
axes[1, 2].scatter(y_val_pred, val_residuals, alpha=0.3, s=1, label='Validation', color='red')
axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Predicted Math Score', fontsize=11)
axes[1, 2].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
axes[1, 2].set_title('Residuals Plot', fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('r2_analysis_train_vs_val_gpu.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'r2_analysis_train_vs_val_gpu.png'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print(f"\n✓ Training R²:   {train_r2:.6f}")
print(f"✓ Validation R²: {val_r2:.6f}")
print(f"✓ R² Gap:        {r2_diff:.6f}")
print(f"✓ Training Time: {training_time:.2f} seconds")
if USE_GPU:
    print(f"✓ GPU Acceleration: ENABLED")

print("\n🎯 KEY FINDINGS:")
if val_r2 >= 0.75:
    print("  • Validation R² is GOOD (≥0.75)")
elif val_r2 >= 0.60:
    print("  • Validation R² is FAIR (0.60-0.75)")
else:
    print("  • Validation R² needs improvement (<0.60)")

if train_var_ratio < 0.7 or val_var_ratio < 0.7:
    print("  • ⚠ Model is compressing predictions (variance too low)")
    print("    → This explains why distribution looks compressed")
    print("    → SOLUTION: Reduce regularization parameters")

if r2_diff > 0.1:
    print(f"  • ⚠ Significant overfitting detected (gap: {r2_diff:.4f})")
    print("    → SOLUTION: Increase regularization or use more data")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# ============================================================================
# STEP 6: SAVE MODEL WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL WEIGHTS")
print("="*80)

# Save the model using XGBoost's native format
model.save_model('model_with_r2_analysis_gpu.json')
print("✓ Model saved as 'model_with_r2_analysis_gpu.json'")

# Also save using pickle
import pickle
with open('model_with_r2_analysis_gpu.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'model_with_r2_analysis_gpu.pkl'")

# Save model metadata
model_info = {
    'model_name': 'Deep Trees XGBoost (GPU-Accelerated)',
    'train_r2': train_r2,
    'validation_r2': val_r2,
    'train_rmse': train_rmse,
    'validation_rmse': val_rmse,
    'train_mae': train_mae,
    'validation_mae': val_mae,
    'r2_gap': r2_diff,
    'training_time': training_time,
    'gpu_enabled': USE_GPU,
    'features': top_features,
    'n_features': len(top_features),
    'model_params': model.get_params(),
    'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None,
    'train_variance_ratio': train_var_ratio,
    'val_variance_ratio': val_var_ratio,
    'numeric_imputer': numeric_imputer,
    'cat_imputer': cat_imputer,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'empty_cols_test': empty_cols_test,
    'cols_to_drop': cols_to_drop
}

with open('model_with_r2_analysis_gpu_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Model info saved as 'model_with_r2_analysis_gpu_info.pkl'")

print("\n📦 SAVED FILES:")
print("  1. model_with_r2_analysis_gpu.json       (XGBoost native format)")
print("  2. model_with_r2_analysis_gpu.pkl        (Pickle format)")
print("  3. model_with_r2_analysis_gpu_info.pkl   (Metadata & metrics)")

print("\n💡 TO LOAD THE MODEL LATER:")
print("  import xgboost as xgb")
print("  model = xgb.XGBRegressor()")
print("  model.load_model('model_with_r2_analysis_gpu.json')")

print("\n" + "="*80)
print("ALL DONE!")
print("="*80)