import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print("="*80)
print("CHECKING GPU AVAILABILITY")
print("="*80)
try:
    import cupy as cp
    gpu_available = True
    print("✓ GPU detected! Training will use GPU acceleration")
    print(f"  Device: {cp.cuda.Device()}")
    TREE_METHOD = 'gpu_hist'
    PREDICTOR = 'gpu_predictor'
except ImportError:
    gpu_available = False
    print("⚠ GPU not available. Install cupy for GPU support:")
    print("  pip install cupy-cuda11x  # For CUDA 11.x")
    print("  pip install cupy-cuda12x  # For CUDA 12.x")
    print("\nFalling back to CPU training...")
    TREE_METHOD = 'hist'
    PREDICTOR = 'cpu_predictor'

print("="*80)
print("TWO-STAGE HURDLE MODEL FOR ZERO-INFLATED DATA")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'validation_size': 0.2,
    'random_state': 42,
    'variance_threshold': 0.0,
    'missing_threshold': 0.9999,
    'top_n_features': 200,
}

# Stage 1: Binary Classification (Zero vs Non-Zero)
STAGE1_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.3,
    'reg_lambda': 2,
    'random_state': 42,
    'tree_method': TREE_METHOD,
    'predictor': PREDICTOR,
    'eval_metric': 'logloss'
}

# Stage 2: Regression (Predicting Non-Zero Scores)
STAGE2_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 12,
    'min_child_weight': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.5,
    'reg_alpha': 0.5,
    'reg_lambda': 2,
  
    'random_state': 42,
    'tree_method': TREE_METHOD,
    'predictor': PREDICTOR
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
X_train_full = pd.read_csv('data/X_train.csv')
y_train_full = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

print(f"X_train shape: {X_train_full.shape}")
print(f"y_train shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}")

# Handle y_train
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

# Analyze zero distribution
n_zeros = (y_train_full == 0).sum()
n_nonzeros = (y_train_full > 0).sum()
pct_zeros = n_zeros / len(y_train_full) * 100

print(f"\n📊 TARGET DISTRIBUTION:")
print(f"  Total samples: {len(y_train_full)}")
print(f"  Zero scores: {n_zeros} ({pct_zeros:.2f}%)")
print(f"  Non-zero scores: {n_nonzeros} ({100-pct_zeros:.2f}%)")
print(f"  Mean (all): {y_train_full.mean():.2f}")
print(f"  Mean (non-zero only): {y_train_full[y_train_full > 0].mean():.2f}")

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\n[2/6] Preprocessing...")

# Remove empty columns
empty_cols_test = X_test.columns[X_test.isnull().all()].tolist()
X_train_full = X_train_full.drop(columns=empty_cols_test, errors='ignore')
X_test = X_test.drop(columns=empty_cols_test, errors='ignore')

# Identify column types
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object']).columns.tolist()

# Remove high missing columns
missing_pct = X_train_full.isnull().sum() / len(X_train_full)
cols_to_drop = missing_pct[missing_pct > CONFIG['missing_threshold']].index.tolist()
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
selector = VarianceThreshold(threshold=CONFIG['variance_threshold'])
X_train_selected = selector.fit_transform(X_train_full)
X_test_selected = selector.transform(X_test)
selected_features = X_train_full.columns[selector.get_support()].tolist()
X_train_full = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)

print(f"Features after preprocessing: {len(selected_features)}")

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================
print("\n[3/6] Creating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=CONFIG['validation_size'],
    random_state=CONFIG['random_state']
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# ============================================================================
# FEATURE SELECTION
# ============================================================================
print("\n[4/6] Selecting important features...")
print("Training quick model for feature importance...")
quick_model = xgb.XGBRegressor(n_estimators=100, random_state=42, tree_method=TREE_METHOD, predictor=PREDICTOR)

with tqdm(total=100, desc="Feature Selection", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    class TqdmCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False
    
    quick_model.fit(X_train, y_train, callbacks=[TqdmCallback()])

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': quick_model.feature_importances_
}).sort_values('importance', ascending=False)

top_n_features = min(CONFIG['top_n_features'], len(X_train.columns))
top_features = feature_importance.head(top_n_features)['feature'].tolist()

X_train = X_train[top_features]
X_val = X_val[top_features]
X_train_full = X_train_full[top_features]
X_test = X_test[top_features]

print(f"✓ Using top {len(top_features)} features")

# ============================================================================
# STAGE 1: TRAIN BINARY CLASSIFIER (ZERO vs NON-ZERO)
# ============================================================================
print("\n[5/6] STAGE 1: Training binary classifier (Zero vs Non-Zero)...")
print("="*80)

# Create binary labels
y_train_binary = (y_train > 0).astype(int)  # 0 = zero score, 1 = non-zero score
y_val_binary = (y_val > 0).astype(int)

print(f"Training set distribution:")
print(f"  Zeros: {(y_train_binary == 0).sum()} ({(y_train_binary == 0).sum()/len(y_train_binary)*100:.2f}%)")
print(f"  Non-zeros: {(y_train_binary == 1).sum()} ({(y_train_binary == 1).sum()/len(y_train_binary)*100:.2f}%)")

# Train Stage 1 classifier
print("\nTraining Stage 1 Classifier...")
stage1_model = xgb.XGBClassifier(**STAGE1_PARAMS)

start_time = time.time()
with tqdm(total=STAGE1_PARAMS['n_estimators'], desc="Stage 1 Training", 
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    class TqdmCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False
    
    stage1_model.fit(
        X_train, y_train_binary,
        eval_set=[(X_val, y_val_binary)],
        callbacks=[TqdmCallback()],
        verbose=False
    )
elapsed = time.time() - start_time
print(f"✓ Stage 1 training completed in {elapsed:.2f} seconds")

# Evaluate Stage 1
y_train_binary_pred = stage1_model.predict(X_train)
y_val_binary_pred = stage1_model.predict(X_val)
y_val_binary_proba = stage1_model.predict_proba(X_val)[:, 1]

print("\n📊 STAGE 1 METRICS (Binary Classification):")
print("-" * 60)
print(f"Training Accuracy:   {accuracy_score(y_train_binary, y_train_binary_pred):.4f}")
print(f"Validation Accuracy: {accuracy_score(y_val_binary, y_val_binary_pred):.4f}")
print(f"Validation Precision: {precision_score(y_val_binary, y_val_binary_pred):.4f}")
print(f"Validation Recall:    {recall_score(y_val_binary, y_val_binary_pred):.4f}")
print(f"Validation F1:        {f1_score(y_val_binary, y_val_binary_pred):.4f}")
print(f"Validation ROC-AUC:   {roc_auc_score(y_val_binary, y_val_binary_proba):.4f}")

# ============================================================================
# STAGE 2: TRAIN REGRESSOR (NON-ZERO SCORES ONLY)
# ============================================================================
print("\n[5/6] STAGE 2: Training regressor on non-zero scores only...")
print("="*80)

# Filter to only non-zero scores
X_train_nonzero = X_train[y_train > 0]
y_train_nonzero = y_train[y_train > 0]

X_val_nonzero = X_val[y_val > 0]
y_val_nonzero = y_val[y_val > 0]

print(f"Non-zero training samples: {len(X_train_nonzero)} (from {len(X_train)} total)")
print(f"Non-zero validation samples: {len(X_val_nonzero)} (from {len(X_val)} total)")
print(f"\nNon-zero target stats:")
print(f"  Mean: {y_train_nonzero.mean():.2f}")
print(f"  Std: {y_train_nonzero.std():.2f}")
print(f"  Min: {y_train_nonzero.min():.2f}")
print(f"  Max: {y_train_nonzero.max():.2f}")

# Train Stage 2 regressor
print("\nTraining Stage 2 Regressor...")
stage2_model = xgb.XGBRegressor(**STAGE2_PARAMS)

start_time = time.time()
with tqdm(total=STAGE2_PARAMS['n_estimators'], desc="Stage 2 Training", 
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    class TqdmCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False
    
    stage2_model.fit(
        X_train_nonzero, y_train_nonzero,
        eval_set=[(X_val_nonzero, y_val_nonzero)],
        callbacks=[TqdmCallback()],
        verbose=False
    )
elapsed = time.time() - start_time
print(f"✓ Stage 2 training completed in {elapsed:.2f} seconds")

# Evaluate Stage 2 (on non-zero subset only)
y_train_nonzero_pred = stage2_model.predict(X_train_nonzero)
y_val_nonzero_pred = stage2_model.predict(X_val_nonzero)

stage2_train_r2 = r2_score(y_train_nonzero, y_train_nonzero_pred)
stage2_val_r2 = r2_score(y_val_nonzero, y_val_nonzero_pred)
stage2_val_rmse = np.sqrt(mean_squared_error(y_val_nonzero, y_val_nonzero_pred))
stage2_val_mae = mean_absolute_error(y_val_nonzero, y_val_nonzero_pred)

print("\n📊 STAGE 2 METRICS (Regression on Non-Zero Scores):")
print("-" * 60)
print(f"Training R²:   {stage2_train_r2:.4f}")
print(f"Validation R²: {stage2_val_r2:.4f}")
print(f"Validation RMSE: {stage2_val_rmse:.4f}")
print(f"Validation MAE:  {stage2_val_mae:.4f}")

# ============================================================================
# COMBINED MODEL EVALUATION
# ============================================================================
print("\n[6/6] Evaluating combined two-stage model...")
print("="*80)

def two_stage_predict(X, stage1_model, stage2_model):
    """
    Two-stage prediction:
    1. Predict if score will be zero or non-zero
    2. If non-zero predicted, predict the actual score
    """
    # Stage 1: Predict zero vs non-zero
    is_nonzero = stage1_model.predict(X)
    
    # Stage 2: Predict scores (only for non-zero predictions)
    predictions = np.zeros(len(X))
    nonzero_mask = is_nonzero == 1
    
    if nonzero_mask.sum() > 0:
        predictions[nonzero_mask] = stage2_model.predict(X[nonzero_mask])
        # Ensure no negative predictions
        predictions = np.maximum(predictions, 0)
    
    return predictions

# Get predictions
y_train_pred_combined = two_stage_predict(X_train, stage1_model, stage2_model)
y_val_pred_combined = two_stage_predict(X_val, stage1_model, stage2_model)

# Calculate metrics
combined_train_r2 = r2_score(y_train, y_train_pred_combined)
combined_val_r2 = r2_score(y_val, y_val_pred_combined)
combined_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_combined))
combined_val_mae = mean_absolute_error(y_val, y_val_pred_combined)

print("\n📊 COMBINED TWO-STAGE MODEL METRICS:")
print("-" * 60)
print(f"Training R²:   {combined_train_r2:.6f}")
print(f"Validation R²: {combined_val_r2:.6f}")
print(f"Validation RMSE: {combined_val_rmse:.4f}")
print(f"Validation MAE:  {combined_val_mae:.4f}")

# Compare with baseline (single model)
print("\n📊 COMPARISON WITH BASELINE (Single XGBoost):")
print("-" * 60)
print("Training baseline model...")
baseline_model = xgb.XGBRegressor(**{**STAGE2_PARAMS, 'n_estimators': 500})  # Fewer trees for speed

start_time = time.time()
with tqdm(total=500, desc="Baseline Training", 
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    class TqdmCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False
    
    baseline_model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[TqdmCallback()],
        verbose=False
    )
elapsed = time.time() - start_time
print(f"✓ Baseline training completed in {elapsed:.2f} seconds")

y_val_pred_baseline = baseline_model.predict(X_val)
baseline_val_r2 = r2_score(y_val, y_val_pred_baseline)
baseline_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_baseline))
baseline_val_mae = mean_absolute_error(y_val, y_val_pred_baseline)

print(f"Baseline R²:   {baseline_val_r2:.6f}")
print(f"Baseline RMSE: {baseline_val_rmse:.4f}")
print(f"Baseline MAE:  {baseline_val_mae:.4f}")
print(f"\nImprovement:")
print(f"  R² gain:   {(combined_val_r2 - baseline_val_r2):.6f} ({(combined_val_r2/baseline_val_r2 - 1)*100:+.2f}%)")
print(f"  RMSE gain: {(baseline_val_rmse - combined_val_rmse):.4f} ({(1 - combined_val_rmse/baseline_val_rmse)*100:+.2f}%)")
print(f"  MAE gain:  {(baseline_val_mae - combined_val_mae):.4f} ({(1 - combined_val_mae/baseline_val_mae)*100:+.2f}%)")

# Analyze predictions on zeros specifically
zeros_mask_val = y_val == 0
nonzeros_mask_val = y_val > 0

print("\n📊 ZERO PREDICTION ANALYSIS:")
print("-" * 60)
print(f"Actual zeros in validation: {zeros_mask_val.sum()}")
print(f"Predicted zeros (two-stage): {(y_val_pred_combined == 0).sum()}")
print(f"Predicted zeros (baseline): {(y_val_pred_baseline < 0.5).sum()}")

if zeros_mask_val.sum() > 0:
    mae_on_zeros_combined = mean_absolute_error(y_val[zeros_mask_val], y_val_pred_combined[zeros_mask_val])
    mae_on_zeros_baseline = mean_absolute_error(y_val[zeros_mask_val], y_val_pred_baseline[zeros_mask_val])
    print(f"\nMAE on actual zeros:")
    print(f"  Two-stage: {mae_on_zeros_combined:.4f}")
    print(f"  Baseline:  {mae_on_zeros_baseline:.4f}")
    print(f"  Improvement: {(mae_on_zeros_baseline - mae_on_zeros_combined):.4f}")

if nonzeros_mask_val.sum() > 0:
    mae_on_nonzeros_combined = mean_absolute_error(y_val[nonzeros_mask_val], y_val_pred_combined[nonzeros_mask_val])
    mae_on_nonzeros_baseline = mean_absolute_error(y_val[nonzeros_mask_val], y_val_pred_baseline[nonzeros_mask_val])
    print(f"\nMAE on actual non-zeros:")
    print(f"  Two-stage: {mae_on_nonzeros_combined:.4f}")
    print(f"  Baseline:  {mae_on_nonzeros_baseline:.4f}")
    print(f"  Improvement: {(mae_on_nonzeros_baseline - mae_on_nonzeros_combined):.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Two-Stage Hurdle Model Analysis', fontsize=16, fontweight='bold')

# Plot 1: Two-Stage Model - Actual vs Predicted
axes[0, 0].scatter(y_val, y_val_pred_combined, alpha=0.3, s=1, label='Two-Stage')
axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                'r--', linewidth=2, label='Perfect')
axes[0, 0].set_xlabel('Actual Score')
axes[0, 0].set_ylabel('Predicted Score')
axes[0, 0].set_title(f'Two-Stage Model: R² = {combined_val_r2:.4f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Baseline Model - Actual vs Predicted
axes[0, 1].scatter(y_val, y_val_pred_baseline, alpha=0.3, s=1, label='Baseline', color='orange')
axes[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                'r--', linewidth=2, label='Perfect')
axes[0, 1].set_xlabel('Actual Score')
axes[0, 1].set_ylabel('Predicted Score')
axes[0, 1].set_title(f'Baseline Model: R² = {baseline_val_r2:.4f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: R² Comparison
models = ['Two-Stage', 'Baseline']
r2_scores = [combined_val_r2, baseline_val_r2]
colors = ['green', 'orange']
bars = axes[0, 2].bar(models, r2_scores, color=colors, alpha=0.7)
axes[0, 2].set_ylabel('R² Score')
axes[0, 2].set_title('Model Comparison')
axes[0, 2].set_ylim([min(r2_scores) - 0.05, max(r2_scores) + 0.05])
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Plot 4: Distribution of predictions
axes[1, 0].hist(y_val, bins=50, alpha=0.5, label='Actual', color='blue', density=True)
axes[1, 0].hist(y_val_pred_combined, bins=50, alpha=0.5, label='Two-Stage', color='green', density=True)
axes[1, 0].hist(y_val_pred_baseline, bins=50, alpha=0.5, label='Baseline', color='orange', density=True)
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Score Distribution Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Residuals - Two-Stage
residuals_combined = y_val - y_val_pred_combined
axes[1, 1].scatter(y_val_pred_combined, residuals_combined, alpha=0.3, s=1, color='green')
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Score')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Two-Stage Residuals')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Residuals - Baseline
residuals_baseline = y_val - y_val_pred_baseline
axes[1, 2].scatter(y_val_pred_baseline, residuals_baseline, alpha=0.3, s=1, color='orange')
axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Predicted Score')
axes[1, 2].set_ylabel('Residuals')
axes[1, 2].set_title('Baseline Residuals')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('two_stage_model_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'two_stage_model_analysis.png'")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save both stage models
stage1_model.save_model('stage1_classifier.json')
stage2_model.save_model('stage2_regressor.json')
print("✓ Stage 1 model saved as 'stage1_classifier.json'")
print("✓ Stage 2 model saved as 'stage2_regressor.json'")

# Save complete package
two_stage_package = {
    'stage1_model': stage1_model,
    'stage2_model': stage2_model,
    'stage1_params': STAGE1_PARAMS,
    'stage2_params': STAGE2_PARAMS,
    'features': top_features,
    'n_features': len(top_features),
    'preprocessing': {
        'numeric_imputer': numeric_imputer,
        'cat_imputer': cat_imputer,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'variance_selector': selector,
        'empty_cols_test': empty_cols_test,
        'cols_to_drop': cols_to_drop
    },
    'metrics': {
        'combined_train_r2': combined_train_r2,
        'combined_val_r2': combined_val_r2,
        'combined_val_rmse': combined_val_rmse,
        'combined_val_mae': combined_val_mae,
        'baseline_val_r2': baseline_val_r2,
        'stage1_val_accuracy': accuracy_score(y_val_binary, y_val_binary_pred),
        'stage1_val_roc_auc': roc_auc_score(y_val_binary, y_val_binary_proba),
        'stage2_val_r2': stage2_val_r2
    }
}

with open('two_stage_model_package.pkl', 'wb') as f:
    pickle.dump(two_stage_package, f)
print("✓ Complete package saved as 'two_stage_model_package.pkl'")

print("\n💡 TO MAKE PREDICTIONS:")
print("  import pickle")
print("  ")
print("  # Load package")
print("  with open('two_stage_model_package.pkl', 'rb') as f:")
print("      package = pickle.load(f)")
print("  ")
print("  # Get models")
print("  stage1 = package['stage1_model']")
print("  stage2 = package['stage2_model']")
print("  features = package['features']")
print("  ")
print("  # Make predictions")
print("  is_nonzero = stage1.predict(X_test[features])")
print("  predictions = np.zeros(len(X_test))")
print("  nonzero_mask = is_nonzero == 1")
print("  predictions[nonzero_mask] = stage2.predict(X_test[features][nonzero_mask])")

print("\n" + "="*80)
print("TWO-STAGE MODEL TRAINING COMPLETE!")
print("="*80)





