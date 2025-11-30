import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("XGBOOST ENSEMBLE METHODS - IMPROVING VALIDATION R²")
print("="*80)

# ============================================================================
# LOAD AND PREPROCESS DATA (Same as before)
# ============================================================================
print("\n[1/6] Loading data...")
X_train_full = pd.read_csv('data/X_train.csv')
y_train_full = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

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
    y_train_full = y_train_full[target_col]
else:
    y_train_full = y_train_full.iloc[:, 0]

if len(y_train_full) != len(X_train_full):
    min_len = min(len(X_train_full), len(y_train_full))
    X_train_full = X_train_full.iloc[:min_len]
    y_train_full = y_train_full.iloc[:min_len]

print(f"Data shapes - X: {X_train_full.shape}, y: {y_train_full.shape}")

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\n[2/6] Preprocessing...")

empty_cols_test = X_test.columns[X_test.isnull().all()].tolist()
X_train_full = X_train_full.drop(columns=empty_cols_test, errors='ignore')
X_test = X_test.drop(columns=empty_cols_test, errors='ignore')

numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object']).columns.tolist()

missing_threshold = 0.999
missing_pct = X_train_full.isnull().sum() / len(X_train_full)
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
X_train_full = X_train_full.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

numeric_cols = [col for col in numeric_cols if col in X_train_full.columns]
categorical_cols = [col for col in categorical_cols if col in X_train_full.columns]

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

numeric_imputer = SimpleImputer(strategy='median')
X_train_full[numeric_cols] = numeric_imputer.fit_transform(X_train_full[numeric_cols])
X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_full[categorical_cols] = cat_imputer.fit_transform(X_train_full[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

selector = VarianceThreshold(threshold=0.0)
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
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"Training: {X_train.shape}, Validation: {X_val.shape}")

# ============================================================================
# FEATURE SELECTION
# ============================================================================
print("\n[4/6] Selecting important features...")
quick_model = xgb.XGBRegressor(n_estimators=100, random_state=42, tree_method='hist')
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
# METHOD 1: SINGLE BASELINE MODEL (Your Current Approach)
# ============================================================================
print("\n" + "="*80)
print("[5/6] METHOD 1: SINGLE BASELINE MODEL")
print("="*80)

baseline_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=12,
    min_child_weight=3,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=0.5,
    reg_alpha=0.15,
    reg_lambda=2.5,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=50
)

baseline_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_val_pred_baseline = baseline_model.predict(X_val)
baseline_r2 = r2_score(y_val, y_val_pred_baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_baseline))

print(f"✓ Baseline Validation R²:  {baseline_r2:.6f}")
print(f"✓ Baseline Validation RMSE: {baseline_rmse:.4f}")

# ============================================================================
# METHOD 2: MULTIPLE XGBOOST MODELS WITH DIFFERENT CONFIGURATIONS
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: ENSEMBLE OF DIVERSE XGBOOST MODELS")
print("="*80)

# Define 5 different XGBoost configurations
model_configs = {
    'deep_trees': {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 12,
        'min_child_weight': 3,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'gamma': 0.5,
        'reg_alpha': 0.15,
        'reg_lambda': 2.5,
    },
    'shallow_trees': {
        'n_estimators': 1500,
        'learning_rate': 0.02,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 1.0,
        'reg_alpha': 0.5,
        'reg_lambda': 5.0,
    },
    'high_learning_rate': {
        'n_estimators': 500,
        'learning_rate': 0.08,
        'max_depth': 8,
        'min_child_weight': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.3,
        'reg_alpha': 0.3,
        'reg_lambda': 3.0,
    },
    'conservative': {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'min_child_weight': 8,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'gamma': 1.5,
        'reg_alpha': 1.0,
        'reg_lambda': 8.0,
    },
    'balanced': {
        'n_estimators': 1200,
        'learning_rate': 0.025,
        'max_depth': 9,
        'min_child_weight': 4,
        'subsample': 0.65,
        'colsample_bytree': 0.65,
        'gamma': 0.7,
        'reg_alpha': 0.4,
        'reg_lambda': 4.0,
    }
}

# Train all models
models = {}
predictions_val = {}

print("\nTraining 5 diverse XGBoost models...")
for name, config in model_configs.items():
    print(f"\n  Training '{name}' model...")
    model = xgb.XGBRegressor(
        **config,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models[name] = model
    predictions_val[name] = model.predict(X_val)
    
    r2 = r2_score(y_val, predictions_val[name])
    print(f"    ✓ {name}: R² = {r2:.6f}")

# ============================================================================
# ENSEMBLE STRATEGIES
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE STRATEGIES")
print("="*80)

# Strategy 1: Simple Average
y_val_pred_avg = np.mean([predictions_val[name] for name in model_configs.keys()], axis=0)
r2_avg = r2_score(y_val, y_val_pred_avg)
rmse_avg = np.sqrt(mean_squared_error(y_val, y_val_pred_avg))

print(f"\n1️⃣  SIMPLE AVERAGE:")
print(f"   Validation R²:  {r2_avg:.6f}")
print(f"   Validation RMSE: {rmse_avg:.4f}")
print(f"   Improvement over baseline: {(r2_avg - baseline_r2):.6f}")

# Strategy 2: Weighted Average (based on individual R² scores)
individual_r2s = {name: r2_score(y_val, predictions_val[name]) for name in model_configs.keys()}
total_r2 = sum(individual_r2s.values())
weights = {name: r2 / total_r2 for name, r2 in individual_r2s.items()}

y_val_pred_weighted = sum(weights[name] * predictions_val[name] for name in model_configs.keys())
r2_weighted = r2_score(y_val, y_val_pred_weighted)
rmse_weighted = np.sqrt(mean_squared_error(y_val, y_val_pred_weighted))

print(f"\n2️⃣  WEIGHTED AVERAGE (by R²):")
print(f"   Weights: {', '.join([f'{name}: {w:.3f}' for name, w in weights.items()])}")
print(f"   Validation R²:  {r2_weighted:.6f}")
print(f"   Validation RMSE: {rmse_weighted:.4f}")
print(f"   Improvement over baseline: {(r2_weighted - baseline_r2):.6f}")

# Strategy 3: Median (more robust to outliers)
y_val_pred_median = np.median([predictions_val[name] for name in model_configs.keys()], axis=0)
r2_median = r2_score(y_val, y_val_pred_median)
rmse_median = np.sqrt(mean_squared_error(y_val, y_val_pred_median))

print(f"\n3️⃣  MEDIAN:")
print(f"   Validation R²:  {r2_median:.6f}")
print(f"   Validation RMSE: {rmse_median:.4f}")
print(f"   Improvement over baseline: {(r2_median - baseline_r2):.6f}")

# Strategy 4: Best 3 models only
top_3_models = sorted(individual_r2s.items(), key=lambda x: x[1], reverse=True)[:3]
y_val_pred_top3 = np.mean([predictions_val[name] for name, _ in top_3_models], axis=0)
r2_top3 = r2_score(y_val, y_val_pred_top3)
rmse_top3 = np.sqrt(mean_squared_error(y_val, y_val_pred_top3))

print(f"\n4️⃣  TOP 3 MODELS AVERAGE:")
print(f"   Models: {', '.join([name for name, _ in top_3_models])}")
print(f"   Validation R²:  {r2_top3:.6f}")
print(f"   Validation RMSE: {rmse_top3:.4f}")
print(f"   Improvement over baseline: {(r2_top3 - baseline_r2):.6f}")

# ============================================================================
# METHOD 3: BAGGING - TRAIN SAME MODEL ON DIFFERENT DATA SUBSETS
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: BAGGING (Bootstrap Aggregating)")
print("="*80)

n_bags = 5
bag_predictions = []

print(f"\nTraining {n_bags} models on bootstrapped samples...")
for i in range(n_bags):
    # Bootstrap sample
    indices = np.random.RandomState(i).choice(len(X_train), size=len(X_train), replace=True)
    X_train_bag = X_train.iloc[indices]
    y_train_bag = y_train.iloc[indices]
    
    # Train model
    bag_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=10,
        min_child_weight=4,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.7,
        reg_alpha=0.3,
        reg_lambda=4.0,
        random_state=i,
        tree_method='hist',
        early_stopping_rounds=50
    )
    bag_model.fit(X_train_bag, y_train_bag, eval_set=[(X_val, y_val)], verbose=False)
    bag_predictions.append(bag_model.predict(X_val))
    print(f"  ✓ Bag {i+1}/{n_bags} complete")

y_val_pred_bagging = np.mean(bag_predictions, axis=0)
r2_bagging = r2_score(y_val, y_val_pred_bagging)
rmse_bagging = np.sqrt(mean_squared_error(y_val, y_val_pred_bagging))

print(f"\n   Validation R²:  {r2_bagging:.6f}")
print(f"   Validation RMSE: {rmse_bagging:.4f}")
print(f"   Improvement over baseline: {(r2_bagging - baseline_r2):.6f}")

# ============================================================================
# METHOD 4: STACKING - TRAIN META-MODEL ON PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("METHOD 4: STACKING (Meta-Model)")
print("="*80)

from sklearn.linear_model import Ridge

# Create stacking features (predictions from base models)
stacking_features_val = np.column_stack([predictions_val[name] for name in model_configs.keys()])

# Split validation set for meta-model training
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
    stacking_features_val, y_val, test_size=0.5, random_state=42
)

# Train meta-model (Ridge regression)
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train, y_meta_train)

# Predict
y_val_pred_stacking = meta_model.predict(X_meta_val)
r2_stacking = r2_score(y_meta_val, y_val_pred_stacking)
rmse_stacking = np.sqrt(mean_squared_error(y_meta_val, y_val_pred_stacking))

print(f"\n   Meta-model weights: {meta_model.coef_}")
print(f"   Validation R²:  {r2_stacking:.6f}")
print(f"   Validation RMSE: {rmse_stacking:.4f}")
print(f"   Improvement over baseline: {(r2_stacking - baseline_r2):.6f}")

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================
print("\n" + "="*80)
print("[6/6] FINAL COMPARISON")
print("="*80)

results = {
    'Baseline (Single Model)': baseline_r2,
    'Ensemble - Simple Average': r2_avg,
    'Ensemble - Weighted Average': r2_weighted,
    'Ensemble - Median': r2_median,
    'Ensemble - Top 3 Models': r2_top3,
    'Bagging (5 models)': r2_bagging,
    'Stacking (Meta-model)': r2_stacking,
}

results_df = pd.DataFrame(list(results.items()), columns=['Method', 'Validation R²'])
results_df = results_df.sort_values('Validation R²', ascending=False)
results_df['Improvement'] = results_df['Validation R²'] - baseline_r2

print("\n📊 RESULTS RANKING:")
print("=" * 70)
for idx, row in results_df.iterrows():
    improvement_str = f"+{row['Improvement']:.6f}" if row['Improvement'] >= 0 else f"{row['Improvement']:.6f}"
    print(f"{row['Method']:30s}  R² = {row['Validation R²']:.6f}  ({improvement_str})")

best_method = results_df.iloc[0]['Method']
best_r2 = results_df.iloc[0]['Validation R²']
best_improvement = results_df.iloc[0]['Improvement']

print("\n" + "="*80)
print("🏆 WINNER")
print("="*80)
print(f"Best Method: {best_method}")
print(f"Validation R²: {best_r2:.6f}")
print(f"Improvement: +{best_improvement:.6f} ({best_improvement/baseline_r2*100:.2f}%)")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Ensemble Methods Comparison', fontsize=16, fontweight='bold')

# Plot 1: R² Comparison
methods = results_df['Method'].tolist()
r2_scores = results_df['Validation R²'].tolist()
colors = ['red' if m == 'Baseline (Single Model)' else 'lightblue' for m in methods]

axes[0, 0].barh(methods, r2_scores, color=colors)
axes[0, 0].set_xlabel('Validation R²', fontsize=12)
axes[0, 0].set_title('Validation R² by Method', fontsize=13, fontweight='bold')
axes[0, 0].axvline(baseline_r2, color='red', linestyle='--', linewidth=2, label='Baseline')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Plot 2: Individual Model Performance
individual_names = list(individual_r2s.keys())
individual_scores = list(individual_r2s.values())
axes[0, 1].bar(individual_names, individual_scores, color='lightgreen')
axes[0, 1].axhline(baseline_r2, color='red', linestyle='--', linewidth=2, label='Baseline')
axes[0, 1].set_ylabel('Validation R²', fontsize=12)
axes[0, 1].set_title('Individual Model Performance', fontsize=13, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Predictions Comparison (Best Ensemble vs Baseline)
if best_method == 'Ensemble - Simple Average':
    best_preds = y_val_pred_avg
elif best_method == 'Ensemble - Weighted Average':
    best_preds = y_val_pred_weighted
elif best_method == 'Ensemble - Top 3 Models':
    best_preds = y_val_pred_top3
elif best_method == 'Bagging (5 models)':
    best_preds = y_val_pred_bagging
else:
    best_preds = y_val_pred_avg

axes[1, 0].scatter(y_val, y_val_pred_baseline, alpha=0.3, s=1, label='Baseline', color='red')
axes[1, 0].scatter(y_val, best_preds, alpha=0.3, s=1, label='Best Ensemble', color='blue')
axes[1, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', linewidth=2)
axes[1, 0].set_xlabel('Actual', fontsize=12)
axes[1, 0].set_ylabel('Predicted', fontsize=12)
axes[1, 0].set_title('Predictions: Baseline vs Best Ensemble', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Improvement Bar Chart
improvements = results_df['Improvement'].tolist()
colors = ['green' if imp > 0 else 'red' for imp in improvements]
axes[1, 1].barh(methods, improvements, color=colors)
axes[1, 1].set_xlabel('R² Improvement over Baseline', fontsize=12)
axes[1, 1].set_title('Improvement Analysis', fontsize=13, fontweight='bold')
axes[1, 1].axvline(0, color='black', linewidth=2)
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('xgboost_ensemble_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'xgboost_ensemble_comparison.png'")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("💡 RECOMMENDATIONS")
print("="*80)

if best_improvement > 0.01:
    print(f"\n✅ ENSEMBLE METHODS WORK!")
    print(f"   Your validation R² improved from {baseline_r2:.4f} to {best_r2:.4f}")
    print(f"   Gain: +{best_improvement:.4f} ({best_improvement/baseline_r2*100:.2f}%)")
    print(f"\n🎯 Use '{best_method}' for your final predictions")
elif best_improvement > 0:
    print(f"\n🟡 SMALL IMPROVEMENT")
    print(f"   Ensembling helps slightly: +{best_improvement:.4f}")
    print(f"   Consider combining with other techniques (feature engineering, etc.)")
else:
    print(f"\n⚠️  ENSEMBLE NOT HELPING")
    print(f"   Your baseline model is already well-optimized")
    print(f"   Focus on: feature engineering, regularization, or more data")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)