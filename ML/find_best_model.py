import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import warnings
from itertools import product
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY THESE TO CONTROL YOUR EXPERIMENTS
# ============================================================================

CONFIG = {
    # Data sampling - set to smaller values for faster experimentation
    'data_fraction': 0.01,  # Use 100% of data (set to 0.1 for 10%, 0.5 for 50%, etc.)
    'validation_size': 0.2,  # 20% validation split
    'random_state': 42,
    
    # Feature selection
    'variance_threshold': 0.01,
    'missing_threshold': 0.8,
    'top_n_features': 200,
    
    # Model evaluation
    'early_stopping_rounds': 50,
    'verbose': False,
}

# ============================================================================
# PREPROCESSING CONFIGURATIONS TO TEST
# ============================================================================

PREPROCESSING_CONFIGS = [
    {
        'name': 'Standard',
        'variance_threshold': 0.01,
        'missing_threshold': 0.8,
        'numeric_imputer': 'median',
        'categorical_imputer': 'most_frequent',
        'top_n_features': 200
    },
    {
        'name': 'Aggressive_Feature_Selection',
        'variance_threshold': 0.05,
        'missing_threshold': 0.7,
        'numeric_imputer': 'median',
        'categorical_imputer': 'most_frequent',
        'top_n_features': 100
    },
    {
        'name': 'Conservative',
        'variance_threshold': 0.001,
        'missing_threshold': 0.9,
        'numeric_imputer': 'mean',
        'categorical_imputer': 'most_frequent',
        'top_n_features': 300
    },
]

# ============================================================================
# HYPERPARAMETER CONFIGURATIONS TO TEST
# ============================================================================

HYPERPARAMETER_CONFIGS = [
    {
        'name': 'Baseline',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
    },
    {
        'name': 'Deep_Trees',
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 12,
        'min_child_weight': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 2,
    },
    {
        'name': 'High_Regularization',
        'n_estimators': 800,
        'learning_rate': 0.04,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'gamma': 1,
        'reg_alpha': 1,
        'reg_lambda': 3,
    },
    {
        'name': 'Fast_Learning',
        'n_estimators': 300,
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.3,
        'reg_alpha': 0.3,
        'reg_lambda': 1.5,
    },
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_prepare_data(data_fraction=1.0, random_state=42):
    """Load data and optionally sample a fraction of it"""
    print(f"\n[DATA] Loading data (using {data_fraction*100:.1f}% of training data)...")
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
    
    # Sample data if requested
    if data_fraction < 1.0:
        n_samples = int(len(X_train_full) * data_fraction)
        indices = np.random.RandomState(random_state).choice(len(X_train_full), n_samples, replace=False)
        X_train_full = X_train_full.iloc[indices].reset_index(drop=True)
        y_train_full = y_train_full.iloc[indices].reset_index(drop=True)
        print(f"  → Sampled {n_samples} rows from training data")
    
    print(f"  X_train: {X_train_full.shape}, y_train: {y_train_full.shape}, X_test: {X_test.shape}")
    return X_train_full, y_train_full, X_test


def preprocess_data(X_train, X_test, prep_config):
    """Apply preprocessing based on configuration"""
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Remove empty columns
    empty_cols = X_test.columns[X_test.isnull().all()].tolist()
    X_train = X_train.drop(columns=empty_cols, errors='ignore')
    X_test = X_test.drop(columns=empty_cols, errors='ignore')
    
    # Identify column types
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Remove high missing columns
    missing_pct = X_train.isnull().sum() / len(X_train)
    cols_to_drop = missing_pct[missing_pct > prep_config['missing_threshold']].index.tolist()
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop, errors='ignore')
    
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    
    # Encode categorical
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Impute
    numeric_imputer = SimpleImputer(strategy=prep_config['numeric_imputer'])
    X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])
    
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy=prep_config['categorical_imputer'])
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
    
    # Variance filtering
    selector = VarianceThreshold(threshold=prep_config['variance_threshold'])
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()].tolist()
    X_train = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test = pd.DataFrame(X_test_selected, columns=selected_features)
    
    return X_train, X_test


def select_features(X_train, y_train, X_val, X_test, top_n):
    """Select top N features based on importance"""
    quick_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    quick_model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': quick_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_n_features = min(top_n, len(X_train.columns))
    top_features = feature_importance.head(top_n_features)['feature'].tolist()
    
    return X_train[top_features], X_val[top_features], X_test[top_features], top_features


def train_and_evaluate(X_train, y_train, X_val, y_val, hyper_config, early_stopping_rounds):
    """Train model with given hyperparameters and return metrics"""
    model = xgb.XGBRegressor(
        n_estimators=hyper_config['n_estimators'],
        learning_rate=hyper_config['learning_rate'],
        max_depth=hyper_config['max_depth'],
        min_child_weight=hyper_config['min_child_weight'],
        subsample=hyper_config['subsample'],
        colsample_bytree=hyper_config['colsample_bytree'],
        gamma=hyper_config['gamma'],
        reg_alpha=hyper_config['reg_alpha'],
        reg_lambda=hyper_config['reg_lambda'],
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    results = {
        'train_r2': r2_score(y_train, y_train_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'r2_gap': r2_score(y_train, y_train_pred) - r2_score(y_val, y_val_pred),
        'train_var_ratio': y_train_pred.std() / y_train.std(),
        'val_var_ratio': y_val_pred.std() / y_val.std(),
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None,
        'model': model
    }
    
    return results


# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def run_grid_search():
    """Run grid search over all preprocessing and hyperparameter combinations"""
    print("="*80)
    print("HYPERPARAMETER & PREPROCESSING GRID SEARCH")
    print("="*80)
    
    # Load data once
    X_train_full, y_train_full, X_test = load_and_prepare_data(
        data_fraction=CONFIG['data_fraction'],
        random_state=CONFIG['random_state']
    )
    
    # Store all results
    all_results = []
    best_val_r2 = -np.inf
    best_config = None
    best_model = None
    
    # Total combinations
    total_combos = len(PREPROCESSING_CONFIGS) * len(HYPERPARAMETER_CONFIGS)
    current_combo = 0
    
    print(f"\n[SEARCH] Testing {total_combos} combinations...")
    print(f"  → {len(PREPROCESSING_CONFIGS)} preprocessing configs")
    print(f"  → {len(HYPERPARAMETER_CONFIGS)} hyperparameter configs")
    print("="*80 + "\n")
    
    # Grid search
    for prep_config in PREPROCESSING_CONFIGS:
        print(f"\n{'='*80}")
        print(f"PREPROCESSING: {prep_config['name']}")
        print(f"{'='*80}")
        
        # Preprocess data
        X_train_prep, X_test_prep = preprocess_data(X_train_full, X_test, prep_config)
        
        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_prep, y_train_full,
            test_size=CONFIG['validation_size'],
            random_state=CONFIG['random_state']
        )
        
        # Feature selection
        X_train_fs, X_val_fs, X_test_fs, selected_features = select_features(
            X_train, y_train, X_val, X_test_prep, prep_config['top_n_features']
        )
        
        print(f"\n  Features after preprocessing: {len(selected_features)}")
        print(f"  Train: {X_train_fs.shape}, Val: {X_val_fs.shape}")
        
        for hyper_config in HYPERPARAMETER_CONFIGS:
            current_combo += 1
            print(f"\n  [{current_combo}/{total_combos}] Testing: {hyper_config['name']}")
            
            # Train and evaluate
            results = train_and_evaluate(
                X_train_fs, y_train, X_val_fs, y_val,
                hyper_config, CONFIG['early_stopping_rounds']
            )
            
            # Store results
            result_entry = {
                'prep_name': prep_config['name'],
                'hyper_name': hyper_config['name'],
                'prep_config': prep_config,
                'hyper_config': hyper_config,
                'n_features': len(selected_features),
                **results
            }
            all_results.append(result_entry)
            
            # Print summary
            print(f"      Train R²: {results['train_r2']:.4f} | Val R²: {results['val_r2']:.4f} | Gap: {results['r2_gap']:.4f}")
            
            # Track best model
            if results['val_r2'] > best_val_r2:
                best_val_r2 = results['val_r2']
                best_config = result_entry
                best_model = results['model']
                print(f"      ⭐ NEW BEST! Val R²: {best_val_r2:.4f}")
    
    return all_results, best_config, best_model


def print_results_summary(all_results, best_config):
    """Print comprehensive results summary"""
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    
    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame([{
        'Preprocessing': r['prep_name'],
        'Hyperparameters': r['hyper_name'],
        'Features': r['n_features'],
        'Train_R2': r['train_r2'],
        'Val_R2': r['val_r2'],
        'Val_RMSE': r['val_rmse'],
        'Val_MAE': r['val_mae'],
        'R2_Gap': r['r2_gap'],
        'Val_Var_Ratio': r['val_var_ratio']
    } for r in all_results])
    
    # Sort by validation R²
    df_results = df_results.sort_values('Val_R2', ascending=False)
    
    print("\n📊 TOP 10 CONFIGURATIONS:")
    print("-"*80)
    print(df_results.head(10).to_string(index=False))
    
    print("\n\n⭐ BEST CONFIGURATION:")
    print("-"*80)
    print(f"Preprocessing:    {best_config['prep_name']}")
    print(f"Hyperparameters:  {best_config['hyper_name']}")
    print(f"Features:         {best_config['n_features']}")
    print(f"\n📊 Metrics:")
    print(f"  Train R²:       {best_config['train_r2']:.6f}")
    print(f"  Val R²:         {best_config['val_r2']:.6f}")
    print(f"  Val RMSE:       {best_config['val_rmse']:.4f}")
    print(f"  Val MAE:        {best_config['val_mae']:.4f}")
    print(f"  R² Gap:         {best_config['r2_gap']:.6f}")
    print(f"  Val Var Ratio:  {best_config['val_var_ratio']:.4f}")
    
    # Analysis by preprocessing
    print("\n\n📊 AVERAGE PERFORMANCE BY PREPROCESSING:")
    print("-"*80)
    prep_summary = df_results.groupby('Preprocessing')[['Val_R2', 'Val_RMSE', 'R2_Gap']].mean()
    print(prep_summary.to_string())
    
    # Analysis by hyperparameters
    print("\n\n📊 AVERAGE PERFORMANCE BY HYPERPARAMETERS:")
    print("-"*80)
    hyper_summary = df_results.groupby('Hyperparameters')[['Val_R2', 'Val_RMSE', 'R2_Gap']].mean()
    print(hyper_summary.to_string())
    
    return df_results


def save_results(all_results, best_config, best_model, df_results):
    """Save all results and best model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results DataFrame
    df_results.to_csv(f'grid_search_results_{timestamp}.csv', index=False)
    print(f"\n✓ Results saved to 'grid_search_results_{timestamp}.csv'")
    
    # Save best model
    best_model.save_model(f'best_model_{timestamp}.json')
    print(f"✓ Best model saved to 'best_model_{timestamp}.json'")
    
    # Save detailed results
    with open(f'grid_search_details_{timestamp}.json', 'w') as f:
        # Remove model objects for JSON serialization
        results_for_json = []
        for r in all_results:
            r_copy = r.copy()
            r_copy.pop('model', None)
            results_for_json.append(r_copy)
        json.dump({
            'config': CONFIG,
            'all_results': results_for_json,
            'best_config': {k: v for k, v in best_config.items() if k != 'model'}
        }, f, indent=2, default=str)
    print(f"✓ Detailed results saved to 'grid_search_details_{timestamp}.json'")


# ============================================================================
# RUN THE SEARCH
# ============================================================================

if __name__ == "__main__":
    # Run grid search
    all_results, best_config, best_model = run_grid_search()
    
    # Print results
    df_results = print_results_summary(all_results, best_config)
    
    # Save everything
    save_results(all_results, best_config, best_model, df_results)
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE!")
    print("="*80)
    print(f"\n💡 Best Validation R²: {best_config['val_r2']:.6f}")
    print(f"   Configuration: {best_config['prep_name']} + {best_config['hyper_name']}")
    print("\n" + "="*80)