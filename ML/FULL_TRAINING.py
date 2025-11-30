import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FULL DATASET TRAINING - FINAL MODEL")
print("="*80)

# ============================================================================
# HYPERPARAMETERS - MODIFY THESE
# ============================================================================

HYPERPARAMETERS = {
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
    'n_jobs': -1
}

PREPROCESSING = {
    'variance_threshold': 0.01,
    'missing_threshold': 0.8,
    'numeric_imputer': 'median',
    'categorical_imputer': 'most_frequent',
    'top_n_features': 200
}

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")
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

print(f"Final shapes - X_train: {X_train_full.shape}, y_train: {y_train_full.shape}")

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================
print("\n[2/4] Preprocessing...")

# Remove empty columns
empty_cols_test = X_test.columns[X_test.isnull().all()].tolist()
X_train_full = X_train_full.drop(columns=empty_cols_test, errors='ignore')
X_test = X_test.drop(columns=empty_cols_test, errors='ignore')

# Identify column types
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_full.select_dtypes(include=['object']).columns.tolist()

# Remove high missing columns
missing_pct = X_train_full.isnull().sum() / len(X_train_full)
cols_to_drop = missing_pct[missing_pct > PREPROCESSING['missing_threshold']].index.tolist()
X_train_full = X_train_full.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

numeric_cols = [col for col in numeric_cols if col in X_train_full.columns]
categorical_cols = [col for col in categorical_cols if col in X_train_full.columns]

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Encode categorical
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Impute
numeric_imputer = SimpleImputer(strategy=PREPROCESSING['numeric_imputer'])
X_train_full[numeric_cols] = numeric_imputer.fit_transform(X_train_full[numeric_cols])
X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy=PREPROCESSING['categorical_imputer'])
    X_train_full[categorical_cols] = cat_imputer.fit_transform(X_train_full[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
else:
    cat_imputer = None

# Variance filtering
selector = VarianceThreshold(threshold=PREPROCESSING['variance_threshold'])
X_train_selected = selector.fit_transform(X_train_full)
X_test_selected = selector.transform(X_test)
selected_features = X_train_full.columns[selector.get_support()].tolist()
X_train_full = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)

print(f"Features after preprocessing: {len(selected_features)}")

# ============================================================================
# STEP 3: FEATURE SELECTION
# ============================================================================
print("\n[3/4] Selecting important features...")
quick_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
quick_model.fit(X_train_full, y_train_full)

feature_importance = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': quick_model.feature_importances_
}).sort_values('importance', ascending=False)

top_n_features = min(PREPROCESSING['top_n_features'], len(X_train_full.columns))
top_features = feature_importance.head(top_n_features)['feature'].tolist()

X_train_full = X_train_full[top_features]
X_test = X_test[top_features]

print(f"Using top {len(top_features)} features")
print(f"\nTop 10 features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# STEP 4: TRAIN FINAL MODEL ON FULL DATASET
# ============================================================================
print("\n[4/4] Training final model on FULL dataset...")
print("="*80)
print("TRAINING WITH THESE HYPERPARAMETERS:")
for key, value in HYPERPARAMETERS.items():
    print(f"  {key}: {value}")
print("="*80)

final_model = xgb.XGBRegressor(**HYPERPARAMETERS)
final_model.fit(X_train_full, y_train_full, verbose=False)

print(f"\n✓ Model trained on {len(X_train_full)} samples")
print(f"✓ Using {len(top_features)} features")

# ============================================================================
# SAVE EVERYTHING
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL AND PREPROCESSING OBJECTS")
print("="*80)

# Save model
final_model.save_model('final_model_full_data.json')
print("✓ Model saved as 'final_model_full_data.json'")

with open('final_model_full_data.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("✓ Model saved as 'final_model_full_data.pkl'")

# Save all preprocessing objects and metadata
model_package = {
    'model': final_model,
    'hyperparameters': HYPERPARAMETERS,
    'preprocessing_config': PREPROCESSING,
    'features': top_features,
    'n_features': len(top_features),
    'n_samples': len(X_train_full),
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'numeric_imputer': numeric_imputer,
    'cat_imputer': cat_imputer,
    'variance_selector': selector,
    'label_encoders': label_encoders,
    'empty_cols_test': empty_cols_test,
    'cols_to_drop': cols_to_drop,
    'feature_importance': feature_importance.to_dict()
}

with open('final_model_package.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print("✓ Complete model package saved as 'final_model_package.pkl'")

print("\n📦 SAVED FILES:")
print("  1. final_model_full_data.json     (XGBoost model only)")
print("  2. final_model_full_data.pkl      (XGBoost model only)")
print("  3. final_model_package.pkl        (Complete package with preprocessing)")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\n✓ Model trained on 100% of training data ({len(X_train_full)} samples)")
print(f"✓ Ready for test set predictions")

print("\n💡 TO MAKE PREDICTIONS:")
print("  import pickle")
print("  ")
print("  # Load complete package")
print("  with open('final_model_package.pkl', 'rb') as f:")
print("      package = pickle.load(f)")
print("  ")
print("  model = package['model']")
print("  features = package['features']")
print("  ")
print("  # Make predictions (X_test must be preprocessed the same way)")
print("  predictions = model.predict(X_test[features])")

print("\n" + "="*80)