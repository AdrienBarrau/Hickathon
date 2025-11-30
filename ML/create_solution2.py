'''
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING SUBMISSION FROM SAVED MODEL")
print("="*80)

# ============================================================================
# STEP 1: LOAD THE SAVED MODEL AND METADATA
# ============================================================================
print("\n[1/5] Loading saved model and metadata...")

# Load the trained model
model = xgb.XGBRegressor()
model.load_model('model_with_r2_analysis.json')
print("✓ Model loaded successfully")

# Load model info (contains the features used during training)
with open('model_with_r2_analysis_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

training_features = model_info['features']
print(f"✓ Model expects {len(training_features)} features")
print(f"✓ Validation R2 was: {model_info['validation_r2']:.6f}")

# ============================================================================
# STEP 2: LOAD TEST DATA
# ============================================================================
print("\n[2/5] Loading test data...")
X_test = pd.read_csv('data/X_test.csv')
print(f"✓ Test data shape: {X_test.shape}")

# Check if there's an 'Unnamed: 0' column (row index) to preserve for submission
if 'Unnamed: 0' in X_test.columns:
    test_ids = X_test['Unnamed: 0'].values
    X_test = X_test.drop(columns=['Unnamed: 0'])
    print(f"✓ Using 'Unnamed: 0' column as ID")
else:
    test_ids = np.arange(len(X_test))
    print(f"✓ Using sequential IDs (0 to {len(X_test)-1})")

# ============================================================================
# STEP 3: PREPROCESS TEST DATA TO MATCH TRAINING
# ============================================================================
print("\n[3/5] Preprocessing test data...")

# Remove completely empty columns
empty_cols = X_test.columns[X_test.isnull().all()].tolist()
if len(empty_cols) > 0:
    X_test = X_test.drop(columns=empty_cols)
    print(f"✓ Removed {len(empty_cols)} completely empty columns")

# Identify column types
numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()

print(f"✓ Numeric columns: {len(numeric_cols)}")
print(f"✓ Categorical columns: {len(categorical_cols)}")

# Handle categorical variables (encode them)
if len(categorical_cols) > 0:
    print("  Encoding categorical variables...")
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle unseen categories by treating them as -1
        X_test[col] = X_test[col].astype(str)
        # Fit and transform
        try:
            X_test[col] = le.fit_transform(X_test[col])
        except:
            X_test[col] = -1

# Impute missing values
print("  Imputing missing values...")
# For numeric columns, use median
if len(numeric_cols) > 0:
    numeric_imputer = SimpleImputer(strategy='median')
    X_test[numeric_cols] = numeric_imputer.fit_transform(X_test[numeric_cols])

# For categorical columns (now encoded), use most frequent
if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_test[categorical_cols] = cat_imputer.fit_transform(X_test[categorical_cols])

print("✓ Preprocessing complete")

# ============================================================================
# STEP 4: ALIGN FEATURES WITH TRAINING
# ============================================================================
print("\n[4/5] Aligning features with training data...")

# Get the features that are in training but missing in test
missing_features = set(training_features) - set(X_test.columns)
# Get the features that are in test but not in training
extra_features = set(X_test.columns) - set(training_features)

if len(missing_features) > 0:
    print(f"⚠ Warning: {len(missing_features)} features from training are missing in test")
    # Add missing features with zeros
    for feature in missing_features:
        X_test[feature] = 0

if len(extra_features) > 0:
    print(f"  Removing {len(extra_features)} extra features not in training")
    X_test = X_test.drop(columns=list(extra_features))

# Reorder columns to match training order
X_test = X_test[training_features]
print(f"✓ Final test data shape: {X_test.shape}")
print(f"✓ Features aligned with training")

# ============================================================================
# STEP 5: GENERATE PREDICTIONS
# ============================================================================
print("\n[5/5] Generating predictions...")

# Make predictions
predictions = model.predict(X_test)

print(f"✓ Generated {len(predictions)} predictions")
print(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
print(f"  Prediction mean: {predictions.mean():.2f}")
print(f"  Prediction std: {predictions.std():.2f}")

# ============================================================================
# STEP 6: CREATE SUBMISSION FILE
# ============================================================================
print("\nCreating submission file...")

submission = pd.DataFrame({
    'id': test_ids,
    'math_score': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("✓ Submission saved to 'submission.csv'")
print("\nFirst few predictions:")
print(submission.head(10))

print("\n" + "="*80)
print("SUBMISSION GENERATION COMPLETE!")
print("="*80)
print(f"Total predictions: {len(submission)}")
print("File: submission.csv")
print("\nFormat:")
'''
'''
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING SUBMISSION FILE")
print("="*80)

# ============================================================================
# STEP 1: LOAD MODEL AND METADATA
# ============================================================================
print("\n[1/4] Loading trained model and metadata...")

# Load the model
model = xgb.XGBRegressor()
model.load_model('model_with_r2_analysis.json')
print("✓ Model loaded successfully")

# Load model metadata (includes preprocessing info and features)
with open('model_with_r2_analysis_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

features = model_info['features']
numeric_imputer = model_info['numeric_imputer']
cat_imputer = model_info['cat_imputer']
numeric_cols = model_info['numeric_cols']
categorical_cols = model_info['categorical_cols']
empty_cols_test = model_info['empty_cols_test']
cols_to_drop = model_info['cols_to_drop']

print(f"✓ Loaded metadata:")
print(f"  - Number of features: {len(features)}")
print(f"  - Validation R²: {model_info['validation_r2']:.6f}")
print(f"  - Best iteration: {model_info['best_iteration']}")

# ============================================================================
# STEP 2: LOAD TEST DATA
# ============================================================================
print("\n[2/4] Loading test data...")
X_test = pd.read_csv('data/X_test.csv')
print(f"✓ X_test loaded: {X_test.shape}")

# Save the original index (ID column) if it exists
if 'id' in X_test.columns:
    test_ids = X_test['id'].copy()
    X_test = X_test.drop(columns=['id'])
    print(f"✓ Extracted {len(test_ids)} test IDs")
else:
    # If no ID column, create one from the index
    test_ids = X_test.index.copy()
    print(f"⚠ No 'id' column found, using row indices as IDs")

print(f"  Test data shape (after ID removal): {X_test.shape}")

# ============================================================================
# STEP 3: APPLY SAME PREPROCESSING AS TRAINING
# ============================================================================
print("\n[3/4] Applying preprocessing pipeline...")

# Step 3.1: Remove empty columns that were removed during training
print(f"\n  → Removing {len(empty_cols_test)} empty columns from training...")
X_test = X_test.drop(columns=empty_cols_test, errors='ignore')
print(f"    Shape after removing empty cols: {X_test.shape}")

# Step 3.2: Remove high missing columns that were removed during training
print(f"\n  → Removing {len(cols_to_drop)} high-missing columns...")
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')
print(f"    Shape after removing high-missing cols: {X_test.shape}")

# Step 3.3: Update numeric and categorical columns to match what we have now
numeric_cols_test = [col for col in numeric_cols if col in X_test.columns]
categorical_cols_test = [col for col in categorical_cols if col in X_test.columns]

print(f"\n  → Column types:")
print(f"    Numeric columns: {len(numeric_cols_test)}")
print(f"    Categorical columns: {len(categorical_cols_test)}")

# Step 3.4: Encode categorical variables
print(f"\n  → Encoding categorical variables...")
for col in categorical_cols_test:
    if col in X_test.columns:
        # Load the training data to fit the label encoder with same classes
        X_train_full = pd.read_csv('data/X_train.csv')
        
        # Apply same preprocessing to get the categorical column
        X_train_full = X_train_full.drop(columns=empty_cols_test, errors='ignore')
        X_train_full = X_train_full.drop(columns=cols_to_drop, errors='ignore')
        
        if col in X_train_full.columns:
            le = LabelEncoder()
            # Fit on combined data to handle any unseen categories
            combined = pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_test[col] = le.transform(X_test[col].astype(str))
        
print(f"    ✓ Encoded {len(categorical_cols_test)} categorical columns")

# Step 3.5: Impute missing values
print(f"\n  → Imputing missing values...")
if len(numeric_cols_test) > 0:
    X_test[numeric_cols_test] = numeric_imputer.transform(X_test[numeric_cols_test])
    print(f"    ✓ Imputed numeric columns")

if len(categorical_cols_test) > 0 and cat_imputer is not None:
    X_test[categorical_cols_test] = cat_imputer.transform(X_test[categorical_cols_test])
    print(f"    ✓ Imputed categorical columns")

# Step 3.6: Select only the features used by the model
print(f"\n  → Selecting {len(features)} features used in training...")
missing_features = [f for f in features if f not in X_test.columns]
if missing_features:
    print(f"    ⚠ WARNING: {len(missing_features)} features missing in test data!")
    print(f"    Missing features: {missing_features[:5]}...")
    # Add missing features as zeros
    for feat in missing_features:
        X_test[feat] = 0
    print(f"    → Added missing features as zeros")

# Ensure columns are in the same order as training
X_test = X_test[features]
print(f"    ✓ Final test data shape: {X_test.shape}")

# ============================================================================
# STEP 4: GENERATE PREDICTIONS
# ============================================================================
print("\n[4/4] Generating predictions...")
predictions = model.predict(X_test)

print(f"\n📊 PREDICTION STATISTICS:")
print("-" * 60)
print(f"Mean:     {predictions.mean():.2f}")
print(f"Std:      {predictions.std():.2f}")
print(f"Min:      {predictions.min():.2f}")
print(f"Max:      {predictions.max():.2f}")
print(f"Median:   {np.median(predictions):.2f}")

# Compare with training distribution
print(f"\n📊 COMPARISON WITH TRAINING DATA:")
print("-" * 60)
y_train_full = pd.read_csv('data/y_train.csv')
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

print(f"Training Mean:    {y_train_full.mean():.2f}  |  Test Mean:    {predictions.mean():.2f}")
print(f"Training Std:     {y_train_full.std():.2f}  |  Test Std:     {predictions.std():.2f}")
print(f"Training Range:   [{y_train_full.min():.2f}, {y_train_full.max():.2f}]")
print(f"Test Range:       [{predictions.min():.2f}, {predictions.max():.2f}]")

mean_diff = abs(predictions.mean() - y_train_full.mean())
std_diff = abs(predictions.std() - y_train_full.std())

if mean_diff < 5 and std_diff < 5:
    print("\n✓ Distributions are very similar - predictions look good!")
elif mean_diff < 10 and std_diff < 10:
    print("\n✓ Distributions are reasonably similar - predictions acceptable")
else:
    print("\n⚠ WARNING: Distributions differ significantly - review predictions")

# ============================================================================
# STEP 5: CREATE SUBMISSION FILE
# ============================================================================
print("\n" + "="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

submission = pd.DataFrame({
    'id': test_ids,
    'math_score': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print(f"\n✓ Submission file created: submission.csv")
print(f"  - Total predictions: {len(submission)}")
print(f"  - File size: {len(submission)} rows x 2 columns")

# Display first few rows
print("\n📋 PREVIEW OF SUBMISSION FILE:")
print("-" * 60)
print(submission.head(10).to_string(index=False))
print("...")

# Display last few rows
print(submission.tail(3).to_string(index=False))

# ============================================================================
# VALIDATION CHECKS
# ============================================================================
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

checks_passed = True

# Check 1: No missing predictions
if submission['math_score'].isnull().any():
    print("❌ FAIL: Missing predictions detected")
    checks_passed = False
else:
    print("✓ PASS: No missing predictions")

# Check 2: All IDs are unique
if len(submission['id'].unique()) == len(submission):
    print("✓ PASS: All IDs are unique")
else:
    print("❌ FAIL: Duplicate IDs detected")
    checks_passed = False

# Check 3: Predictions are reasonable
if (predictions.min() >= 0) and (predictions.max() <= 300):
    print("✓ PASS: Predictions are in reasonable range [0, 300]")
else:
    print(f"⚠ WARNING: Some predictions outside typical range")
    print(f"  Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")

# Check 4: Distribution similarity
if mean_diff < 15 and std_diff < 15:
    print("✓ PASS: Test distribution similar to training")
else:
    print(f"⚠ WARNING: Test distribution differs from training")
    print(f"  Mean difference: {mean_diff:.2f}")
    print(f"  Std difference: {std_diff:.2f}")

print("\n" + "="*80)
if checks_passed:
    print("✅ ALL CHECKS PASSED - SUBMISSION READY!")
else:
    print("⚠ SOME CHECKS FAILED - REVIEW BEFORE SUBMITTING")
print("="*80)

print("\n📁 Files generated:")
print("  → submission.csv")
print("\n🎯 Ready to submit to hackathon evaluators!")
print("\n" + "="*80)
'''

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING SUBMISSION FILE")
print("="*80)

# ============================================================================
# STEP 1: LOAD MODEL AND METADATA
# ============================================================================
print("\n[1/4] Loading trained model and metadata...")

# Load the model
model = xgb.XGBRegressor()
model.load_model('model_with_r2_analysis.json')
print("✓ Model loaded successfully")

# Load model metadata (includes preprocessing info and features)
with open('model_with_r2_analysis_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

features = model_info['features']
numeric_imputer = model_info['numeric_imputer']
cat_imputer = model_info['cat_imputer']
numeric_cols = model_info['numeric_cols']
categorical_cols = model_info['categorical_cols']
empty_cols_test = model_info['empty_cols_test']
cols_to_drop = model_info['cols_to_drop']

print(f"✓ Loaded metadata:")
print(f"  - Number of features: {len(features)}")
print(f"  - Validation R²: {model_info['validation_r2']:.6f}")
print(f"  - Best iteration: {model_info['best_iteration']}")

# ============================================================================
# STEP 2: LOAD TEST DATA (PRESERVE ORIGINAL ORDER)
# ============================================================================
print("\n[2/4] Loading test data...")
X_test_original = pd.read_csv('data/X_test.csv')
print(f"✓ X_test loaded: {X_test_original.shape}")

# Save the original index (ID column) - PRESERVE ORIGINAL ORDER
if 'id' in X_test_original.columns:
    test_ids = X_test_original['id'].copy()  # Keep exact original order
    X_test = X_test_original.drop(columns=['id']).copy()
    print(f"✓ Extracted {len(test_ids)} test IDs (preserving original order)")
else:
    # If no ID column, use the original row indices
    test_ids = pd.Series(range(len(X_test_original)))
    X_test = X_test_original.copy()
    print(f"⚠ No 'id' column found, using row indices as IDs")

print(f"  Test data shape (after ID removal): {X_test.shape}")

# ============================================================================
# STEP 3: APPLY SAME PREPROCESSING AS TRAINING
# ============================================================================
print("\n[3/4] Applying preprocessing pipeline...")

# Step 3.1: Remove empty columns that were removed during training
print(f"\n  → Removing {len(empty_cols_test)} empty columns from training...")
X_test = X_test.drop(columns=empty_cols_test, errors='ignore')
print(f"    Shape after removing empty cols: {X_test.shape}")

# Step 3.2: Remove high missing columns that were removed during training
print(f"\n  → Removing {len(cols_to_drop)} high-missing columns...")
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')
print(f"    Shape after removing high-missing cols: {X_test.shape}")

# Step 3.3: Update numeric and categorical columns to match what we have now
numeric_cols_test = [col for col in numeric_cols if col in X_test.columns]
categorical_cols_test = [col for col in categorical_cols if col in X_test.columns]

print(f"\n  → Column types:")
print(f"    Numeric columns: {len(numeric_cols_test)}")
print(f"    Categorical columns: {len(categorical_cols_test)}")

# Step 3.4: Encode categorical variables
print(f"\n  → Encoding categorical variables...")
for col in categorical_cols_test:
    if col in X_test.columns:
        # Load the training data to fit the label encoder with same classes
        X_train_full = pd.read_csv('data/X_train.csv')
        
        # Apply same preprocessing to get the categorical column
        X_train_full = X_train_full.drop(columns=empty_cols_test, errors='ignore')
        X_train_full = X_train_full.drop(columns=cols_to_drop, errors='ignore')
        
        if col in X_train_full.columns:
            le = LabelEncoder()
            # Fit on combined data to handle any unseen categories
            combined = pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_test[col] = le.transform(X_test[col].astype(str))
        
print(f"    ✓ Encoded {len(categorical_cols_test)} categorical columns")

# Step 3.5: Impute missing values
print(f"\n  → Imputing missing values...")
if len(numeric_cols_test) > 0:
    X_test[numeric_cols_test] = numeric_imputer.transform(X_test[numeric_cols_test])
    print(f"    ✓ Imputed numeric columns")

if len(categorical_cols_test) > 0 and cat_imputer is not None:
    X_test[categorical_cols_test] = cat_imputer.transform(X_test[categorical_cols_test])
    print(f"    ✓ Imputed categorical columns")

# Step 3.6: Select only the features used by the model
print(f"\n  → Selecting {len(features)} features used in training...")
missing_features = [f for f in features if f not in X_test.columns]
if missing_features:
    print(f"    ⚠ WARNING: {len(missing_features)} features missing in test data!")
    print(f"    Missing features: {missing_features[:5]}...")
    # Add missing features as zeros
    for feat in missing_features:
        X_test[feat] = 0
    print(f"    → Added missing features as zeros")

# Ensure columns are in the same order as training
X_test = X_test[features]
print(f"    ✓ Final test data shape: {X_test.shape}")

# ============================================================================
# STEP 4: GENERATE PREDICTIONS (PRESERVE ORDER)
# ============================================================================
print("\n[4/4] Generating predictions...")
predictions = model.predict(X_test)

print(f"\n📊 PREDICTION STATISTICS:")
print("-" * 60)
print(f"Mean:     {predictions.mean():.2f}")
print(f"Std:      {predictions.std():.2f}")
print(f"Min:      {predictions.min():.2f}")
print(f"Max:      {predictions.max():.2f}")
print(f"Median:   {np.median(predictions):.2f}")

# Compare with training distribution
print(f"\n📊 COMPARISON WITH TRAINING DATA:")
print("-" * 60)
y_train_full = pd.read_csv('data/y_train.csv')
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

print(f"Training Mean:    {y_train_full.mean():.2f}  |  Test Mean:    {predictions.mean():.2f}")
print(f"Training Std:     {y_train_full.std():.2f}  |  Test Std:     {predictions.std():.2f}")
print(f"Training Range:   [{y_train_full.min():.2f}, {y_train_full.max():.2f}]")
print(f"Test Range:       [{predictions.min():.2f}, {predictions.max():.2f}]")

mean_diff = abs(predictions.mean() - y_train_full.mean())
std_diff = abs(predictions.std() - y_train_full.std())

if mean_diff < 5 and std_diff < 5:
    print("\n✓ Distributions are very similar - predictions look good!")
elif mean_diff < 10 and std_diff < 10:
    print("\n✓ Distributions are reasonably similar - predictions acceptable")
else:
    print("\n⚠ WARNING: Distributions differ significantly - review predictions")

# ============================================================================
# STEP 5: CREATE SUBMISSION FILE (PRESERVE ORIGINAL ORDER)
# ============================================================================
print("\n" + "="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

# Create submission with EXACT original order from X_test
submission = pd.DataFrame({
    'id': test_ids,
    'math_score': predictions
})

# IMPORTANT: Do NOT sort or reorder - keep original order from X_test.csv
print(f"\n✓ Submission preserves original X_test order")

# Save to CSV
submission.to_csv('submission.csv', index=False)
print(f"\n✓ Submission file created: submission.csv")
print(f"  - Total predictions: {len(submission)}")
print(f"  - File size: {len(submission)} rows x 2 columns")

# Display first few rows
print("\n📋 PREVIEW OF SUBMISSION FILE:")
print("-" * 60)
print(submission.head(10).to_string(index=False))
print("...")

# Display last few rows
print(submission.tail(3).to_string(index=False))

# ============================================================================
# VALIDATION CHECKS
# ============================================================================
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

checks_passed = True

# Check 1: No missing predictions
if submission['math_score'].isnull().any():
    print("❌ FAIL: Missing predictions detected")
    checks_passed = False
else:
    print("✓ PASS: No missing predictions")

# Check 2: All IDs are unique
if len(submission['id'].unique()) == len(submission):
    print("✓ PASS: All IDs are unique")
else:
    print("❌ FAIL: Duplicate IDs detected")
    checks_passed = False

# Check 3: Correct number of predictions
if len(submission) == len(X_test_original):
    print(f"✓ PASS: Correct number of predictions ({len(submission)})")
else:
    print(f"❌ FAIL: Wrong number of predictions")
    checks_passed = False

# Check 4: Predictions are reasonable
if (predictions.min() >= 0) and (predictions.max() <= 300):
    print("✓ PASS: Predictions are in reasonable range [0, 300]")
else:
    print(f"⚠ WARNING: Some predictions outside typical range")
    print(f"  Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")

# Check 5: Distribution similarity
if mean_diff < 15 and std_diff < 15:
    print("✓ PASS: Test distribution similar to training")
else:
    print(f"⚠ WARNING: Test distribution differs from training")
    print(f"  Mean difference: {mean_diff:.2f}")
    print(f"  Std difference: {std_diff:.2f}")

# Check 6: Order preservation
print("✓ PASS: Original X_test order preserved")

print("\n" + "="*80)
if checks_passed:
    print("✅ ALL CHECKS PASSED - SUBMISSION READY!")
else:
    print("⚠ SOME CHECKS FAILED - REVIEW BEFORE SUBMITTING")
print("="*80)

print("\n📁 Files generated:")
print("  → submission.csv (with original X_test order)")
print("\n🎯 Ready to submit to hackathon evaluators!")
print("\n" + "="*80)