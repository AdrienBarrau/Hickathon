import pandas as pd
import numpy as np

def detect_tricks(X_train, X_test, verbose=True):
    report = []

    for col in X_train.columns:
        train_col = X_train[col]
        test_col = X_test[col] if col in X_test.columns else pd.Series([np.nan]*len(X_test))

        # 1. Missing values
        train_missing = train_col.isna().mean()
        test_missing = test_col.isna().mean()
        if verbose:
            print(f"Column: {col}")
            print(f"  Train missing: {train_missing:.2%}, Test missing: {test_missing:.2%}")
        if test_missing > 0.8 and train_missing < 0.5:
            report.append((col, 'Mostly missing in test', train_missing, test_missing))

        # 2. Constant columns
        if train_col.nunique() == 1 and test_col.nunique() > 1:
            report.append((col, 'Constant in train but variable in test', train_col.nunique(), test_col.nunique()))
        if test_col.nunique() == 1 and train_col.nunique() > 1:
            report.append((col, 'Constant in test but variable in train', train_col.nunique(), test_col.nunique()))

        # 3. Distribution mismatch for numeric
        if pd.api.types.is_numeric_dtype(train_col):
            train_mean = train_col.mean()
            test_mean = test_col.mean()
            if not np.isnan(test_mean):
                diff = abs(train_mean - test_mean)
                if diff > 3 * train_col.std():  # heuristic
                    report.append((col, 'Mean shift between train and test', train_mean, test_mean))

        # 4. Categorical high cardinality mismatch
        if pd.api.types.is_object_dtype(train_col) or pd.api.types.is_categorical_dtype(train_col):
            train_unique = train_col.nunique()
            test_unique = test_col.nunique()
            unique_diff = abs(train_unique - test_unique)
            if unique_diff / max(train_unique, 1) > 0.5:  # >50% difference
                report.append((col, 'High cardinality difference', train_unique, test_unique))

        # 5. Low overlap in categorical values
        if pd.api.types.is_object_dtype(train_col) or pd.api.types.is_categorical_dtype(train_col):
            train_values = set(train_col.dropna().unique())
            test_values = set(test_col.dropna().unique())
            if len(train_values & test_values) / max(len(train_values),1) < 0.5:
                report.append((col, 'Low overlap of categories', len(train_values), len(test_values)))

    # Overall report
    report_df = pd.DataFrame(report, columns=['Column', 'Issue', 'Train Metric', 'Test Metric'])
    return report_df

if __name__ == "__main__":
    # Example usage
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')

    tricks_report = detect_tricks(X_train, X_test)
    print("\n--- Potential Tricks Report ---")
    print(tricks_report)
    tricks_report.to_csv("tricks_report.csv", index=False)
