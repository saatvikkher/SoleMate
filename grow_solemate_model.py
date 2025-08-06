import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import os
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import glob

METRICS_DIR = "results/"
OUTPUT_MODEL_NAME = "solemate_model"

def load_similarity_metrics(metrics_dir):
    """
    Load and combine all *TRAIN* CSV files in the metrics directory.
    Label: mated (TRUE=1, FALSE=0)
    Features: specific similarity metrics selected for training.
    """
    print(f"Loading all *TRAIN* files from {metrics_dir}...")
    csv_files = glob.glob(os.path.join(metrics_dir, '*TRAIN*.csv'))
    print(f"Found {len(csv_files)} TRAIN files.")
    print(csv_files)
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    if not dfs:
        raise ValueError("No TRAIN files loaded!")
    df = pd.concat(dfs, ignore_index=True)

    # Convert 'mated' to binary label
    df['label'] = df['mated'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 1: 1, 0: 0})
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Define specific features to use for training
    feature_columns = [
        'q_points_count', 'k_points_count', 'mean', 'std', '0.1', '0.25', '0.5', '0.75', '0.9',
        'centroid_distance_n_clusters_20', 'cluster_proportion_n_clusters_20', 
        'iterations_k_n_clusters_20', 'wcv_ratio_n_clusters_20',
        'centroid_distance_n_clusters_100', 'cluster_proportion_n_clusters_100', 
        'iterations_k_n_clusters_100', 'wcv_ratio_n_clusters_100',
        'q_pct_threshold_1', 'k_pct_threshold_1', 'q_pct_threshold_2', 'k_pct_threshold_2',
        'q_pct_threshold_3', 'k_pct_threshold_3', 'q_pct_threshold_5', 'k_pct_threshold_5',
        'q_pct_threshold_10', 'k_pct_threshold_10', 'peak_value', 'MSE', 'SSIM', 'NCC',
        'PSR', 'jaccard_index_0', 'jaccard_index_-1', 'jaccard_index_-2'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Remove missing columns from feature list
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    print(f"Selected {len(feature_columns)} feature columns.")

    X = df[feature_columns].values
    y = df['label'].values

    # Remove rows with NaN
    if np.isnan(X).any():
        print("Warning: Found NaN values in features. Removing rows with NaN values.")
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Feature names: {feature_columns}")

    # Data quality
    print("\nData Quality Analysis:")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Feature statistics:")
    for i, col in enumerate(feature_columns):
        print(f"  {col}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}, min={X[:, i].min():.4f}, max={X[:, i].max():.4f}")
    feature_df = pd.DataFrame(X, columns=feature_columns)
    print(f"\nFeature correlation matrix:")
    print(feature_df.corr())

    return X, y, feature_columns

def define_hyperparameter_grid():
    param_grid = {
        'n_estimators': [500, 1000, 2000, 5000],
        'max_depth': [10, 30, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    return param_grid

def grid_search_with_cv(X, y, param_grid, cv_folds=5, random_state=42):
    print(f"Starting grid search with {cv_folds}-fold cross-validation...")
    print(f"Total parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results = []
    for i, params in tqdm(enumerate(param_combinations)):
        param_dict = dict(zip(param_names, params))
        model_random_state = random_state + i
        rf = RandomForestClassifier(
            **param_dict,
            random_state=model_random_state,
            n_jobs=-1
        )
        cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        result = {
            'params': param_dict,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score,
            'cv_scores': cv_scores.tolist()
        }
        results.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Progress: {i+1}/{len(param_combinations)} combinations tested")
            print(f"Current best: {max([r['mean_cv_score'] for r in results]):.4f}")
    return results

def find_best_parameters(results):
    best_result = max(results, key=lambda x: x['mean_cv_score'])
    print("\n" + "="*50)
    print("BEST PARAMETERS FOUND:")
    print("="*50)
    for param, value in best_result['params'].items():
        print(f"{param}: {value}")
    print(f"Mean CV Score: {best_result['mean_cv_score']:.4f} (+/- {best_result['std_cv_score']*2:.4f})")
    print("="*50)
    return best_result

def train_final_model(X, y, best_params, feature_names, random_state=42):
    print("\nTraining final model with best parameters...")
    final_model = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=10
    )
    final_model.fit(X, y)
    y_pred = final_model.predict(X)
    train_accuracy = accuracy_score(y, y_pred)
    print(f"Final model training accuracy: {train_accuracy:.4f}")
    feature_importance = final_model.feature_importances_
    print("\nFeature Importance:")
    for feature, importance in zip(feature_names, feature_importance):
        print(f"{feature}: {importance:.4f}")
    return final_model, train_accuracy

def save_results(results, output_file="random_forest_results_solemate.csv"):
    print(f"\nSaving results to {output_file}...")
    rows = []
    for result in results:
        row = {
            'n_estimators': result['params']['n_estimators'],
            'max_depth': result['params']['max_depth'],
            'min_samples_split': result['params']['min_samples_split'],
            'min_samples_leaf': result['params']['min_samples_leaf'],
            'mean_cv_score': result['mean_cv_score'],
            'std_cv_score': result['std_cv_score']
        }
        rows.append(row)
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values('mean_cv_score', ascending=False)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return results_df

def main():
    print("="*60)
    print("RANDOM FOREST CLASSIFIER FOR SOLEMATE SIMILARITY METRICS")
    print("="*60)
    random_state = 42
    np.random.seed(random_state)
    metrics_dir = METRICS_DIR
    try:
        X, y, feature_names = load_similarity_metrics(metrics_dir)
        param_grid = define_hyperparameter_grid()
        results = grid_search_with_cv(X, y, param_grid, cv_folds=5, random_state=random_state)
        best_result = find_best_parameters(results)
        final_model, train_accuracy = train_final_model(X, y, best_result['params'], feature_names, random_state)
        results_df = save_results(results)
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Best CV score: {best_result['mean_cv_score']:.4f}")
        print(f"Final training accuracy: {train_accuracy:.4f}")
        print(f"Total parameter combinations tested: {len(results)}")
        print("="*60)
        import joblib
        model_file = OUTPUT_MODEL_NAME + ".joblib"
        joblib.dump(final_model, model_file)
        print(f"Final model saved to {model_file}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()