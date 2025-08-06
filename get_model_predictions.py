import pandas as pd
import pickle
import zipfile
import argparse
import os
import joblib

def load_model(pkl_name="solemate_model.joblib"):
    # Extract and load the trained model

    with open(pkl_name, 'rb') as f:
        model = joblib.load(f)
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Run Solemate model on precomputed similarity metrics"
    )
    parser.add_argument(
        "--metrics_fp",
        type=str,
        required=True,
        help="Path to the CSV file with similarity metrics"
    )
    parser.add_argument(
        "--output_fp",
        type=str,
        required=True,
        help="Name (without extension) for the output CSV"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary classification"
    )
    args = parser.parse_args()

    print("Loading model...")
    model = load_model()
    # feature_names = model.feature_names_in_
    feature_names = [
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

    print("Reading metrics...")
    df = pd.read_csv(args.metrics_fp)

    # Verify required features
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in metrics file: {missing}")

    # Select and order features
    X = df[feature_names].round(2)

    print("Predicting probabilities...")
    probs = model.predict_proba(X)[:, 1]
    df['_Pred'] = probs
    # df['_Real'] = df['mated'].apply(lambda x: 1 if x in [True, 1, 'True', 'true'] else 0)

    # Compute accuracy: first half true, second half false
    # n = len(df)
    # true_labels = [1 if i < n // 2 else 0 for i in range(n)]
    # pred_labels = (df['_Pred'] > 0.5).astype(int)
    # accuracy = (pred_labels == true_labels).mean()
    # print(f"Accuracy: {accuracy:.4f} for {n} trials")

    # Save results
    if 'trial_id' not in df.columns:
        df.insert(0, 'trial_id', df.index.astype(str))
    cols = ['trial_id'] + list(feature_names) + ['_Pred']
    df.to_csv(f"{args.output_fp}.csv", columns=cols, index=False)
    print(f"Results saved to {args.output_fp}.csv")


if __name__ == '__main__':
    main()