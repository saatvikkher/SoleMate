from sole import Sole
from solepair import SolePair
from solepaircompare import SolePairCompare
import pandas as pd
import pickle
import zipfile
import time
import argparse

class SolemateParams:
    def __init__(self, params={}):
        # DEFAULT PARAMS FOR SOLEMATE
        if not params:
            self.downsample_rates = [0.05]
            self.cluster_n1 = 20
            self.cluster_n2 = 100
            self.po_1 = 1
            self.po_2 = 2
            self.po_3 = 3
            self.po_4 = 5
            self.po_5 = 10
        else:
            self.downsample_rates = params["downsample_rates"]
            self.cluster_n1 = params["cluster_n1"]
            self.cluster_n2 = params["cluster_n2"]
            self.po_1 = params["propn_overlap_1"]
            self.po_2 = params["propn_overlap_2"]
            self.po_3 = params["propn_overlap_3"]
            self.po_4 = params["propn_overlap_4"]
            self.po_5 = params["propn_overlap_5"]


class Benchmark:
    def __init__(self, border_width: int, params: SolemateParams):
        self.BORDER_WIDTH = border_width
        self.params = params
    
    # run one trial
    def trial(self, q_fp: str, k_fp: str, trial_id: str, is_mated: bool):
        # Q = Sole(q_fp[::-1].replace('_depporc', '', 1)[::-1], border_width=self.BORDER_WIDTH)
        # K = Sole(k_fp[::-1].replace('_depporc', '', 1)[::-1], border_width=self.BORDER_WIDTH)
        def add_cropped_to_tiff(path):
            if path.endswith('.tiff'):
                return path[:-5] + '_cropped.tiff'
            return path

        Q = Sole(add_cropped_to_tiff(q_fp), border_width=self.BORDER_WIDTH)
        K = Sole(add_cropped_to_tiff(k_fp), border_width=self.BORDER_WIDTH)

        pair = SolePair(Q, K, mated=True)
        print(f"Q shape: {Q.coords.shape[0]}")

        sc = SolePairCompare(pair, 
                        icp_downsample_rates=self.params.downsample_rates,
                        shift_up=True,
                        shift_down=True,
                        shift_left=True,
                        shift_right=True,
                        two_way=True) # icp is called here
        
        dist_metrics = sc.min_dist()
        all_cluster_metrics = sc.cluster_metrics(n_clusters=self.params.cluster_n1)
        all_cluster_metrics.update(sc.cluster_metrics(n_clusters=self.params.cluster_n2))
        phase_correlation_metrics = sc.pc_metrics()
        jaccard_index = sc.jaccard_index()

        q1 = sc.propn_overlap(threshold=self.params.po_1)
        k1 = sc.propn_overlap(threshold=self.params.po_1, Q_as_base=False)
        q2 = sc.propn_overlap(threshold=self.params.po_2)
        k2 = sc.propn_overlap(threshold=self.params.po_2, Q_as_base=False)
        q3 = sc.propn_overlap(threshold=self.params.po_3)
        k3 = sc.propn_overlap(threshold=self.params.po_3, Q_as_base=False)
        q4 = sc.propn_overlap(threshold=self.params.po_4)
        k4 = sc.propn_overlap(threshold=self.params.po_4, Q_as_base=False)
        q5 = sc.propn_overlap(threshold=self.params.po_5)
        k5 = sc.propn_overlap(threshold=self.params.po_5, Q_as_base=False)

        row = {
            'trial_id': trial_id,
            'q_points_count': Q.coords.shape[0],
            'k_points_count': K.coords.shape[0],
            'mated': is_mated,
            f'q_pct_threshold_{self.params.po_1}': q1,
            f'k_pct_threshold_{self.params.po_1}': k1,
            f'q_pct_threshold_{self.params.po_2}': q2,
            f'k_pct_threshold_{self.params.po_2}': k2,
            f'q_pct_threshold_{self.params.po_3}': q3,
            f'k_pct_threshold_{self.params.po_3}': k3,
            f'q_pct_threshold_{self.params.po_4}': q4,
            f'k_pct_threshold_{self.params.po_4}': k4,
            f'q_pct_threshold_{self.params.po_5}': q5,
            f'k_pct_threshold_{self.params.po_5}': k5
        }
        row.update(dist_metrics)
        row.update(all_cluster_metrics)
        row.update(phase_correlation_metrics)
        row.update(jaccard_index)
        row = pd.DataFrame(row, index=[0])
        return row
    
    def run(self, csv_fp: str, data_fp: str, output_fp: str, num_trials: int, shuffle_files: bool, save_time_per_pair: bool = False) -> str:
        trial_files = pd.read_csv(csv_fp, header=0)
        if shuffle_files:
            trial_files = trial_files.sample(frac=1).reset_index(drop=True)
        if num_trials > len(trial_files):
            print(f"WARN: tried {num_trials} trials, only {len(trial_files)} available")
            num_trials = len(trial_files)
        output_path = f"{output_fp}"
        timing_path = f"{output_fp}_timing.csv"
        for trial in range(num_trials):
            trial_row = trial_files.iloc[trial]
            q_fp = trial_row["q"]
            k_fp = trial_row["k"]
            is_mated = trial_row["mated"]
            trial_id = f"{trial}_{q_fp}-{k_fp}"

            start_time = time.time()
            row_df = self.trial(f"{data_fp}{q_fp}", f"{data_fp}{k_fp}", trial_id, is_mated)
            elapsed = time.time() - start_time

            # Write to CSV after each trial
            if trial == 0:
                row_df.to_csv(output_path, mode='w', header=True, index=False)
                if save_time_per_pair:
                    print('saving time')
                    pd.DataFrame({"trial_id": [trial_id], "time_seconds": [elapsed]}).to_csv(timing_path, mode='w', header=True, index=False)
            else:
                row_df.to_csv(output_path, mode='a', header=False, index=False)
                if save_time_per_pair:
                    pd.DataFrame({"trial_id": [trial_id], "time_seconds": [elapsed]}).to_csv(timing_path, mode='a', header=False, index=False)
        return "SUCCESS"

def parse_args():
    parser = argparse.ArgumentParser(description="Process trial configuration.")
    default_csv = 'scripts/pair_info/BASELINE_TEST_KM.csv'
    default_data_fp = '../../download/longitudinal_cropped/'
    default_output_fp = 'solemate-metrics-output'
    default_num_trials = 10
    default_border_width = 1

    parser.add_argument(
        "--border_width",
        type=int,
        default=default_border_width,
        required=False,
        help="Border width of input images"
    )
    parser.add_argument(
        "--csv_fp",
        type=str,
        default=default_csv,
        required=False,
        help="Path to the CSV file with file info"
    )
    parser.add_argument(
        "--data_fp",
        type=str,
        default=default_data_fp,
        required=False,
        help="Path to the directory containing image files"
    )
    parser.add_argument(
        "--output_fp",
        type=str,
        default=default_output_fp,
        required=False,
        help="Name of output CSV"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=default_num_trials,
        required=False,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--shuffle_files",
        action="store_true",
        help="Whether to shuffle the input files"
    )
    parser.add_argument(
        "--save_time_per_pair",
        action="store_true",
        help="If set, saves the time taken for each pair to <output_fp>_timing.csv"
    )

    return parser.parse_args()

args = parse_args()
print("Configuration:")
print(f"  border_width  = {args.border_width}")
print(f"  csv_fp        = {args.csv_fp}")
print(f"  data_fp       = {args.data_fp}")
print(f"  output_fp     = {args.output_fp}")
print(f"  num_trials    = {args.num_trials}")
print(f"  shuffle_files = {args.shuffle_files}")

params = SolemateParams() # default params (as specified in paper)
bench = Benchmark(args.border_width, params)

bench.run(
    args.csv_fp,
    args.data_fp,
    args.output_fp,
    args.num_trials,
    args.shuffle_files,
    args.save_time_per_pair
)