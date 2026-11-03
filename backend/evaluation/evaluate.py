import os
import sys
import time
import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure backend directory is in path
backend_dir = os.path.abspath('backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from models.echonet_wrapper import EchoNetAnalyzer

def run_evaluation(num_samples=50):
    project_root = os.path.abspath('.')
    csv_path = os.path.join(project_root, 'dataset', 'Echocardiogram Dataset', 'EchoNet-Dynamic', 'FileList.csv')
    videos_dir = os.path.join(project_root, 'dataset', 'Echocardiogram Dataset', 'EchoNet-Dynamic', 'Videos')
    results_dir = os.path.join(project_root, 'results')
    output_csv_path = os.path.join(results_dir, 'evaluation_results.csv')

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("      ECHOSCAN AI PRO — SCIENTIFIC EVALUATION SUITE")
    print("=" * 80)
    print(f"Reading dataset manifest: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FileList.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Total rows in FileList.csv: {len(df)}")

    # Filter to TEST split
    if 'Split' in df.columns:
        test_df = df[df['Split'].str.upper() == 'TEST'].copy()
        print(f"Filtered to TEST split: {len(test_df)} videos")
    else:
        test_df = df.copy()
        print(f"No 'Split' column found; using full dataset ({len(test_df)} videos)")

    # Find videos existing on disk
    valid_records = []
    for idx, row in test_df.iterrows():
        fname = str(row['FileName']).strip()
        vname = fname if fname.endswith('.avi') else f"{fname}.avi"
        vpath = os.path.join(videos_dir, vname)

        if os.path.exists(vpath):
            valid_records.append({
                'FileName': fname,
                'VideoPath': vpath,
                'GroundTruthEF': float(row['EF'])
            })

    print(f"Available TEST videos on local disk: {len(valid_records)}")
    if len(valid_records) == 0:
        raise FileNotFoundError(f"No test videos found in {videos_dir}")

    # Select sample count
    sample_records = valid_records[:num_samples]
    print(f"Evaluating target sample size: {len(sample_records)} videos")
    print("-" * 80)

    # Initialize model analyzer
    ef_weights = os.path.join(project_root, 'backend', 'models', 'r2plus1d_18_32_2_pretrained.pt')
    seg_weights = os.path.join(project_root, 'backend', 'models', 'deeplabv3_resnet50_random.pt')

    print("Initializing EchoNet-Dynamic Model Wrapper...")
    analyzer = EchoNetAnalyzer(ef_weights, seg_weights)
    print("Model initialized successfully!")
    print("-" * 80)

    results = []
    start_time = time.time()

    for idx, rec in enumerate(sample_records, 1):
        v_id = rec['FileName']
        v_path = rec['VideoPath']
        gt_ef = rec['GroundTruthEF']

        t0 = time.time()
        res = analyzer.analyze(v_path)
        dt = time.time() - t0

        pred_ef = float(res['ef'])
        abs_err = abs(pred_ef - gt_ef)
        num_beats = int(res.get('num_beats', 0))
        ef_std = float(res.get('ef_std', 0.0))
        consistency = str(res.get('consistency', 'UNKNOWN'))

        row_data = {
            'video_id': v_id,
            'ground_truth_ef': round(gt_ef, 2),
            'predicted_ef': round(pred_ef, 2),
            'absolute_error': round(abs_err, 2),
            'num_beats': num_beats,
            'beat_ef_std': round(ef_std, 2),
            'consistency': consistency,
            'inference_time_sec': round(dt, 2)
        }
        results.append(row_data)

        print(f"[{idx:02d}/{len(sample_records):02d}] {v_id:<22} | GT: {gt_ef:5.1f}% | Pred: {pred_ef:5.1f}% | AbsErr: {abs_err:5.2f}% | Beats: {num_beats} | Std: {ef_std:4.1f} | Consistency: {consistency:<8} ({dt:.2f}s)")

    eval_df = pd.DataFrame(results)
    eval_df.to_csv(output_csv_path, index=False)
    print("-" * 80)
    print(f"Evaluation dataset saved to: {output_csv_path}")

    # Compute Summary Statistics
    gt_arr = eval_df['ground_truth_ef'].values
    pred_arr = eval_df['predicted_ef'].values
    abs_err_arr = eval_df['absolute_error'].values

    mae = float(np.mean(abs_err_arr))
    rmse = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
    r_val, p_val = pearsonr(gt_arr, pred_arr)

    median_err = float(np.median(abs_err_arr))
    min_err = float(np.min(abs_err_arr))
    max_err = float(np.max(abs_err_arr))

    consistency_counts = eval_df['consistency'].value_counts().to_dict()
    avg_beats = float(np.mean(eval_df['num_beats']))
    total_time = time.time() - start_time

    print("=" * 80)
    print("                     EVALUATION SUMMARY RESULTS")
    print("=" * 80)
    print(f"Total Test Videos Evaluated : {len(eval_df)}")
    print(f"Mean Absolute Error (MAE)   : {mae:.2f}%")
    print(f"Root Mean Sq Error (RMSE)   : {rmse:.2f}%")
    print(f"Pearson Correlation (r)     : {r_val:.4f} (p-value: {p_val:.4e})")
    print(f"Median Absolute Error       : {median_err:.2f}%")
    print(f"Min Error / Max Error       : {min_err:.2f}% / {max_err:.2f}%")
    print(f"Average Beats per Video     : {avg_beats:.2f}")
    print(f"Rhythm Consistency Counts   : {consistency_counts}")
    print(f"Total Execution Time        : {total_time:.2f} seconds ({total_time/len(eval_df):.2f}s per video)")
    print("=" * 80)

if __name__ == '__main__':
    run_evaluation(num_samples=50)
