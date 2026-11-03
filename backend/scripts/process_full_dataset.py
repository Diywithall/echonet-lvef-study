import sys
import os
import pandas as pd
from tqdm import tqdm
import time

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.echonet_wrapper import EchoNetAnalyzer

def batch_process(dataset_dir, output_csv):
    print(f"📦 BATCH PROCESSOR STARTING")
    print(f"📁 Dataset Directory: {dataset_dir}")
    
    # 1. Gather all videos
    valid_extensions = ('.mp4', '.avi', '.mov')
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                video_files.append(os.path.join(root, file))
    
    print(f"🔍 Found {len(video_files)} echocardiogram videos.")
    
    if not video_files:
        print("❌ No videos found. Exiting.")
        return

    # 2. Init Analyzer
    ef_weights = os.path.join("..", "models", "r2plus1d_18_32_2_pretrained.pt")
    seg_weights = os.path.join("..", "models", "deeplabv3_resnet50_random.pt")
    
    print("⚙️  Initializing Backend Engine...")
    analyzer = EchoNetAnalyzer(ef_weights, seg_weights)
    
    results = []
    
    # 3. Process with Progress Bar
    print("🚀 Starting inference loop...")
    for video_path in tqdm(video_files, desc="Analyzing Videos"):
        try:
            start_time = time.time()
            data = analyzer.analyze(video_path)
            processing_time = time.time() - start_time
            
            if "error" not in data:
                results.append({
                    "filename": os.path.basename(video_path),
                    "path": video_path,
                    "ef": data["ef"],
                    "diagnosis": data["diagnosis"],
                    "processing_time_sec": processing_time
                })
        except Exception as e:
            print(f"⚠️ Error processing {video_path}: {e}")
            continue
            
    # 4. Save Results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print("="*50)
    print(f"✅ BATCH ANALYSIS FINISHED")
    print(f"📄 Report saved to: {output_csv}")
    print(f"📊 Processed: {len(results)}/{len(video_files)} videos")
    print(f"📈 Average EF: {df['ef'].mean():.2f}%")
    print("="*50)

if __name__ == "__main__":
    DATASET_PATH = r"D:\pocus\dataset\Echocardiogram Dataset"
    REPORT_PATH = r"D:\pocus\dataset_analysis_report.csv"
    
    batch_process(DATASET_PATH, REPORT_PATH)
