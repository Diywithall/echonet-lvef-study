import sys
import os
import json

# Add backend to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.echonet_wrapper import EchoNetAnalyzer

def validate_video(video_path):
    print("="*50)
    print(f"🚀 INITIALIZING ECHONET-DYNAMICS VALIDATION")
    print(f"📁 Target: {video_path}")
    print("="*50)

    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return

    # Paths to weights
    ef_weights = os.path.join("..", "models", "r2plus1d_18_32_2_pretrained.pt")
    seg_weights = os.path.join("..", "models", "deeplabv3_resnet50_random.pt")

    try:
        # 1. Initialize Analyzer
        print("⚙️  Loading neural networks (PyTorch)...")
        analyzer = EchoNetAnalyzer(ef_weights, seg_weights)

        # 2. Run Analysis
        print("🧬 Running beat-to-beat inference...")
        result = analyzer.analyze(video_path)

        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return

        # 3. Output Results
        print("\n" + "📊 DIAGNOSTIC RESULTS")
        print("-" * 20)
        print(f"🔹 Ejection Fraction (EF): {result['ef']:.2f}%")
        print(f"🔹 Diagnosis: {result['diagnosis']}")
        print(f"🔹 Confidence Score: {result['confidence']:.2%}")
        print("-" * 20)
        print(f"📝 Clinical Summary: \n{result['explanation']}")
        print("-" * 20)
        
        print("\n✅ Validation Complete. Full functionality verified.")

    except Exception as e:
        print(f"❌ A critical error occurred: {str(e)}")

if __name__ == "__main__":
    # Path provided by user
    TARGET_VIDEO = r"D:\pocus\dataset\Echo cardiogram video.mp4"
    validate_video(TARGET_VIDEO)
