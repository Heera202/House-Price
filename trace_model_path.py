# find_real_mlruns.py
import os
import glob
from rich import print

def find_real_mlruns():
    print("🔍 Finding REAL MLruns with experiments...")
    print("=" * 70)
    
    # Search everywhere for mlruns directories that actually have runs
    search_patterns = [
        r"C:\Users\iamvi\**\mlruns",
        r"D:\**\mlruns", 
        r"**\mlruns"
    ]
    
    found_locations = []
    
    for pattern in search_patterns:
        try:
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                if os.path.isdir(match):
                    # Check if this mlruns has REAL experiments with runs
                    experiments = [d for d in os.listdir(match) if d.isdigit()]
                    
                    total_runs_with_artifacts = 0
                    
                    for exp in experiments:
                        exp_path = os.path.join(match, exp)
                        runs_path = os.path.join(exp_path, "runs")
                        
                        if os.path.exists(runs_path):
                            runs = os.listdir(runs_path)
                            for run in runs:
                                run_path = os.path.join(runs_path, run)
                                # Check if run has artifacts (real training run)
                                artifacts_path = os.path.join(run_path, "artifacts")
                                if os.path.exists(artifacts_path):
                                    total_runs_with_artifacts += 1
                    
                    if total_runs_with_artifacts > 0:
                        found_locations.append({
                            'path': match,
                            'experiments': len(experiments),
                            'runs_with_artifacts': total_runs_with_artifacts
                        })
                        print(f"✅ FOUND: {match}")
                        print(f"   📊 Experiments: {len(experiments)}, Runs with artifacts: {total_runs_with_artifacts}")
                        
        except Exception as e:
            continue
    
    print("\n" + "=" * 70)
    print("📊 SEARCH RESULTS:")
    print("=" * 70)
    
    if found_locations:
        # Sort by most runs
        found_locations.sort(key=lambda x: x['runs_with_artifacts'], reverse=True)
        
        for i, loc in enumerate(found_locations, 1):
            print(f"\n{i}. {loc['path']}")
            print(f"   📁 Experiments: {loc['experiments']}")
            print(f"   🏃 Runs with artifacts: {loc['runs_with_artifacts']}")
            
            clean_path = loc['path'].replace('\\', '/')
            print(f"   🎯 Command: mlflow ui --backend-store-uri \"file:///{clean_path}\" --port 5000")
        
        print(f"\n🎯 RECOMMENDED: Use location #1")
        best_path = found_locations[0]['path'].replace('\\', '/')
        print(f"💡 Run: mlflow ui --backend-store-uri \"file:///{best_path}\" --port 5000")
        
    else:
        print("❌ No MLruns directories with actual training runs found!")
        print("💡 You need to run training first: python run_pipeline.py")
    
    return found_locations

if __name__ == "__main__":
    find_real_mlruns()