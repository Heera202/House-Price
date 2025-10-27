# find_exact_path.py
import os
import glob

def find_exact_mlruns_path():
    print("🔍 Finding your exact MLruns path...")
    
    # Common ZenML storage locations
    search_locations = [
        r"C:\Users\iamvi\AppData\Roaming\zenml",
        r"C:\Users\iamvi\zenml", 
        os.getcwd()  # Current directory
    ]
    
    for location in search_locations:
        if os.path.exists(location):
            print(f"\n📁 Searching in: {location}")
            
            # Look for mlruns directories
            mlruns_paths = glob.glob(os.path.join(location, "**", "mlruns"), recursive=True)
            
            for mlruns_path in mlruns_paths:
                if os.path.isdir(mlruns_path):
                    # Check if it has experiments
                    items = os.listdir(mlruns_path)
                    experiments = [d for d in items if d.isdigit()]
                    
                    if experiments:
                        print(f"✅ FOUND! Path: {mlruns_path}")
                        print(f"   Experiments: {len(experiments)}")
                        
                        # Show the exact command to use
                        clean_path = mlruns_path.replace('\\', '/')
                        print(f"\n🎯 RUN THIS COMMAND:")
                        print(f'mlflow ui --backend-store-uri "file:///{clean_path}" --port 5000')
                        return mlruns_path
    
    print("❌ No MLruns directory found with experiments!")
    return None

if __name__ == "__main__":
    find_exact_mlruns_path()