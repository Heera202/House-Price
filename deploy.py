# simple_deploy.py - Minimal working deployment
import os
import subprocess
import sys
import time
import requests
import json
import glob

def find_latest_model():
    """Find the most recent trained model"""
    print("Searching for trained models...")
    
    search_pattern = r"C:\Users\iamvi\AppData\Roaming\zenml\**\mlruns\**\artifacts\model\MLmodel"
    found_models = []
    
    for match in glob.glob(search_pattern, recursive=True):
        model_dir = os.path.dirname(match)
        if os.path.exists(model_dir):
            mtime = os.path.getmtime(model_dir)
            found_models.append((mtime, model_dir))
    
    if found_models:
        found_models.sort(reverse=True)
        latest_model = found_models[0][1]
        print(f"Found model: {latest_model}")
        return latest_model
    else:
        print("No trained models found! Run: python run_pipeline.py")
        return None

def stop_existing_servers(port=8237):
    """Stop any existing servers"""
    try:
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                        time.sleep(2)
    except:
        pass

def start_mlflow_server(model_path, port=8237):
    """Start MLflow server directly"""
    try:
        clean_path = model_path.replace('\\', '/')
        model_uri = f"file:///{clean_path}"
        
        print(f"Starting MLflow server on port {port}...")
        
        process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "models", "serve",
            "-m", model_uri,
            "-p", str(port),
            "--no-conda",
            "--workers", "1"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server
        print("Waiting for server to start...")
        for i in range(20):
            time.sleep(1)
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    print("Server is running!")
                    return process
            except:
                if i % 5 == 0:
                    print(f"...{i+1}/20 seconds")
        
        print("Server failed to start")
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None



def main():
    print("House Price Prediction Deployment")
    print("=" * 50)
    
    # Stop existing servers
    stop_existing_servers(8237)
    
    # Find model
    model_path = find_latest_model()
    if not model_path:
        return
    
    # Start server
    port = 8237
    process = start_mlflow_server(model_path, port)
    
    if process:
        print("\n" + "=" * 50)
        print("SERVER IS RUNNING!")
        print("=" * 50)
        print("Test the server with:")
        print("   python predict_sample.py")
        print("\nOr use this test prediction:")
        
  
        print("\nServer will stay running. Press Ctrl+C to stop.")
        print("=" * 50)
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping server...")
            process.terminate()
            print("Server stopped")
    else:
        print("Failed to start server!")

if __name__ == "__main__":
    main()