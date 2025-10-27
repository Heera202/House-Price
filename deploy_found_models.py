# deploy_windows.py - Windows compatible MLflow deployment
import os
import subprocess
import sys
import time
import requests
import json
from rich import print
import glob

def find_latest_model():
    """Find the most recent trained model"""
    print("🔍 Searching for trained models...")
    
    search_pattern = r"C:\Users\iamvi\AppData\Roaming\zenml\**\mlruns\**\artifacts\model\MLmodel"
    found_models = []
    
    for match in glob.glob(search_pattern, recursive=True):
        model_dir = os.path.dirname(match)
        if os.path.exists(model_dir):
            mtime = os.path.getmtime(model_dir)
            found_models.append((mtime, model_dir))
    
    if found_models:
        found_models.sort(reverse=True)  # Most recent first
        latest_model = found_models[0][1]
        print(f"✅ Found latest model: {latest_model}")
        return latest_model
    else:
        print("❌ No trained models found!")
        return None

def stop_existing_servers(port=8237):
    """Stop any existing MLflow servers on the same port"""
    try:
        print("🛑 Checking for existing servers...")
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
                        print(f"🛑 Stopping existing process on port {port} (PID: {pid})")
                        subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                        time.sleep(2)
        else:
            print("✅ No existing servers found")
    except Exception as e:
        print(f"⚠️  Error checking existing servers: {e}")

def start_mlflow_server(model_path, port=8237):
    """Start MLflow model server (Windows compatible)"""
    try:
        clean_path = model_path.replace('\\', '/')
        model_uri = f"file:///{clean_path}"
        
        print(f"🚀 Starting MLflow server on port {port}...")
        print(f"📦 Model: {os.path.basename(model_path)}")
        
        # Start MLflow server as a subprocess
        process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "models", "serve",
            "-m", model_uri,
            "-p", str(port),
            "--no-conda",
            "--workers", "1"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        server_started = False
        
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ MLflow server is running!")
                    server_started = True
                    break
            except requests.exceptions.RequestException:
                if i % 5 == 0:
                    print(f"   ...{i+1}/30 seconds")
        
        if not server_started:
            print("❌ Server failed to start within 30 seconds")
            # Print any error messages
            try:
                stdout, stderr = process.communicate(timeout=5)
                if stderr:
                    print(f"🔍 Server errors: {stderr}")
            except subprocess.TimeoutExpired:
                process.terminate()
            return None
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None


def main():
    print("🏠 Windows MLflow Model Deployment")
    print("=" * 50)
    
    # Stop any existing servers first
    stop_existing_servers(8237)
    
    # Find the latest model
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found!")
        print("💡 Please run the training pipeline first:")
        print("   python run_pipeline.py")
        return
    
    # Start MLflow server
    port = 8237
    process = start_mlflow_server(model_path, port)
    
    if process:
        print("\n" + "=" * 50)
        print("🎯 DEPLOYMENT SUCCESSFUL!")
        print("=" * 50)
        print(f"📡 Server URL: http://127.0.0.1:{port}")
        print(f"🔧 Prediction endpoint: /invocations")
        print(f"❤️  Health check: /health")
        
        # Test the prediction endpoint
    
        
        print("\n📝 Now you can run:")
        print("   python predict_sample.py")
        print("\n💡 Make sure your predict_sample.py uses:")
        print("   - URL: http://127.0.0.1:8237/invocations")
        print("   - Data format: dataframe_split")
        print("\n🛑 To stop server: Press Ctrl+C")
        print("=" * 50)
        
        try:
            # Keep the server running
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("✅ Server stopped")
    else:
        print("❌ Failed to start MLflow server!")

if __name__ == "__main__":
    main()