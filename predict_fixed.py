# predict_fixed.py - Use this for better predictions
import json
import requests
import pandas as pd
import numpy as np

def preprocess_features(features_dict):
    """Apply the same preprocessing as training"""
    df = pd.DataFrame([features_dict])
    
    # Apply log transformation (same as your training pipeline)
    if 'Gr Liv Area' in df.columns:
        df['Gr Liv Area'] = np.log1p(df['Gr Liv Area'])
    
    return df

def predict_with_preprocessing(house_features, port=8237):
    """Make prediction with proper preprocessing"""
    
    # Preprocess the data
    processed_df = preprocess_features(house_features)
    
    # Prepare data for MLflow
    input_data = {
        "dataframe_split": {
            "columns": processed_df.columns.tolist(),
            "data": processed_df.values.tolist()
        }
    }
    
    url = f"http://127.0.0.1:{port}/invocations"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(input_data), timeout=10)
        
        if response.status_code == 200:
            prediction = response.json()
            return prediction['predictions'][0]
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request error: {e}")
        return None

def main():
    print("House Price Prediction (Fixed)")
    print("=" * 40)
    
    # Better house features
    house_features = {
        "Order": 1, "PID": 5286, "MS SubClass": 60, "Lot Frontage": 65, 
        "Lot Area": 9000, "Overall Qual": 7, "Overall Cond": 5, 
        "Year Built": 2003, "Year Remod/Add": 2003, "Mas Vnr Area": 200, 
        "BsmtFin SF 1": 800, "BsmtFin SF 2": 0, "Bsmt Unf SF": 200, 
        "Total Bsmt SF": 1000, "1st Flr SF": 1200, "2nd Flr SF": 600, 
        "Low Qual Fin SF": 0, "Gr Liv Area": 1800, "Bsmt Full Bath": 1, 
        "Bsmt Half Bath": 0, "Full Bath": 2, "Half Bath": 1, 
        "Bedroom AbvGr": 3, "Kitchen AbvGr": 1, "TotRms AbvGrd": 7, 
        "Fireplaces": 1, "Garage Yr Blt": 2003, "Garage Cars": 2, 
        "Garage Area": 480, "Wood Deck SF": 200, "Open Porch SF": 50, 
        "Enclosed Porch": 0, "3Ssn Porch": 0, "Screen Porch": 0, 
        "Pool Area": 0, "Misc Val": 0, "Mo Sold": 6, "Yr Sold": 2010
    }
    
    print("Making prediction with preprocessing...")
    prediction = predict_with_preprocessing(house_features)
    
    if prediction is not None:
        if prediction < 0:
            print(f"Predicted Price: ${prediction:,.2f}")
            print("\nNote: Model needs retraining for better predictions")
        else:
            print(f"Predicted House Price: ${prediction:,.2f}")
    
    print("=" * 40)

def test_connection():
    print("Testing server connection...")
    
    # Test 1: Check if server is reachable
    try:
        health_response = requests.get("http://127.0.0.1:8237/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is running and healthy")
            return True
        else:
            print(f"❌ Server responded with: {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server - it's not running")
        print("💡 Run: python simple_deploy.py")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    test_connection()
    main()