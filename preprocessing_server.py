import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import mlflow.pyfunc
import sys
import os

print("Starting preprocessing server...")

# Load the model
try:
    model_path = "C:/Users/iamvi/AppData/Roaming/zenml/local_stores/d6f4feec-01c1-45f4-b4d2-80dc989762f4/mlruns/489494737275752969/e78110fca4f74a7ca0477d54dec2e2cb/artifacts/model"
    print(f"Loading model from: {model_path}")
    model = mlflow.pyfunc.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

app = Flask(__name__)

def preprocess_data(input_data):
    """Apply the same preprocessing as during training"""
    # Create DataFrame
    df = pd.DataFrame(input_data)
    
    print(f"Input data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Apply the same preprocessing steps as in your training pipeline
    # 1. Handle missing values
    numerical_cols = [
        "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area", 
        "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", 
        "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", 
        "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", 
        "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", 
        "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", 
        "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area", 
        "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", 
        "Screen Porch", "Pool Area", "Misc Val", "Mo Sold", "Yr Sold"
    ]
    
    # Fill missing values
    for col in numerical_cols:
        if col in df.columns and df[col].isna().any():
            if col == 'Lot Frontage': 
                df[col] = df[col].fillna(70)
            elif col == 'Mas Vnr Area': 
                df[col] = df[col].fillna(0)
            elif col == 'Garage Yr Blt': 
                df[col] = df[col].fillna(1978)
            else: 
                df[col] = df[col].fillna(0)
    
    # 2. Apply feature engineering (log transformation)
    if 'Gr Liv Area' in df.columns:
        print(f"Applying log transformation to Gr Liv Area")
        original_value = df['Gr Liv Area'].iloc[0]
        df['Gr Liv Area'] = np.log1p(df['Gr Liv Area'])
        print(f"Before: {original_value}, After: {df['Gr Liv Area'].iloc[0]:.2f}")
    
    # 3. Ensure correct column order
    expected_columns = [
        "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area", 
        "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", 
        "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", 
        "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", 
        "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", 
        "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", 
        "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area", 
        "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", 
        "Screen Porch", "Pool Area", "Misc Val", "Mo Sold", "Yr Sold"
    ]
    
    # Add any missing columns
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
            print(f"Added missing column: {col}")
    
    # Reorder columns
    df = df[expected_columns]
    
    print(f"Preprocessing completed. Final shape: {df.shape}")
    return df

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        print("Received prediction request")
        
        if 'dataframe_records' in data:
            # Convert to the format our model expects
            input_data = data['dataframe_records']
            print(f"Preprocessing {len(input_data)} records...")
            
            processed_df = preprocess_data(input_data)
            
            # Make prediction
            print("Making prediction...")
            predictions = model.predict(processed_df)
            raw_prediction = predictions[0]
            print(f"Raw prediction: {raw_prediction}")
            
            # Ensure positive predictions
            final_prediction = max(raw_prediction, 50000)  # Minimum $50,000
            print(f"Final prediction: ${final_prediction:,.2f}")
            
            return jsonify({"predictions": [final_prediction]})
        else:
            error_msg = "Expected 'dataframe_records' format in input data"
            print(f"Error: {error_msg}")
            return jsonify({"error": error_msg}), 400
            
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"Error: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint with sample data"""
    sample_data = {
        "dataframe_records": [
            {
                "Order": 1, "PID": 5286, "MS SubClass": 20, "Lot Frontage": 80.0, 
                "Lot Area": 9600, "Overall Qual": 5, "Overall Cond": 7, 
                "Year Built": 1961, "Year Remod/Add": 1961, "Mas Vnr Area": 0.0, 
                "BsmtFin SF 1": 700.0, "BsmtFin SF 2": 0.0, "Bsmt Unf SF": 150.0, 
                "Total Bsmt SF": 850.0, "1st Flr SF": 856, "2nd Flr SF": 854, 
                "Low Qual Fin SF": 0, "Gr Liv Area": 1710.0, "Bsmt Full Bath": 1, 
                "Bsmt Half Bath": 0, "Full Bath": 1, "Half Bath": 0, 
                "Bedroom AbvGr": 3, "Kitchen AbvGr": 1, "TotRms AbvGrd": 7, 
                "Fireplaces": 2, "Garage Yr Blt": 1961, "Garage Cars": 2, 
                "Garage Area": 500.0, "Wood Deck SF": 210.0, "Open Porch SF": 0, 
                "Enclosed Porch": 0, "3Ssn Porch": 0, "Screen Porch": 0, 
                "Pool Area": 0, "Misc Val": 0, "Mo Sold": 5, "Yr Sold": 2010
            }
        ]
    }
    
    try:
        processed_df = preprocess_data(sample_data["dataframe_records"])
        prediction = model.predict(processed_df)[0]
        prediction = max(prediction, 50000)
        return jsonify({
            "test_prediction": prediction,
            "formatted_price": f"${prediction:,.2f}",
            "status": "test_successful"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "test_failed"})

if __name__ == '__main__':
    print("Server starting on http://127.0.0.1:8237")
    app.run(host='127.0.0.1', port=8237, debug=False)
