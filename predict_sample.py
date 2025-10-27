# predict_sample.py - FIXED VERSION
import json
import requests
from rich import print

# URL of the MLflow prediction server - CORRECT ENDPOINT
url = "http://127.0.0.1:8237/invocations"

# Sample input data for prediction - CORRECT FORMAT
input_data = {
    "dataframe_split": {
        "columns": [
            "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area", 
            "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", 
            "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", 
            "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", 
            "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", 
            "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", 
            "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area", 
            "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", 
            "Screen Porch", "Pool Area", "Misc Val", "Mo Sold", "Yr Sold"
        ],
        "data": [[
            1, 5286, 20, 80.0, 9600, 5, 7, 1961, 1961, 0.0, 
            700.0, 0.0, 150.0, 850.0, 856, 854, 0, 1710.0, 1, 0, 
            1, 0, 3, 1, 7, 2, 1961, 2, 500.0, 210.0, 
            0, 0, 0, 0, 0, 0, 5, 2010
        ]]
    }
}

# Convert the input data to JSON format
json_data = json.dumps(input_data)

# Set the headers for the request
headers = {"Content-Type": "application/json"}

print("🏠 Making House Price Prediction...")
print("=" * 40)

try:
    # Send the POST request to the server
    response = requests.post(url, headers=headers, data=json_data, timeout=10)

    # Check the response status code
    if response.status_code == 200:
        # If successful, print the prediction result
        prediction = response.json()
        predicted_price = prediction['predictions'][0]
        
        if predicted_price < 0:
            print(f"⚠️  Predicted Price: ${predicted_price:,.2f}")
            print("💡 Note: Negative price may indicate data scaling issues")
        else:
            print(f"🎯 Predicted House Price: ${predicted_price:,.2f}")
            
    else:
        # If there was an error, print the status code and the response
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Cannot connect to the server")
    print("💡 Make sure the MLflow server is running on port 8237")
    
except requests.exceptions.Timeout:
    print("❌ Timeout Error: Request took too long")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("=" * 40)