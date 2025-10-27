# test_endpoints.py
import requests
import json

def test_endpoints():
    base_url = "http://127.0.0.1:8237"
    
    print("Testing MLflow Endpoints:")
    print("=" * 50)
    
    # Test 1: GET /health (should work)
    print("1. Testing GET /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ SUCCESS - Server is healthy")
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        return False
    
    # Test 2: GET /invocations (should fail)
    print("2. Testing GET /invocations...")
    try:
        response = requests.get(f"{base_url}/invocations", timeout=5)
        print(f"   ❌ EXPECTED FAILURE - GET not allowed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
    
    # Test 3: POST /invocations (should work)
    print("3. Testing POST /invocations...")
    try:
        test_data = {
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
        
        response = requests.post(
            f"{base_url}/invocations",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            price = result['predictions'][0]
            print(f"   ✅ SUCCESS - Prediction: ${price:,.2f}")
            return True
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        return False

if __name__ == "__main__":
    test_endpoints()