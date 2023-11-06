import requests

API_URL = "http://127.0.0.1:8000/predict"

def get_prediction(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        print("Error:", response.status_code)
        return None

if __name__ == "__main__":
    sample_data = {
        "Age": 3,                
        "HbA1c": 1,              
        "Genetic_Info": 0,       
        "Family_History": 0,     
        "Birth_Weight": 2.045,     
        "Developmental_Delay": 0, 
        "Insulin_Level": 1.65 
    }

    prediction = get_prediction(sample_data)
    if prediction is not None:
        print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'} for PNDM")

