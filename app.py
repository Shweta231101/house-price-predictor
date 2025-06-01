from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get inputs from form
            MedInc = float(request.form['MedInc'])
            HouseAge = float(request.form['HouseAge'])
            AveRooms = float(request.form['AveRooms'])
            AveBedrms = float(request.form['AveBedrms'])
            Population = float(request.form['Population'])
            AveOccup = float(request.form['AveOccup'])
            Latitude = float(request.form['Latitude'])
            Longitude = float(request.form['Longitude'])
            
            # Prepare features for prediction
            features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
            
            # Predict
            pred = model.predict(features)
            
            prediction = round(pred[0], 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
