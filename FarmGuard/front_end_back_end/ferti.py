from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load models and scalers
model = pickle.load(open('fermodel.pkl', 'rb'))
sc = pickle.load(open('ferstandscaler.pkl', 'rb'))
mx = pickle.load(open('ferminmaxscaler.pkl', 'rb'))

# Crop and soil types (all in lowercase)
crop_types = {0: 'barley', 1: 'cotton', 2: 'ground nuts', 3: 'maize', 4: 'millets', 5: 'oil seeds', 6: 'paddy', 7: 'pulses', 8: 'sugarcane', 9: 'tobacco', 10: 'wheat'}
soil_types = {0: 'black', 1: 'clayey', 2: 'loamy', 3: 'red', 4: 'sandy'}

# Initialize and fit the label encoders
crop_label_encoder = LabelEncoder()
crop_label_encoder.fit(list(crop_types.values()))

soil_label_encoder = LabelEncoder()
soil_label_encoder.fit(list(soil_types.values()))

# Fertilizer dictionary
fer_dict = {0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}

def process(temperature, humidity, moister, soil_type, crop_type, nitrogen, phosphorus, potassium):
    try:
        encoded_crop_type = crop_label_encoder.transform([crop_type])[0]
    except ValueError:
        return {'error': f"Invalid crop type '{crop_type}'. Expected one of: {', '.join(crop_label_encoder.classes_)}"}

    try:
        encoded_soil_type = soil_label_encoder.transform([soil_type])[0]
    except ValueError:
        return {'error': f"Invalid soil type '{soil_type}'. Expected one of: {', '.join(soil_label_encoder.classes_)}"}

    feature_list = [temperature, humidity, moister, encoded_soil_type, encoded_crop_type, nitrogen, phosphorus, potassium]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    if prediction[0] in fer_dict:
        fer = fer_dict[prediction[0]]
        result = f"{fer} is the best fertilizer to be used."
    else:
        result = "Sorry, we could not determine the best fertilizer to be used with the provided data."
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'moister': moister,
        'soil_type': soil_type,
        'crop_type': crop_type,
        'Nitrogen': nitrogen,
        'Phosphorous': phosphorus,
        'Pottassium': potassium,
        'result': result
    }

@app.route('/')
def index():
    return render_template("ferti.html")

@app.route("/submit", methods=['POST'])
def submit():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    moister = float(request.form['moister'])
    soil_type = request.form['soil_type']
    crop_type = request.form['crop_type']
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
   
    result_data = process(temperature, humidity, moister, soil_type, crop_type, nitrogen, phosphorus, potassium)
    
    return render_template('result1.html', result=result_data)

if __name__ == "__main__":
    app.run(debug=True)
