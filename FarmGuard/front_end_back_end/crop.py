from flask import Flask,request,render_template
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)


crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

def process(Ni, Pi, Ki,temp,humidity, ph, rainfall):
    feature_list = [Ni, Pi, Ki,temp,humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0],prediction[1]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return{
        'Nitrogen':Ni,
        'Phosphorous': Pi,
        'Pottassium':Ki,
        'temperature':temp,
        'humidity':humidity,
        'ph':ph,
        'rainfall':rainfall,
        'result': result
    }
   
@app.route('/')
def index():
    return render_template("crop.html")

@app.route("/predict",methods=['POST'])
def predict():
    Ni= request.form['nitrogen']
    Pi =request.form['phosphorous']
    Ki = request.form['potassium']
    temp =request.form['temperature']
    humidity = request.form['humidity']
    ph = request.form['ph']
    rainfall =request.form['rainfall']
 
    
    result_data=process(Ni,Pi,Ki,temp,humidity,ph,rainfall)
    
    return render_template('result.html',result = result_data)
     
if __name__ == "__main__":
    app.run(debug=True)