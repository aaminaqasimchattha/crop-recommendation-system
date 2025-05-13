from flask import Flask, request, render_template
import numpy as np
import pickle

# Load your saved scalers and model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('minmaxscaler.pkl', 'rb') as file:
    ms = pickle.load(file)

with open('standscaler.pkl', 'rb') as file:
    sc = pickle.load(file)

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        feature_list = [Nitrogen, Phosphorus, Potassium, temperature, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        predicted_crop = prediction[0]
        result = f"{predicted_crop.capitalize()}"

        return render_template('index.html', prediction=result,
                               Nitrogen=Nitrogen, Phosphorus=Phosphorus, Potassium=Potassium,
                               temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall)

    except Exception as e:
        return render_template('index.html', prediction=f"Error occurred: {str(e)}",
                               Nitrogen=request.form.get('Nitrogen', ''),
                               Phosphorus=request.form.get('Phosphorus', ''),
                               Potassium=request.form.get('Potassium', ''),
                               temperature=request.form.get('temperature', ''),
                               humidity=request.form.get('humidity', ''),
                               ph=request.form.get('ph', ''),
                               rainfall=request.form.get('rainfall', ''))

if __name__ == "__main__":
    app.run(debug=True)