from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fake_profile_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from HTML
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        result = "Fake Profile Detected" if prediction == 1 else "Real Profile"
        
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
