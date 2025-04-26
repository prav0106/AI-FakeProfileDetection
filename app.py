from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")  # Optional homepage

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get 11 inputs from the form
            inputs = [
                float(request.form['num_followers']),
                float(request.form['num_following']),
                float(request.form['num_photos']),
                float(request.form['num_videos']),
                float(request.form['account_age_days']),
                float(request.form['avg_likes']),
                float(request.form['avg_comments']),
                float(request.form['engagement_rate']),
                float(request.form['has_profile_picture']),
                float(request.form['is_verified']),
                float(request.form['username_length'])
            ]

            # # Add 4 default values for the missing features
            # inputs += [
            #     0.0,  # bio_length
            #     0.0,  # username_length
            #     0.0,  # external_url_present
            #     0.0   # activity_score
            # ]

            features = np.array(inputs).reshape(1, -1)
            prediction = model.predict(features)[0]

            result = "Fake Account ❌" if prediction == 1 else "Genuine Account ✅"
            return render_template("result.html", result=result)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
