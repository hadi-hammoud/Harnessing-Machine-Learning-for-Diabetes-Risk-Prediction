from flask import Flask, render_template, request
import numpy as np
import joblib
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model and scaler
model = joblib.load('random_forest_model.pkl')  # Load the trained model

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form inputs
        age = int(request.form.get('age', 0))  # Default age to 0 if missing
        gender = 1 if request.form.get('gender', "Male") == "Male" else 0
        polyuria = int(request.form.get('polyuria', 0))
        polydipsia = int(request.form.get('polydipsia', 0))
        sudden_weight_loss = int(request.form.get('sudden_weight_loss', 0))
        weakness = int(request.form.get('weakness', 0))
        polyphagia = int(request.form.get('polyphagia', 0))
        genital_thrush = int(request.form.get('genital_thrush', 0))
        visual_blurring = int(request.form.get('visual_blurring', 0))
        itching = int(request.form.get('itching', 0))
        irritability = int(request.form.get('irritability', 0))
        delayed_healing = int(request.form.get('delayed_healing', 0))
        partial_paresis = int(request.form.get('partial_paresis', 0))
        muscle_stiffness = int(request.form.get('muscle_stiffness', 0))
        alopecia = int(request.form.get('alopecia', 0))
        obesity = int(request.form.get('obesity', 0))

        # Combine features into an array
        features = np.array([age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
                             polyphagia, genital_thrush, visual_blurring, itching, irritability,
                             delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity])

        # Log input features for debugging
        logging.debug(f"Input Features: {features}")

        # Make prediction
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0][1] * 100  # Probability for class 1

        # Log prediction and confidence for debugging
        logging.debug(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")

        # Render template with prediction and confidence
        return render_template('index.html', prediction="Positive" if prediction == 1 else "Negative", confidence=round(confidence, 2))
    except Exception as e:
        logging.error("Error during prediction:", exc_info=True)
        return render_template('index.html', error="An error occurred during prediction. Please check your inputs.")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
