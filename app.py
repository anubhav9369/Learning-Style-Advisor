from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('learning_style_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the home route to render the form
@app.route('/')
def home():
    return render_template('index.html')  # HTML form

# Define the predict route to get user input and make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect responses from the form as feature inputs
    try:
        responses = [int(request.form[f'feature{i}']) for i in range(1, 19)]
    except KeyError:
        return "Incomplete form submission, please ensure all fields are filled."

    # Convert the responses to a numpy array and reshape for model input
    features = np.array(responses).reshape(1, -1)

    # Predict learning style using the model
    prediction = model.predict(features)

    # Map numeric predictions back to readable labels if required
    label_map = {0: "Visual Learner", 1: "Auditory Learner", 2: "Kinesthetic Learner", 3: "Mixed Learning Style"}
    result = label_map.get(prediction[0], "Unknown Learning Style")

    # Render the result in the result template
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
