# Learning Style Predictor

This project predicts a user's learning style (Visual, Auditory, Kinesthetic, or Mixed) based on their responses to a set of questions. The model is trained using a dataset of responses, and the prediction is made using a Random Forest Classifier.

## Project Structure

- `app.py`: A Flask web application that allows users to input their responses and get predictions.
- `train_model.py`: A script to train the model using the dataset and save it as a `.pkl` file.
- `learning_style_model.pkl`: The trained model file.
- `learning_styles.csv`: The dataset used for training the model.
- `index.html`: The HTML form for collecting user input.
- `result.html`: The HTML page to display the predicted learning style.

## How to Use

### Running the Application

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/learning-style-predictor.git
   cd learning-style-predictor
   
2. Install required dependencies
   
pip install -r requirements.txt
