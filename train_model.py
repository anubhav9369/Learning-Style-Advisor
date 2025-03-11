import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('learning_styles.csv')

# Encode the target variable 'Learner'
label_encoder = LabelEncoder()
data['Learner'] = label_encoder.fit_transform(data['Learner'])

# Separate features and target variable
X = data.drop('Learner', axis=1)
y = data['Learner']

# Convert categorical features to numeric if necessary
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to disk
with open('learning_style_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Display model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
