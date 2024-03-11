import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load synthetic data
data_path = 'synthetic_menopausal_data_final.csv'
data = pd.read_csv(data_path)

# Initialize a dictionary to hold multiple label encoders, one for each categorical column
label_encoders = {}

# Convert categorical features into numeric using Label Encoding
for column in ['Removal of Ovaries', 'Last period', 'Contraception', 'Hot flushes/night sweats', 'Aware of cycle']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Now encode the 'Diagnosis' column separately
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(data['Diagnosis'])

# Prepare features and target variable
X = data[['Age', 'Removal of Ovaries', 'Last period', 'Contraception', 'Hot flushes/night sweats', 'Aware of cycle']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print classification report to see how well the classifier performed
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred, target_names=label_encoder_y.classes_))


# Individual Test Case
# Case details
case_data = pd.DataFrame({
    'Age': [49],
    'Removal of Ovaries': ['No'],
    'Last period': ['More than 12 months ago'],
    'Contraception': ['MHT'],
    'Hot flushes/night sweats': ['No'],
    'Aware of cycle': ['No']  
})

# Apply the appropriate label encoders to the new case to encode categorical variables numerically
for column in case_data.columns:
    if column != 'Age':
        case_data[column] = label_encoders[column].transform(case_data[column])

# Convert the DataFrame to match the features format expected by the classifier
case_features = case_data.values

# Make a prediction using the trained classifier
predicted_label_num = clf.predict(case_features)

# Decode the predicted label back into its original form
predicted_label = label_encoder_y.inverse_transform(predicted_label_num)

# Output the prediction
print(f"The predicted diagnosis for the case is: {predicted_label[0]}")