
#_______________________TRAINING AREA______________________________________________

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv(r"C:\Users\GENIUS\OneDrive\Desktop\Samayal daww\data\gesture_data.csv")

# Split features and labels
X = data.drop(columns='label', axis=1)  # all columns except label
y = data['label']  # label column

# Encode labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Neural Network model
model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

train_preds=model.predict(X_train)
train_accuracy=accuracy_score(y_train,train_preds)
test_accuracy=accuracy_score(y_test,y_pred)

print(f"The Training accuracy : {train_accuracy}")
print(f"The Testing accuracy : {test_accuracy}")

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#Save the model
joblib.dump(model, 'gesture_model.pkl')
print("Model saved as gesture_model.pkl")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved as label_encoder.pkl")
