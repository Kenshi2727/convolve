# Step 1: Importing libraries
import pandas as pd                 # To handle data (e.g., reading CSV, cleaning data)
from sklearn.model_selection import train_test_split  # To split data into training and test sets
from sklearn.preprocessing import StandardScaler      # To standardize features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score  # To evaluate the model performance

# Step 2: Load the data
dev_data = pd.read_csv("Dev_data_to_be_shared.csv")  # Load your development data (the training data)

# Step 3: Preprocess the data
# Check for missing values and handle them
# dev_data = dev_data.fillna(dev_data.mean())  # Fill missing values with the mean of each column (this is a simple approach)

# Step 4: Split data into features (X) and target (y)
X = dev_data.drop(columns=["account_number", "bad_flag"])  # Drop columns that are not features
y = dev_data["bad_flag"]  # Target column is 'bad_flag', which indicates default or not

# Step 5: Split the data into training and testing sets (we will train the model on 80% of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% for training, 20% for testing

# Step 6: Standardize the features (important for many machine learning models)
scaler = StandardScaler()  # Create a scaler object
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler to the training data and transform it
X_test_scaled = scaler.transform(X_test)  # Use the same scaler to transform the test data

# Step 7: Create and train the Logistic Regression model
model = HistGradientBoostingClassifier()  # Create the logistic regression model
model.fit(X_train_scaled, y_train)  # Train the model on the scaled training data

# Step 8: Make predictions on the test data
y_pred = model.predict(X_test_scaled)  # Predict using the trained model

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])  # Calculate ROC AUC score (a good metric for binary classification)

# Step 10: Output the results
print(f"Accuracy: {accuracy:.4f}")  # Print the accuracy score
print(f"ROC AUC Score: {roc_auc:.4f}")  # Print the ROC AUC score

# Step 11: Make predictions on the validation data
validation_data = pd.read_csv("Validation_data_to_be_shared.csv")  # Load the validation data
X_validation = validation_data.drop(columns=["account_number"])  # Drop the account_number column, we don't need it for predictions
X_validation_scaled = scaler.transform(X_validation)  # Standardize the validation data

# Step 12: Predict the probability of default for the validation data
validation_predictions = model.predict_proba(X_validation_scaled)[:, 1]  # Get the probability of default (class 1)

# Step 13: Create the submission file
submission = pd.DataFrame({"account_number": validation_data["account_number"], "predicted_probability": validation_predictions})
submission.to_csv("predictions.csv", index=False)  # Save the predictions to a CSV file
