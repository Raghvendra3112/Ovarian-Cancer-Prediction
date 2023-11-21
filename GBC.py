from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, confusion_matrix , precision_score, recall_score, f1_score, roc_auc_score, log_loss  # Import metrics

# Load the data
data = pd.read_csv("C:\ZZZZ-MINE\Manipal\Semester 5\DMPA\OC_Marker.csv")
data = data.iloc[:, :-1]  # Update this line

# Separate features and target variable
X = data.drop('TYPE', axis=1)  # Features
y = data['TYPE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1, max_features='sqrt', random_state=42)

# Train the model on the scaled training data
baseline.fit(X_train_scaled, y_train)

# Make predictions on the scaled testing data
gbm_predict_test = baseline.predict(X_test_scaled)
feature_column_names = list(X.columns)
baseline = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=42)

baseline.fit(X_train, y_train)

gbm_predict_train = baseline.predict(X_train)

#get accuracy
gbm_accuracy = metrics.accuracy_score(y_train, gbm_predict_train)

gbm_predict_test = baseline.predict(X_test)

# Display accuracy and other metrics
print("GBM testing Accuracy: {:.4f}".format(metrics.accuracy_score(y_test, gbm_predict_test)))
print("GBM Log Loss: {:.4f}".format(log_loss(y_test, gbm_predict_test)))
print("GBM AUC: {:.4f}".format(roc_auc_score(y_test, gbm_predict_test)))

#get accuracy
gbm_accuracy_testdata = metrics.accuracy_score(y_test, gbm_predict_test)

#print accuracy
print ("GBM testing Accuracy: {0:.4f}".format(gbm_accuracy_testdata))

from sklearn.metrics import log_loss
logloss = log_loss(y_test, gbm_predict_test)
print ("GBM Log Loss: {0:.4f}".format(logloss))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, gbm_predict_test)
print ("GBM AUC: {0:.4f}".format(auc))
importances = baseline.feature_importances_

#Sort it
print ("GBM Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(feature_column_names)), reverse=True)
#print(importances)
print (sorted_feature_importance)
sorted_importances = importances.sort()

print('Training confusion matrix')
print(confusion_matrix(y_train, gbm_predict_train))
print('Testing confusion matrix')
print(confusion_matrix(y_test, gbm_predict_test))

print ("Confusion Matrix for GBM")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, gbm_predict_test, labels=[1, 0])))

print ("")

print ("Classification Report\n")

print ("{0}".format(metrics.classification_report(y_test, gbm_predict_test, labels=[1, 0])))

for i in importances:
    print(i)

# Display top 3 important features
print("Top 3 Important Features:")
for importance, feature in sorted_feature_importance[:3]:
    print(f"{feature}: {importance}")

user_input = {}
for feature in X.columns:
    user_input[feature] = float(input(f"Enter value for {feature}: "))

# Create a DataFrame from user input with explicit column names and order
user_df = pd.DataFrame([user_input], columns=X.columns)

# Scale the user input using the same scaler
user_input_scaled = scaler.transform(user_df)

# Use the trained model to make a prediction on the scaled user input
user_prediction = baseline.predict(user_input_scaled)

# Print the prediction
print("Prediction for the user input:", user_prediction)








