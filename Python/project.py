import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the dataset
# file_path = "C:/Users/Benjamin/development/credit-card-fraud-detection/datatset/card_transdata.csv"
file_path = "/Users/benjaminlindeen/developement/credit-card-fraud-detection/datatset/card_transdata.csv"
data = pd.read_csv(file_path)

# Log transformation
small_constant = 1e-7
data['distance_from_home_log'] = np.log(data['distance_from_home'].clip(lower=small_constant))
data['distance_from_last_transaction_log'] = np.log(data['distance_from_last_transaction'].clip(lower=small_constant))
data['ratio_to_median_purchase_price_log'] = np.log(data['ratio_to_median_purchase_price'].clip(lower=small_constant))

# Drop original skewed variables
data.drop(['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price'], axis=1, inplace=True)

# Extracting features and target variable
X = data.drop('fraud', axis=1)
y = data['fraud']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Calculating class weights
class_weights = {0: 1 / y_train.value_counts()[0], 1: 1 / y_train.value_counts()[1]}

# Building the ANN model
model_smote = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_smote.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile and fit the model with early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

model_smote.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model_smote.fit(X_train_smote, y_train_smote, class_weight=class_weights, batch_size=64, epochs=100, validation_split=0.1, callbacks=[early_stopping])


# Evaluate the model on test data
loss_smote, accuracy_smote = model_smote.evaluate(X_test_scaled, y_test)
print(f"SMOTE Test Accuracy: {accuracy_smote*100:.2f}%")

# Making predictions on the test data
y_pred_smote = model_smote.predict(X_test_scaled)
y_pred_classes_smote = (y_pred_smote > 0.5).astype("int32")

# Calculating the confusion matrix
conf_matrix_smote = confusion_matrix(y_test, y_pred_classes_smote)

# Calculating precision, recall, F1-score, and ROC-AUC
classification_rep_smote = classification_report(y_test, y_pred_classes_smote)
roc_auc_smote = roc_auc_score(y_test, y_pred_smote)

print("Confusion Matrix:\n", conf_matrix_smote)
print("\nClassification Report:\n", classification_rep_smote)
print(f"ROC-AUC Score: {roc_auc_smote:.2f}")

