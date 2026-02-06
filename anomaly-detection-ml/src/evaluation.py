import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score
from datetime import datetime

def generate_data(num_samples):
    data = []
    for _ in range(num_samples):
        try:
            speed = np.random.uniform(low=500, high=1000)
            temperature = np.random.uniform(low=20, high=30)
            voltage = np.random.uniform(low=5, high=20)
            current = np.random.uniform(low=1, high=5)
            data.append({'Time': datetime.now(), 'Speed': speed, 'Temperature': temperature, 'Voltage': voltage, 'Current': current})
        except Exception as e:
            print(f"Error generating data: {e}")
    return data

# Generate data
num_samples = 1000
data = generate_data(num_samples)
df = pd.DataFrame(data)
print(df)
#Split data into training and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Train Isolation Forest models
models = {column: IsolationForest(contamination=0.1, random_state=42) for column in ['Speed', 'Temperature', 'Voltage', 'Current']}
for column in ['Speed', 'Temperature', 'Voltage', 'Current']:
    models[column].fit(train[[column]])
# Evaluate models
for column in ['Speed', 'Temperature', 'Voltage', 'Current']:
    anomaly_scores = models[column].decision_function(test[[column]])
    threshold = np.percentile(anomaly_scores, 100 * 0.1)  # Set threshold dynamically based on anomaly scores percentile
    preds = np.where(anomaly_scores > threshold, 1, 0)  # Label anomalies as 1 and normal instances as 0

    # Convert test labels to binary (0 for normal, 1 for anomalies)
    test_labels = np.where(test[column] > threshold, 1, 0)

    # Calculate evaluation metrics

    precision = precision_score(test_labels, preds, zero_division=1)
    recall = recall_score(test_labels, preds, zero_division=1)
    f1 = f1_score(test_labels, preds, zero_division=1)
    accuracy = accuracy_score(test_labels, preds )

    # Print evaluation metrics

    print(f"Model Performance for {column}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


