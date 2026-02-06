import socket
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Server setup
HOST = '127.0.0.1'  # Adjust as necessary
PORT = 65432

# Initialize server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

print("Server is listening for incoming connections...")

# Accept connections
clientsocket, address = server_socket.accept()
print(f'Connected to: {address}')


# Function to receive data from the client
def receive_data(sock):
    try:
        serialized_data = sock.recv(4096)
        if not serialized_data:
            return None
        return pickle.loads(serialized_data)
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None


# Initialize Isolation Forest models for anomaly detection
models = {column: IsolationForest(contamination=0.1, random_state=42) for column in
          ['Speed', 'Temperature', 'Voltage', 'Current']}

# Initialize an empty DataFrame to store real-time data
real_time_data = pd.DataFrame(columns=['Date', 'Time', 'Speed', 'Temperature', 'Voltage', 'Current'])

# Dictionary to store anomaly scores and information for history tracking
anomaly_scores_history = {column: pd.DataFrame(columns=['Date', 'Time', 'Data Point Value', 'Anomaly Scores']) for
                          column in ['Speed', 'Temperature', 'Voltage', 'Current']}

# Initialize the next save time for Excel file
next_save_time = datetime.now() + timedelta(minutes=1)  # First save time
save_interval = timedelta(minutes=1)  # Interval for saving Excel files
file_count = 0  # Counter for naming Excel files

# Initialize subplots for plotting anomaly detection results
plt.ion()
fig, axes = plt.subplots(4, 1, figsize=(10, 10))

# Continuously receive and process data
while True:
    # Receive data from client
    received_data = receive_data(clientsocket)
    if received_data is None:
        print("No more data received. Closing connection.")
        clientsocket.close()
        break  # Break if no more data received

    # Convert received data to DataFrame
    new_data_df = pd.DataFrame([received_data])

    # Add current date and time to the DataFrame
    current_datetime = datetime.now()
    new_data_df['Date'] = current_datetime.date()
    new_data_df['Time'] = current_datetime.time()

    # Concatenate the new data with the existing real-time data
    real_time_data = pd.concat([real_time_data, new_data_df], ignore_index=True)

    if not real_time_data.empty:  # Ensure there's data before processing
        # Update Isolation Forest model with new data and detect anomalies
        for column in ['Speed', 'Temperature', 'Voltage', 'Current']:
            try:
                # Fit model if there's enough data
                if len(real_time_data) > 1:  # Assuming we need at least 2 data points to fit the model
                    models[column].fit(real_time_data[[column]])
                    anomaly_scores = models[column].decision_function(real_time_data[[column]])

                    # Check for anomalies and update anomaly_scores_history
                    anomaly_indices = np.where(anomaly_scores < 0)[0]
                    if anomaly_indices.size > 0:
                        anomaly_info = pd.DataFrame({
                            'Date': real_time_data.iloc[anomaly_indices]['Date'].tolist(),
                            'Time': real_time_data.iloc[anomaly_indices]['Time'].tolist(),
                            'Data Point Value': real_time_data.iloc[anomaly_indices][column].tolist(),
                            'Anomaly Scores': anomaly_scores[anomaly_indices].tolist()
                        })
                        print(f"Anomalies detected in {column}:\n{anomaly_info}")
                        if anomaly_scores_history[column].empty:
                            anomaly_scores_history[column] = anomaly_info
                        else:
                            anomaly_scores_history[column] = pd.concat([anomaly_scores_history[column], anomaly_info],
                                                                       ignore_index=True)
            except Exception as e:
                print(f"Error updating model for {column}: {e}")

        # Plotting anomaly detection results
        for i, column in enumerate(['Speed', 'Temperature', 'Voltage', 'Current']):
            axes[i].clear()
            axes[i].plot(real_time_data['Time'], real_time_data[column], label='Data', marker='o', linestyle='-',
                         color='blue')
            if column in anomaly_scores_history:
                anomaly_info = anomaly_scores_history[column]
                if not anomaly_info.empty:
                    axes[i].scatter(anomaly_info['Time'], anomaly_info['Data Point Value'],
                                    color='red', label='Anomaly')
            axes[i].set_title(f'Anomaly Detection - {column}')
            axes[i].legend()

        plt.pause(0.01)

    # Check if it's time to save the Excel file
    if datetime.now() >= next_save_time:
        current_date = datetime.now().strftime("%Y-%m-%d")
        excel_filename = f"anomaly_detection_results_{current_date}_{file_count}.xlsx"
        with pd.ExcelWriter(excel_filename) as writer:
            for column, anomaly_info in anomaly_scores_history.items():
                anomaly_info.to_excel(writer, sheet_name=f'{column}', index=False)
        print(f"Saved Excel file: {excel_filename}")
        file_count += 1  # Prepare for the next file
        next_save_time += save_interval  # Set the time for the next save operation

plt.ioff()
plt.show()
