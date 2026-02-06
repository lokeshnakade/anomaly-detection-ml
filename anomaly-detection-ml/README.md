# Real-Time Anomaly Detection System using Machine Learning

## Overview
This project implements a real-time anomaly detection system designed for industrial
sensor monitoring. The system receives live data from an external source and detects
abnormal behavior using machine learning techniques.

The primary focus is on building a scalable and interpretable anomaly detection
pipeline suitable for real-world deployment.

## System Design
- Live sensor data is streamed to the system via socket communication
- Data is processed in real time on a centralized ML server
- Isolation Forest models are used to detect anomalies per sensor parameter
- Anomalies are visualized live and logged periodically for analysis

Client-side data generation code is intentionally excluded to keep the repository
focused on the machine learning and analysis components.

## Features
- Real-time data ingestion
- Machine learning–based anomaly detection
- Live visualization of sensor behavior
- Automated anomaly reporting
- Offline performance evaluation

## Model Evaluation
An offline evaluation script is included to measure accuracy, precision, recall, and
F1-score using synthetically generated anomalies. This approach is commonly used in
anomaly detection problems where labeled anomalies are scarce.

The evaluation validates model behavior rather than claiming absolute production
accuracy.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Socket Programming

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the real-time anomaly detection server:
```bash
python src/realtime_anomaly_server.py
```

Run performance evaluation:
```bash
python src/evaluation.py
```

## Use Case
Industrial IoT monitoring and predictive maintenance through real-time anomaly
detection.
