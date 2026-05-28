# Machine Learning Project

A practical machine learning project built to predict productivity based on daily activities. It combines Linear Regression and Random Forest models with a clean web interface for real-time predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Models](#models)
- [Contact](#contact)

## Overview

This project demonstrates a complete ML workflow from data preprocessing to deployment. We built models to predict student productivity using various daily metrics, and created a web interface so users can get predictions instantly.

## Features

- **Multiple Models**: Linear Regression and Random Forest implementations
- **Data Visualization**: Correlation analysis, feature distributions, and importance charts
- **Web Interface**: Simple Flask app for making predictions
- **Model Comparison**: Side-by-side evaluation of different models
- **Feature Analysis**: Understanding which factors matter most

## Screenshots

Here's what the project looks like in action:

<table>
<tr>
<td width="50%">

### AI Productivity Dashboard

![Dashboard](backend/images/image1.png)

*Enter your daily metrics to get predictions*

</td>
<td width="50%">

### Prediction Results  
![Prediction](backend/images/image_2.png)

*Real-time productivity score and analysis*

</td>
</tr>
<tr>
<td width="50%">

### Model Comparison
![Model Comparison](backend/images/image_3.png)

*Comparing different models performance*

</td>
<td width="50%">

### Feature Analysis
![Feature Analysis](backend/images/image_4.png)
*Understanding what drives productivity*

</td>
</tr>
</table>

## Getting Started

### Prerequisites

You'll need Python 3.8+ and pip installed on your system.

### Installation

1. Clone the repo:
```bash
git clone https://github.com/SUNILNAGRKOTI/Machine_Learning_Project.git
cd Machine_Learning_Project
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Machine_Learning_Project/
├── backend/
│   ├── app.py                      # Flask backend server
│   ├── main.py                     # Model training script
│   ├── random_forest_model.pkl     # Trained Random Forest model
│   ├── linear_regression_model.pkl # Trained Linear Regression model
│   ├── model_metadata.json         # Model performance metrics
│   ├── student_sleep_patterns.csv  # Training dataset
│   └── images/                     # Screenshot folder
│
├── Front-End/
│   ├── index.html                  # Web interface
│   ├── script.js                   # Frontend logic
│   └── style.css                   # Styling
│
└── README.md                       # This file
```

## How to Use

### Run the Web Application

1. Start the Flask backend:
```bash
cd backend
python app.py
```

2. Open your browser and go to: `http://localhost:5000`

3. Enter your daily metrics (sleep, study hours, screen time, exercise, caffeine) and click "Predict" to get your productivity score

### Train the Models

If you want to retrain the models with new data:
```bash
python main.py
```

This will preprocess the data, train both models, and save them along with visualizations.

## Models

**Linear Regression**: A baseline model that learns linear relationships between features and productivity.

**Random Forest**: An ensemble model that captures non-linear patterns and typically performs better than linear regression.

Both models are trained on student daily activity data and generate feature importance scores to show which activities matter most for productivity.

## Contact

**Sunil Nagarkoti**
- GitHub: [@SUNILNAGRKOTI](https://github.com/SUNILNAGRKOTI)
- Project: [Machine_Learning_Project](https://github.com/SUNILNAGRKOTI/Machine_Learning_Project)
