# HireSense: AI-Powered Candidate Selection

## Overview
**HireSense** is a machine learning project designed to streamline candidate selection processes for HR teams. By leveraging **multivariate linear regression**, the model evaluates candidates based on their **experience**, **test scores (out of 10)**, and **interview scores (out of 10)** to predict their suitability for a given role.

## Features
- Analyze candidate experience, test scores, and interview scores.
- Predict the best-fit candidates for a job role.
- Enhance HR decision-making with data-driven insights.
- Built using Python and essential ML libraries.

## Dataset
The dataset should include the following columns:
- `experience` (e.g., "two", "five", etc., or numerical values)
- `test_score(out of 10)`
- `interview_score(out of 10)`
- `salary($)` (target variable for salary prediction)

Example dataset:

| experience | test_score(out of 10) | interview_score(out of 10) | salary($) |
|------------|------------------------|----------------------------|-----------|
| NaN        | 8.0                    | 9                          | 50000     |
| NaN        | 8.0                    | 6                          | 45000     |
| five       | 6.0                    | 7                          | 60000     |
| two        | 10.0                   | 10                         | 65000     |
| seven      | 9.0                    | 6                          | 70000     |

## Prerequisites
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/HireSense.git
   cd HireSense
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Snippets

### 1. Data Preprocessing
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('dataset.csv')

# Handle missing and non-numeric values
data['experience'] = data['experience'].map({'two': 2, 'three': 3, 'five': 5, 'seven': 7, 'ten': 10, None: 0})
data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean(), inplace=True)

# Split features and target
X = data[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y = data['salary($)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Training the Model
```python
from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
```

### 3. Model Evaluation
```python
from sklearn.metrics import mean_squared_error, r2_score

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
```

### 4. Visualizing Results
```python
import matplotlib.pyplot as plt

# Plot predicted vs actual
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.show()
```

## Usage
1. Place your dataset in the `dataset.csv` file.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. View predictions and performance metrics in the terminal.

## Future Enhancements
- Add support for additional candidate evaluation metrics.
- Integrate with an HR management system for real-time use.
- Deploy the model as a web application.

## Contributing
Feel free to fork this repository, create new features, and submit pull requests!
