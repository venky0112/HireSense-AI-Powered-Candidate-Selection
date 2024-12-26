### README for HireSense: AI-Powered Candidate Selection

---

## Overview
**HireSense** is a machine learning project designed to streamline candidate selection processes for HR teams. By leveraging **multivariate linear regression**, the model evaluates candidates based on their **experience**, **written test scores**, and **personal interview scores** to predict their suitability for a given role.

---

## Features
- Analyze candidate experience, written test scores, and personal interview scores.
- Predict the best-fit candidates for a job role.
- Enhance HR decision-making with data-driven insights.
- Built using Python and essential ML libraries.

---

## Dataset
The dataset should include the following columns:
- `Experience` (in years)
- `Written Test Score` (out of 100)
- `Personal Interview Score` (out of 100)
- `Hired` (1 for hired, 0 for not hired - target variable)

Example dataset:

| Experience (years) | Written Test Score | Personal Interview Score | Hired |
|---------------------|--------------------|--------------------------|-------|
| 5                   | 88                 | 85                       | 1     |
| 2                   | 72                 | 78                       | 0     |
| 3                   | 75                 | 80                       | 1     |

---

## Prerequisites
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

---

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

---

## Code Snippets

### 1. Data Preprocessing
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('dataset.csv')

# Split features and target
X = data[['Experience', 'Written Test Score', 'Personal Interview Score']]
y = data['Hired']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Training the Model
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
```

### 3. Model Evaluation
```python
from sklearn.metrics import confusion_matrix, classification_report

# Evaluate performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### 4. Visualizing Results
```python
import matplotlib.pyplot as plt

# Plot predicted vs actual
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```

---

## Usage
1. Place your dataset in the `dataset.csv` file.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. View predictions and performance metrics in the terminal.

---

## Future Enhancements
- Add support for additional candidate evaluation metrics.
- Integrate with an HR management system for real-time use.
- Deploy the model as a web application.

---

## Contributing
Feel free to fork this repository, create new features, and submit pull requests!
