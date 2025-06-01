import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv('boston.csv')

# Debug: print actual column names
print("📋 Column names:")
print(data.columns)

# Use correct column names (lowercase)
X = data[['rm', 'lstat', 'crim']]
y = data['medv']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved to model.pkl")
