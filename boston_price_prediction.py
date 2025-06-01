import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('boston.csv')
print("✅ Data loaded successfully")

# Step 2: Explore the data
print("\n🔍 First 5 rows of the dataset:")
print(data.head())

print("\n🧼 Checking for null values:")
print(data.isnull().sum())

# Step 3: Visualize the data
sns.pairplot(data[['rm', 'lstat', 'crim', 'medv']])
plt.show()

# Step 4: Prepare data for training
X = data.drop('medv', axis=1)
y = data['medv']

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✂️ Data split into training and testing sets.")

# Step 6: Train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n📈 Model trained successfully")

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Mean Squared Error (MSE): {mse:.2f}")
print(f"📊 R-squared (R2 Score): {r2:.2f}")

# Step 9: Visualize predictions
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
# After loading the CSV
data = pd.read_csv('boston.csv')
print("✅ Data loaded successfully\n")

# Print all column names
print("📋 Column names in dataset:")
print(data.columns)
# Load data
data = pd.read_csv('boston.csv')
print("✅ Dataset loaded successfully!")

# PRINT COLUMN NAMES TO DEBUG
print("\n📋 Column names in dataset:")
print(data.columns)
