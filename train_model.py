import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_data.csv")

# Input features
X = data[["area", "bedrooms", "bathrooms"]]

# Target variable
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save trained model
joblib.dump(model, "house_model.pkl")

print("Model trained successfully!")
