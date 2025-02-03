import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the data
data = pd.read_csv("material.csv")

# Prompt user for material condition
print("Choose Material Condition:")
conditions = {
    "A": "as-rolled",
    "B": "normalized",
    "C": "annealed",
    "D": "sand casting",
    "E": "wrought",
    "F": "cast",
    "G": "as extruded",
    "H": "tempered at 400 F",
    "I": "tempered at 600 F",
    "J": "tempered at 800 F",
    "K": "1/4-hard",
    "L": "1/2-hard",
    "M": "3/4-hard",
    "N": "full-hard",
    "O": "heat treated",
    "P": "case-hardened",
    "Q": "face hardened",
    "R": "improved",
    "S": "cold working",
    "T": "high tempering",
    "U": "quenched and tempered",
    "V": "quenching and cooling"
}
for key, value in conditions.items():
    print(f"{key}: {value}")

condition = input("Condition: ").strip().upper()

# Validate and filter data based on the chosen condition
if condition in conditions:
    chosen_condition = conditions[condition]
    data = data[data['Material'].str.contains(chosen_condition, case=False)]
else:
    raise ValueError("please select a valid option.")

# Define features and target
features = data[['E', 'G', 'mu', 'Ro', 'Sy']]
target = data['Su']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Evaluate the model
mscore = model.score(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the model and scaler
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Prompt user for new inputs to predict Su
try:
    E_input = float(input("Enter Young's Modulus (E): "))
    G_input = float(input("Enter Shear Modulus (G): "))
    μ_input = float(input("Enter Poisson's Ratio (μ): ")) 
    ρ_input = float(input("Enter Density (ρ): "))
    Sy_input = float(input("Enter Yield Strength (σₛ): ")) 
except ValueError:
    print("Invalid Input!")
    exit()

# Prepare input for prediction
user_input = np.array([[E_input, G_input, μ_input, ρ_input, Sy_input]])
user_input_scaled = scaler.transform(user_input)

print("Model Evaluation:")
print(f"Model Score: {mscore}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared (R2): {r2}")

# Predict Su
predicted_Su = model.predict(user_input_scaled)
print(f"Predicted Ultimate Tensile Strength (Su): {predicted_Su[0]:.2f}")
