import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "A": "as-rolled", "B": "normalized", "C": "annealed", "D": "sand casting", "E": "wrought", "F": "cast",
    "G": "as extruded","H": "tempered", "I": "hardened", "J": "heat treated", "K": "case-hardened", "L": "cold working",
    "M": "quenched and tempered", "N": "quenching and cooling", "O": "improved", "NONE": "no pre-processing"
}

for key, value in conditions.items():
    print(f"{key}: {value}")

condition = input("Condition: ").strip().upper()

# Validate and filter data based on the chosen condition
if condition == "NONE":
    data = data
elif condition in conditions:
    chosen_condition = conditions[condition]
    if chosen_condition == "tempered":
        data = data[data['Material'].str.contains("temper", case=False)] #"tempered" chooses data with "temper"
    elif chosen_condition == "hardened":
        data = data[data['Material'].str.contains("hard", case=False)] #hardened chooses data with "hard"
    else:
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
    user_data = {
        "E": float(input("Enter Young's Modulus in MPa (E): ")),
        "G": float(input("Enter Shear Modulus in MPa (G): ")),
        "mu": float(input("Enter Poisson's Ratio (μ): ")), 
        "Ro": float(input("Enter Density in Kg/m3 (ρ): ")),
        "Sy": float(input("Enter Yield Strength in MPa (σₛ): ")) 
    }
except ValueError:
    print("Invalid Input!")
    exit()

# Convert user input to DataFrame (Fixes the warning)
user_input_df = pd.DataFrame([user_data])

# Standardize the input using the saved scaler
user_input_scaled = scaler.transform(user_input_df)

print("Model Evaluation:")
print(f"Model Score: {mscore}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared (R2): {r2}")

# Predict Su
predicted_Su = model.predict(user_input_scaled)
print(f"Predicted Ultimate Tensile Strength (Su): {predicted_Su[0]:.2f}")

E_input = user_data["E"]
Sy_input = user_data["Sy"]

# Calculate strain at yield point
ε_y = Sy_input / E_input  # Engineering strain at yield

σ_LY = Sy_input * 0.85 # assumed (approx)
ε_LY = ε_y + 0.002  # Approximated

# True stress and true strain at yield
σ_y = Sy_input * (1 + ε_y)  # True stress at yield
ϵ_y = np.log(1 + ε_y)       # True strain at yield

# Define strain offsets based on material condition and Young's modulus
strain_offsets = {
    "A": [(190000, 225000, 0.05), (150000, 189999, 0.07), (100000, 149999, 0.09), (80000, 99999, 0.12), (40000, 79999, 0.18), (0, 39999, 0.25)],
    "B": [(190000, 225000, 0.06), (150000, 189999, 0.08), (100000, 149999, 0.10), (80000, 99999, 0.13), (40000, 79999, 0.19), (0, 39999, 0.26)],
    "C": [(190000, 225000, 0.07), (150000, 189999, 0.09), (100000, 149999, 0.11), (80000, 99999, 0.14), (40000, 79999, 0.20), (0, 39999, 0.27)],
    "D": [(190000, 225000, 0.04), (150000, 189999, 0.06), (100000, 149999, 0.08), (80000, 99999, 0.11), (40000, 79999, 0.17), (0, 39999, 0.24)],
    "E": [(190000, 225000, 0.08), (150000, 189999, 0.10), (100000, 149999, 0.12), (80000, 99999, 0.15), (40000, 79999, 0.21), (0, 39999, 0.28)],
    "F": [(190000, 225000, 0.05), (150000, 189999, 0.07), (100000, 149999, 0.09), (80000, 99999, 0.12), (40000, 79999, 0.18), (0, 39999, 0.25)],
    "G": [(190000, 225000, 0.06), (150000, 189999, 0.08), (100000, 149999, 0.10), (80000, 99999, 0.13), (40000, 79999, 0.19), (0, 39999, 0.26)],
    "H": [(190000, 225000, 0.07), (150000, 189999, 0.09), (100000, 149999, 0.11), (80000, 99999, 0.14), (40000, 79999, 0.20), (0, 39999, 0.27)],
    "I": [(190000, 225000, 0.08), (150000, 189999, 0.10), (100000, 149999, 0.12), (80000, 99999, 0.15), (40000, 79999, 0.21), (0, 39999, 0.28)],
    "J": [(190000, 225000, 0.09), (150000, 189999, 0.11), (100000, 149999, 0.13), (80000, 99999, 0.16), (40000, 79999, 0.22), (0, 39999, 0.29)],
    "K": [(190000, 225000, 0.05), (150000, 189999, 0.07), (100000, 149999, 0.09), (80000, 99999, 0.12), (40000, 79999, 0.18), (0, 39999, 0.25)],
    "L": [(190000, 225000, 0.06), (150000, 189999, 0.08), (100000, 149999, 0.10), (80000, 99999, 0.13), (40000, 79999, 0.19), (0, 39999, 0.26)],
    "M": [(190000, 225000, 0.07), (150000, 189999, 0.09), (100000, 149999, 0.11), (80000, 99999, 0.14), (40000, 79999, 0.20), (0, 39999, 0.27)],
    "N": [(190000, 225000, 0.08), (150000, 189999, 0.10), (100000, 149999, 0.12), (80000, 99999, 0.15), (40000, 79999, 0.21), (0, 39999, 0.28)],
    "O": [(190000, 225000, 0.09), (150000, 189999, 0.11), (100000, 149999, 0.13), (80000, 99999, 0.16), (40000, 79999, 0.22), (0, 39999, 0.29)],
    "NONE": [(190000, 225000, 0.05), (150000, 189999, 0.07), (100000, 149999, 0.09), (80000, 99999, 0.12), (40000, 79999, 0.18), (0, 39999, 0.25)]
}

def get_strain_offset(E_input, condition):
    for (E_min, E_max, offset) in strain_offsets[condition]:
        if E_min <= E_input <= E_max:
            return offset
    return 0.1  # Default fallback

# Get strain offset based on Young's modulus input and material condition
strain_offset = get_strain_offset(E_input, condition)

# Estimate strain at UTS using dynamically determined offset
ε_u = (predicted_Su[0] / E_input) + strain_offset
σ_u = predicted_Su[0] * (1 + ε_u)  # True stress at UTS
ϵ_u = np.log(1 + ε_u)              # True strain at UTS

# Calculate strain-hardening exponent (n) and strength coefficient (K)
n = np.log(σ_u / σ_y) / np.log(ϵ_u / ϵ_y)
K = σ_y / (ϵ_y ** n)

fracture_stress_ratio = 0.85 # assumed (approx)
σ_fracture = predicted_Su[0] * fracture_stress_ratio

# alloting (approx) fracture strain scaling factors based on pre-processing conditions

fracture_strain_factors = {
    "A": 1.3, "B": 1.5, "C": 2.0, "D": 1.1, "E": 1.8,
    "F": 1.2, "G": 1.6, "H": 1.4, "I": 1.5,
    "J": 1.7, "K": 1.0, "L": 1.1, "M": 1.4, "N": 1.3, "O": 1.6,
    "NONE": 1.5
}

# Calculate fracture strain
fracture_strain_factor = fracture_strain_factors[condition]
ε_fracture = ε_u * fracture_strain_factor

ε_transition = np.linspace(ε_y, ε_LY, 50)
σ_transition = np.linspace(Sy_input, σ_LY, 50)

# Generate strain values for the plastic region up to fracture
ϵ_plastic = np.linspace(ε_LY, ϵ_u, 200)  # True strain in the plastic region

# Calculate stress in the plastic region using Hollomon's equation
σ_plastic = K * (ϵ_plastic ** n)
σ_plastic = np.minimum(σ_plastic, predicted_Su[0])

ϵ_fracture_range = np.linspace(ϵ_u, np.log(1 + ε_fracture), 50)
σ_fracture_range = np.linspace(predicted_Su[0], σ_fracture, 50)  # Gradually decreasing stress

# Convert true strain back to engineering strain for plotting
ε_plastic = np.exp(ϵ_plastic) - 1
ε_fracture_curve = np.exp(ϵ_fracture_range) - 1

# Combine elastic and plastic regions
ε_total = np.concatenate((np.linspace(0, ε_y, 100), ε_transition, ε_plastic, ε_fracture_curve))
σ_total = np.concatenate((E_input * np.linspace(0, ε_y, 100), σ_transition, σ_plastic, σ_fracture_range))

# Plot the stress-strain curve
plt.figure(figsize=(10, 6))
plt.plot(ε_total, σ_total, color='k', label='Stress-Strain Curve')
plt.xlabel('Strain (ε)')
plt.ylabel('Stress (σ) [MPa]')
plt.title('Stress-Strain Curve')
plt.grid(True)
plt.axhline(y=Sy_input, color='r', linestyle='--', label=f'Yield Strength (σₛ = {Sy_input:.2f} MPa)')
plt.axvline(x=ε_y, color='b', linestyle='--', label=f'Yield Strain (ε_y = {ε_y:.4f})')
plt.axhline(y=σ_LY, color='orange', linestyle='--', label=f'Lower Yield Point (σ_LY = {σ_LY:.2f} MPa)')
plt.axvline(x=ε_LY, color='c', linestyle='--', label=f'Lower Yield Point (ε_fracture = {ε_LY:.4f})')
plt.axvline(x=ε_fracture, color='g', linestyle='--', label=f'Fracture Point (ε_fracture = {np.log(1 + ε_fracture):.4f})')
plt.axhline(y=predicted_Su, color='m', linestyle='--', label=f'Predicted UTS (σᵤ = {predicted_Su[0]:.2f} MPa)')
plt.legend()
plt.show()
