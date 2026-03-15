import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("dataset_regresion_logistica.csv")

# Define predictor variables (features)
X = df[[
    "edad",              # age
    "ingreso_mensual",   # monthly income
    "visitas_web_mes",   # website visits per month
    "tiempo_sitio_min",  # time spent on site (minutes)
    "compras_previas",   # number of previous purchases
    "descuento_usado"    # discount used (likely 0/1)
]]

# Define target variable
y = df["target"]         # binary: 1 = purchased, 0 = did not purchase

# Train logistic regression model
model = LogisticRegression(max_iter=1000)  # increased max_iter just in case
model.fit(X, y)

# ────────────────────────────────────────────────
# Partial dependence visualization:
# We show the effect of "visitas_web_mes" while holding
# all other variables fixed at their average values
# ────────────────────────────────────────────────

# Extract the variable we want to visualize
feature_to_plot = "visitas_web_mes"
x_values = df[feature_to_plot].values.reshape(-1, 1)

# Create a smooth range of values for plotting the curve
x_range = np.linspace(x_values.min(), x_values.max(), 300).reshape(-1, 1)

# Build a temporary dataset where only the chosen feature varies
X_temp = pd.DataFrame({
    "edad":              [df["edad"].mean()]             * len(x_range),
    "ingreso_mensual":   [df["ingreso_mensual"].mean()]  * len(x_range),
    "visitas_web_mes":   x_range.flatten(),
    "tiempo_sitio_min":  [df["tiempo_sitio_min"].mean()] * len(x_range),
    "compras_previas":   [df["compras_previas"].mean()]  * len(x_range),
    "descuento_usado":   [df["descuento_usado"].mean()]  * len(x_range)
})

# Get predicted probabilities of class 1 (purchase)
probabilities = model.predict_proba(X_temp)[:, 1]

# ────────────────────────────────────────────────
# Plotting – improved version
# ────────────────────────────────────────────────

plt.figure(figsize=(10, 6))

# Scatter of real data points with color by class + jitter
# Jitter helps separate overlapping 0/1 points
y_jitter = y + np.random.uniform(-0.07, 0.07, size=len(y))

plt.scatter(
    df[feature_to_plot],
    y_jitter,
    c=y,                   # color by actual class (0 or 1)
    cmap="coolwarm",       # red = purchase, blue = no purchase
    alpha=0.5,
    s=35,                  # point size
    edgecolor="black",
    linewidth=0.4,
    label="Actual outcomes"
)

# Logistic regression curve (smooth probability)
plt.plot(
    x_range,
    probabilities,
    color="darkred",
    linewidth=2.8,
    label="Predicted probability"
)

# Optional: add a decision boundary line at 0.5
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label="Decision threshold (0.5)")

plt.xlabel("Website visits per month", fontsize=12)
plt.ylabel("Probability of purchase / Actual outcome", fontsize=12)
plt.title("Logistic Regression – Effect of Website Visits per Month\n(other variables held at mean values)", fontsize=14)

plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.25, linestyle="--")
plt.ylim(-0.15, 1.15)   # give some breathing room for jitter

plt.tight_layout()
plt.show()