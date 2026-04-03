import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline


# ===============================
# LOAD & PREPROCESS DATA
# ===============================
ds = pd.read_excel("energy_efficiency.xlsx")

ds.columns = [
    "Relative_Compactness",
    "Surface_area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
    "Heating_Load",
    "Cooling_Load"
]

print("\nDataset Info:")
ds.info()

print("\nNull Values:\n", ds.isnull().sum())
print("Duplicate Rows:", ds.duplicated().sum())

# Droped Cooling Load
ds = ds.drop("Cooling_Load", axis=1)


X = ds.drop("Heating_Load", axis=1)
y = ds["Heating_Load"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# TARGET DISTRIBUTION
ymean, ymedian = y.mean(), y.median()
ymin, ymax = y.min(), y.max()

plt.figure(figsize=(10,6))
plt.hist(y, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(ymean, linestyle='--', label=f"Mean = {ymean:.2f}")
plt.axvline(ymedian, linestyle='-', label=f"Median = {ymedian:.2f}")
plt.title("Heating Load Distribution")
plt.xlabel("Heating Load")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()



# FEATURE RELATION (EXAMPLE)
rel_cmpt = ds["Relative_Compactness"]

m, b = np.polyfit(rel_cmpt, y, 1)
sorted_idx = np.argsort(rel_cmpt)

plt.figure(figsize=(10,6))
plt.scatter(rel_cmpt, y, alpha=0.7)
plt.plot(rel_cmpt.iloc[sorted_idx], m*rel_cmpt.iloc[sorted_idx]+b)

plt.axvline(rel_cmpt.mean(), linestyle='--', label="Mean Compactness")
plt.axhline(y.mean(), linestyle=':', label="Mean Heating Load")

plt.title("Relative Compactness vs Heating Load")
plt.xlabel("Relative Compactness")
plt.ylabel("Heating Load")
plt.grid(True)
plt.legend()
plt.show()


# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)



# MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# EVALUATION FUNCTION
def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nLinear Regression Performance:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

evaluate(y_test, y_pred)


# ACTUAL VS PREDICTED PLOT
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')

min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))

plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Linear Regression)")
plt.axis('equal')
plt.grid(True)
plt.show()


# RESIDUAL PLOT
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# RESULTS TABLE
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results["Absolute Error"] = abs(results["Actual"] - results["Predicted"])

print("\nTop Errors:")
print(results.sort_values(by="Absolute Error", ascending=False).head())


# COEFFICIENT ANALYSIS
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance (Linear Coefficients):")
print(coef_df)

polynomial = PolynomialFeatures(degree=2)
x_train_poly = polynomial.fit_transform(X_train)
x_test_poly = polynomial.transform(X_test)

print("Original Shape: ", X_train.shape)
print("After Polynomial: ", x_test_poly.shape)
print(X_test.iloc[0], "\n", x_test_poly[0])

model_poly = LinearRegression()
model_poly.fit(x_train_poly, y_train)
y_pred_poly = model_poly.predict(x_test_poly)

print("R2 score of Polynomial: ", r2_score(y_test, y_pred_poly))
print("Training score: ", model_poly.score(x_train_poly, y_train))
print("Testing score: ", model_poly.score(x_test_poly, y_test))


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_poly, alpha=0.7, edgecolors='k')
min_val=min(min(y_test), min(y_pred_poly))
max_val=max(max(y_test), max(y_pred_poly))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Polynomial)")
plt.axis('equal')
plt.grid(True)
plt.show()

pipeline=make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())

train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y, cv=5, scoring="r2", train_sizes=np.linspace(0.3,1.0, 5))
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation_Score")
plt.xlabel("Training Score")
plt.ylabel("R2 Score")
plt.title("Learning Curve (Polynomial Regression)")
plt.legend()
plt.grid(True)
plt.show()

