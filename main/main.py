import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree


# load data
ds = pd.read_excel("energy_efficiency.xlsx")

ds.columns = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
    "heating_load",
    "cooling_load"
]

print("\nDataset Info:")
ds.info()
print("\nnull values:\n", ds.isnull().sum())
print("duplicate rows:", ds.duplicated().sum())
print("\ndescribe:")
print(ds.describe())


# drop target/unused
ds = ds.drop("cooling_load", axis=1)


# heating load distribution
y = ds["heating_load"]
ymean   = y.mean()
ymedian = y.median()
plt.figure(figsize=(12, 6))
bins = np.histogram_bin_edges(y, bins='fd')
plt.hist(y, bins=bins, edgecolor='black', alpha=0.75)
plt.axvline(ymean,   color='red',   linestyle='--', linewidth=2, label=f"Mean = {ymean:.2f}")
plt.axvline(ymedian, color='green', linestyle='-',  linewidth=2, label=f"Median = {ymedian:.2f}")
plt.xticks(np.arange(round(min(y)), round(max(y))+1, 2))
plt.title("Heating Load Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Heating Load", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# outlier detection
plt.figure(figsize=(12, 6))
ds.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot — Outlier Detection")
plt.tight_layout()
plt.show()

# correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(ds.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# compactness scatter
rel_cmpt   = ds["relative_compactness"]
y_all      = ds["heating_load"]
m, b       = np.polyfit(rel_cmpt, y_all, 1)
sorted_idx = np.argsort(rel_cmpt)
plt.figure(figsize=(10, 6))
plt.scatter(rel_cmpt, y_all, alpha=0.7)
plt.plot(rel_cmpt.iloc[sorted_idx], m * rel_cmpt.iloc[sorted_idx] + b)
plt.axvline(rel_cmpt.mean(), linestyle='--', label="Mean Compactness")
plt.axhline(y_all.mean(),    linestyle=':',  label="Mean Heating Load")
plt.title("Relative Compactness vs Heating Load")
plt.xlabel("Relative Compactness")
plt.ylabel("Heating Load")
plt.grid(True)
plt.legend()
plt.show()


# feature selection
# dropped orientation  — 0.00 correlation with heating load (useless)
# dropped surface_area — 0.99 correlated with relative_compactness (redundant)
# dropped roof_area    — 0.97 correlated with overall_height (redundant)
# dropped overall_height, glazing_area_distribution — near-zero feature importance
ds_clean = ds.drop(["orientation", "surface_area", "roof_area",
                     "overall_height", "glazing_area_distribution"], axis=1)

print("\nfeatures after selection:", list(ds_clean.columns))


# feature engineering
ds_eng = ds_clean.copy()
ds_eng["compactness_per_height"] = ds["relative_compactness"] / ds["overall_height"]

print("\nengineered dataset columns:", list(ds_eng.columns))
print(ds_eng.head())


# evaluation function
def evaluate(y_test, y_pred, model_name="Model"):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# train/test split
X = ds_eng.drop("heating_load", axis=1)
y = ds_eng["heating_load"]
print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


# linear regression
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
r1 = evaluate(y_test, y_pred_linear, "Linear Regression")

# linear actual vs predicted
plt.figure(figsize=(6, 6))
min_val = min(min(y_test), min(y_pred_linear))
max_val = max(max(y_test), max(y_pred_linear))
plt.scatter(y_test, y_pred_linear, alpha=0.7, edgecolors='k')
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Linear Regression)")
plt.axis('equal')
plt.grid(True)
plt.show()

# linear residuals
residuals_linear = y_test - y_pred_linear
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_linear, residuals_linear, alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot (Linear Regression)")
plt.grid(True)
plt.show()


# depth selection
def train_tree():
    best_r2, best_depth = 0, 0
    for i in range(1, 10):
        model = DecisionTreeRegressor(max_depth=i, random_state=42)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"depth={i}: CV R² = {cv.mean():.4f}")
        if cv.mean() > best_r2:
            best_r2    = cv.mean()
            best_depth = i
    return best_depth


# decision tree
i = train_tree()
print(f"\nBest depth: {i}")
model_tree = DecisionTreeRegressor(max_depth=i, random_state=42)
model_tree.fit(X_train, y_train)
print("Train R²:", model_tree.score(X_train, y_train))
y_pred_tree = model_tree.predict(X_test)
r2_tree = evaluate(y_test, y_pred_tree, "Decision Tree")

# tree actual vs predicted
plt.figure(figsize=(6, 6))
min_val = min(min(y_test), min(y_pred_tree))
max_val = max(max(y_test), max(y_pred_tree))
plt.scatter(y_test, y_pred_tree, alpha=0.7, edgecolors='k')
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Decision Tree)")
plt.axis('equal')
plt.grid(True)
plt.show()

# tree residuals
residuals_tree = y_test - y_pred_tree
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_tree, residuals_tree, alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot (Decision Tree)")
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
plot_tree(
    model_tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Structure (Regression)")
plt.show()

# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_linear = cross_val_score(model_linear, X_train, y_train, cv=kf, scoring="r2")
cv_tree   = cross_val_score(model_tree,   X_train, y_train, cv=kf, scoring="r2")
print("\nLinear CV R²:", cv_linear.mean())
print("Tree   CV R²:", cv_tree.mean())


# feature importance
importances = pd.Series(model_tree.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8, 5), title='Feature Importance')
plt.tight_layout()
plt.show()


# before/after engineering
X_before = ds_clean.drop("heating_load", axis=1)
X_after  = ds_eng.drop("heating_load", axis=1)

print("\nBefore vs After Feature Engineering (Decision Tree, depth=8):")
for label, Xb in [("Before Engineering", X_before), ("After Engineering", X_after)]:
    Xtr, Xte, ytr, yte = train_test_split(Xb, y, test_size=0.2, random_state=20)
    m = DecisionTreeRegressor(max_depth=i, random_state=42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    print(f"  {label} → MAE: {mean_absolute_error(yte, pred):.4f} | "
          f"RMSE: {math.sqrt(mean_squared_error(yte, pred)):.4f} | "
          f"R²: {r2_score(yte, pred):.4f}")


# error distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals_linear, bins=30, edgecolor='black', alpha=0.75, color='steelblue')
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_title("Linear Regression — Error Distribution")
axes[0].set_xlabel("Residual (Actual - Predicted)")
axes[0].set_ylabel("Frequency")

axes[1].hist(residuals_tree, bins=30, edgecolor='black', alpha=0.75, color='darkorange')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title("Decision Tree — Error Distribution")
axes[1].set_xlabel("Residual (Actual - Predicted)")

plt.suptitle("Error Distribution Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nLinear Regression Residuals:")
print(f"  Mean : {residuals_linear.mean():.4f}  <- should be near 0")
print(f"  Std  : {residuals_linear.std():.4f}   <- spread of errors")
print(f"  Max  : {residuals_linear.abs().max():.4f} <- worst prediction")

print("\nDecision Tree Residuals:")
print(f"  Mean : {residuals_tree.mean():.4f}")
print(f"  Std  : {residuals_tree.std():.4f}")
print(f"  Max  : {residuals_tree.abs().max():.4f}")


# overfitting diagnosis
depths = range(1, 10)
train_scores, test_scores = [], []

for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_scores.append(m.score(X_train, y_train))
    test_scores.append(m.score(X_test,  y_test))

plt.figure(figsize=(9, 5))
plt.plot(depths, train_scores, 'o-', label='Train R²',  color='steelblue')
plt.plot(depths, test_scores,  's-', label='Test R²',   color='darkorange')
plt.axvline(i, color='green', linestyle='--', label=f'Selected depth={i}')
plt.title("Overfitting Diagnosis — Train vs Test R² by Depth")
plt.xlabel("max_depth")
plt.ylabel("R²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# model comparison
comparison = pd.DataFrame([r1, r2_tree]).sort_values("R2", ascending=False)
print("\n", comparison.to_string(index=False))