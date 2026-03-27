import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ds = pd.read_excel("energy_efficiency.xlsx")

print(ds.to_string())

ds.head()

ds.columns = [
        "Relative_Compactness", #X1
        "Surface_area",         #X2
        "Wall_Area",            #X3
        "Roof_Area",            #X4
        "Overall_Height",       #X5
        "Orientation",          #X6
        "Glazing_Area",         #X7
        "Glazing_Area_Distribution", #X8        
        "Heating_Load",         #Y1
        "Cooling_Load"          #Y2
    ]

ds.info()
print("null values: \n", ds.isnull().sum())

duplicates = ds.duplicated().sum()
print("duplicate rows: ", duplicates)

ds.describe()

ds = ds.drop("Cooling_Load", axis=1)
ds.head()

x = ds.drop("Heating_Load", axis=1)
y = ds["Heating_Load"]

print("Feature Shape: ", x.shape)
print("Target Shape: ", y.shape)

print(ds.head())
print(ds.shape)


ds.columns

ymean = y.mean()
ymedian = y.median()
ymin = y.min()
ymax = y.max()

plt.figure(figsize=(10,6))
plt.hist(y, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(ymean, linestyle='--', linewidth=2, label=f"Mean = {ymean:.2f}")
plt.axvline(ymedian, linestyle='-', linewidth=2, label=f"Median = {ymedian:.2f}")
plt.title("Distribution of Heating Load in Buildings", fontsize=14)
plt.xlabel("Heating Load (Energy Units)", fontsize=12)
plt.ylabel("Number of Buildings", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.text(ymin, 30, f"Min = {ymin}", rotation=90)
plt.text(ymax, 30, f"Max = {ymax}", rotation=90)
plt.legend()
plt.show()


rel_cmpt=ds["Relative_Compactness"]
m,b=np.polyfit(rel_cmpt,y,1)
sorted_idx=np.argsort(rel_cmpt)
x_sorted=rel_cmpt.iloc[sorted_idx]
y_pred=m*x_sorted+b
plt.figure(figsize=(10,6))
plt.scatter(rel_cmpt,y,marker='o',alpha=0.7,label='Building Samples')
plt.plot(x_sorted,y_pred,linewidth=2,label="Trend Line")
plt.axvline(rel_cmpt.mean(),linestyle='--',label=f"Mean Compactness = {rel_cmpt.mean():.2f}")
plt.axhline(y.mean(),linestyle=':',label=f"Mean Heating Load = {y.mean():.2f}")
plt.title("Relative Compactness vs Heating Load",fontsize=14)
plt.xlabel("Relative Compactness",fontsize=12)
plt.ylabel("Heating Load (Energy Units)",fontsize=12)
plt.grid(True,linestyle='--',alpha=0.6)
plt.legend()
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=20) 

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Linear Regression: ")
print("MAE: ", mae)
print("MSE: ", mse)

print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print(y_test.values[:5])
print(y_pred[:5])

results = pd.DataFrame({"Actual Heating Load": y_test.values, "Predicted Heating Load": y_pred})
results["Absolute Error"] = abs(results["Actual Heating Load"] - results["Predicted Heating Load"])


results.head()
results.sort_values(by="Absolute Error", ascending=False).head()

results.to_excel("E:/Semesters/4th Semester/AI/Labs/AI-Driven Buildings Energy Efficiency Analysis & Prediction/report/heating_load_prediction.xlsx", index=False)

coef_df = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_
})

print(coef_df)

coef_df.to_excel("E:/Semesters/4th Semester/AI/Labs/AI-Driven Buildings Energy Efficiency Analysis & Prediction/report/learned_coefficients.xlsx", index=False)



