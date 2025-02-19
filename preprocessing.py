import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load HR dataset
file_path = "hr_data.xlsx"
df = pd.read_excel(file_path, sheet_name="Employees")

# Data Cleaning
df.dropna(inplace=True)  # Remove missing values
df["Salary"] = df["Salary"].astype(float)  # Ensure salary is numeric
df["Age"] = df["Age"].astype(int)

# Employee Attrition Analysis
attrition_rate = df["Attrition"].value_counts(normalize=True) * 100
print("Attrition Rate:")
print(attrition_rate)

# Average Salary by Department
avg_salary = df.groupby("Department")["Salary"].mean().reset_index()
print("\nAverage Salary by Department:")
print(avg_salary)

# Performance Rating Distribution
performance_dist = df["PerformanceRating"].value_counts()
print("\nPerformance Rating Distribution:")
print(performance_dist)

# Save Processed Data
df.to_csv("processed_hr_data.csv", index=False)
print("\nProcessed data saved to 'processed_hr_data.csv'")

# Gender Distribution
gender_dist = df["Gender"].value_counts()
print("\nGender Distribution:")
print(gender_dist)

# Correlation between Salary and Performance
salary_perf_corr = df[["Salary", "PerformanceRating"]].corr()
print("\nCorrelation between Salary and Performance Rating:")
print(salary_perf_corr)

# Employees with High Performance Ratings
high_performers = df[df["PerformanceRating"] >= 4]
print("\nHigh Performing Employees:")
print(high_performers)

# Employees at Risk of Attrition
at_risk = df[df["Attrition"] == "Yes"]
print("\nEmployees at Risk of Attrition:")
print(at_risk)

# Average Age of Employees
avg_age = df["Age"].mean()
print("\nAverage Age of Employees:", avg_age)

# Department-wise Attrition Rate
dept_attrition = df.groupby("Department")["Attrition"].value_counts(normalize=True) * 100
print("\nDepartment-wise Attrition Rate:")
print(dept_attrition)

# Save Summary Statistics
df.describe().to_csv("summary_statistics.csv")
print("\nSummary statistics saved to 'summary_statistics.csv'")

# Employees with Salary Above 100K
high_salary_employees = df[df["Salary"] > 100000]
print("\nEmployees with Salary Above 100K:")
print(high_salary_employees)

# Hiring Trend Over the Years
df["HireYear"] = pd.to_datetime(df["HireDate"]).dt.year
hiring_trend = df["HireYear"].value_counts().sort_index()
print("\nHiring Trend Over the Years:")
print(hiring_trend)

# Department-wise Average Age
dept_age = df.groupby("Department")["Age"].mean()
print("\nAverage Age by Department:")
print(dept_age)

# Performance Distribution by Department
dept_performance = df.groupby("Department")["PerformanceRating"].mean()
print("\nPerformance Rating by Department:")
print(dept_performance)

# Employee Tenure Analysis
df["Tenure"] = pd.to_datetime("today").year - df["HireYear"]
tenure_dist = df["Tenure"].value_counts()
print("\nEmployee Tenure Distribution:")
print(tenure_dist)

# Salary Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Salary"], bins=30, kde=True, color="blue")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# Attrition by Tenure
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Attrition"], y=df["Tenure"], palette="coolwarm")
plt.title("Attrition vs Tenure")
plt.xlabel("Attrition")
plt.ylabel("Tenure")
plt.show()

# Stacked Bar Chart - Gender Distribution by Department
plt.figure(figsize=(8,5))
department_gender = df.groupby(["Department", "Gender"]).size().unstack()
department_gender.plot(kind="bar", stacked=True, colormap="coolwarm")
plt.title("Gender Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Swarm Plot - Salary vs Age
plt.figure(figsize=(8,5))
sns.swarmplot(x=df["Department"], y=df["Salary"], hue=df["Gender"], palette="coolwarm")
plt.title("Salary vs Age by Department")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.xticks(rotation=45)
plt.show()

# Predictive Analytics - Salary Prediction
X = df[["Age", "Tenure", "PerformanceRating"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nSalary Prediction Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[["Salary", "Age", "PerformanceRating", "Tenure"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Export Processed Data to Excel
df.to_excel("processed_hr_data.xlsx", index=False)
print("\nProcessed HR data saved to 'processed_hr_data.xlsx'")
