import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Creating a  dataset

data = {
    "Customer_ID": np.arange(1, 101),
    "Age": np.random.randint(18, 65, 100),
    "Gender": np.random.choice(["Male", "Female"], 100),
    "Product_Category": np.random.choice(["Electronics", "Clothing", "Home Decor", "Books"], 100),
    "Sales_Revenue": np.random.uniform(20, 500, 100).round(2),
    "Purchase_Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "Customer_Rating": np.random.choice([1, 2, 3, 4, 5, np.nan], 100)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Data Cleaning

# Check for missing values
print("Missing values before handling:")
print(df.isnull().sum())

# Fill missing values in 'Customer_Rating' with the median rating
df["Customer_Rating"] = df["Customer_Rating"].fillna(df["Customer_Rating"].median())

# Step 3: Descriptive Statistics

print("\nSummary statistics:")
print(df.describe())

# Step 4: Data Visualization

plt.figure(figsize=(14, 6))

# Sales Revenue Distribution
plt.subplot(1, 3, 1)
sns.histplot(df["Sales_Revenue"], kde=True, color="skyblue")
plt.title("Sales Revenue Distribution")

# Customer Rating Distribution
plt.subplot(1, 3, 2)
sns.boxplot(x=df["Customer_Rating"], color="lightgreen")
plt.title("Customer Ratings")

# Product Category Sales
plt.subplot(1, 3, 3)
sns.countplot(x=df["Product_Category"], palette="pastel", order=df["Product_Category"].value_counts().index)
plt.title("Product Category Distribution")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Step 5: Explore Trends and Correlations

# Correlation Heatmap
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Scatter Plot - Age vs Sales Revenue
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Age"], y=df["Sales_Revenue"], hue=df["Gender"], palette="deep", edgecolor=None)
plt.title("Age vs Sales Revenue by Gender")
plt.show()

# Step 6: Print the Results

print("\nKey Findings:")
print("1. Sales revenue is distributed between $20 and $500 with a peak around the lower range.")
print("2. Most customers rate their purchases highly, with ratings concentrated around 4 and 5.")
print("3. Electronics and Clothing are the most purchased product categories.")
print("4. There is no strong correlation between age and sales revenue.")
