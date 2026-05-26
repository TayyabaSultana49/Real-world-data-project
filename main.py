import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("supermarket_sales.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

sales_by_product = df.groupby("Product line")["Total"].sum()
sales_by_product.plot(kind="bar")
plt.title("Sales by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

sales_by_city = df.groupby("City")["Total"].sum()
sales_by_city.plot(kind="bar")
plt.title("Sales by City")
plt.xlabel("City")
plt.ylabel("Total Sales")
plt.show()

plt.hist(df["Total"], bins=30)
plt.title("Sales Distribution")
plt.xlabel("Total Sales")
plt.ylabel("Count")
plt.show()

sns.boxplot(y=df["Total"])
plt.title("Sales Outliers")
plt.show()

numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

print("\nFINAL INSIGHTS:")
print("1. Some product lines generated more sales than others.")
print("2. Sales varied across different cities.")
print("3. Total sales contain some outliers.")
print("4. Strong relationships exist between some numeric features.")

print("\nProject completed successfully!")