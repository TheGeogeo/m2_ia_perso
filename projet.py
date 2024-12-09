import pandas as pd

df = pd.read_csv('sales_and_customer_insights.csv')

print(df.head())
print(df.info())
print(df.describe())
