import pandas as pd

df = pd.read_csv('sales_and_customer_insights.csv')

print(df.head())
print("--------")
print(df.info())
print("--------")
print(df.info(memory_usage='deep'))
print("--------")
print(df.describe())
print("--------")
