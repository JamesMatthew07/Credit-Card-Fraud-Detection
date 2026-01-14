#Load data with pandas
#Check basic  info: shaoe, dtypes, missing values
# Check class distribution (values_counts, oercentages)
# Summary statistics for Amount by Class
# Visualizations:
#   - Class distribution (bar plot)
#   - Amount distribution by Class (box plot or histogram)
#   - Correlation heatmap of a few features

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import os

df = pd.read_csv('dataset/creditcard.csv')

# Basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df['Class'].value_counts())

# Calculate fraud percentage
fraud_count = df['Class'].sum()
total_count = len(df)
fraud_pct = (fraud_count / total_count) * 100
print(f"\nFraud percentage: {fraud_pct:.2f}%")

print("\nAmount statistics by Class:")
print(df.groupby('Class')['Amount'].describe())

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Class distribution bar plot
plt.figure(figsize=(8, 5))
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution (Legitimate vs Fraud)')
plt.xlabel('Class (0=Legitimate, 1=Fraud)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 2. Amount distribution by Class (box plot)
plt.figure(figsize=(10, 5))
df.boxplot(column='Amount', by='Class')
plt.suptitle('')
plt.title('Transaction Amount Distribution by Class')
plt.ylabel('Amount ($)')
plt.xlabel('Class (0=Legitimate, 1=Fraud)')
plt.savefig(os.path.join(output_dir, "amount_by_class_boxplot.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 3. Log-scaled amount distribution
plt.figure(figsize=(12, 5))
df[df['Class'] == 0]['Amount'].apply(lambda x: np.log(x + 1)).hist(bins=50, alpha=0.5, label='Legitimate')
df[df['Class'] == 1]['Amount'].apply(lambda x: np.log(x + 1)).hist(bins=50, alpha=0.5, label='Fraud')
plt.xlabel('Log(Amount + 1)')
plt.ylabel('Frequency')
plt.title('Log-scaled Amount Distribution')
plt.legend()
plt.savefig(os.path.join(output_dir, "log_amount_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()