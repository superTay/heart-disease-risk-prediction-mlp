import numpy as np
import pandas as pd

# Exploration of the dataset

df = pd.read_csv('heart.csv')

#print("First 5 rows of the dataset:")
#print(df.head())
#print("\nDataset Columns:")
#print(df.columns)    


'''In the Heart Disease UCI dataset, all columns are relevant for prediction.
Unlike the Titanic dataset, this dataset does not include irrelevant fields
(such as names, tickets or cabin identifiers). Therefore, no columns are 
removed at this stage.'''


# Data analysis

''' # df.info() returns:
 - total number of rows and columns
 - data type of each column
 - number of non-null values (helps detect missing data)
 - memory usage of the DataFrame'''

print("\nDataset Info:")
print(df.info())

''' # df.describe() returns descriptive statistics for all numerical columns:
 - count: number of non-null entries
 - mean: average value of the column
 - std: standard deviation (measure of dispersion)
 - min: minimum observed value
 - 25%: first quartile (25th percentile)
 - 50%: median (50th percentile)
 - 75%: third quartile (75th percentile)
 - max: maximum observed value'''

print("\nStatistical Summary:")
print(df.describe())

# Percentage of males

male_percentage = df['sex'].mean() * 100
print(f"Percentage of males: {male_percentage:.2f}%")

# Checking for null values

n_rows = len(df)
null_counts = df.isna().sum()
null_percentages = (null_counts / n_rows) * 100
print("\nMissing Values per Column:")
print(null_counts)
print("\nPercentage of Missing Values per Column:")
print(null_percentages.round(2))
# In this dataset, there are no null values in any column.


# Detecting hidden missing values or incorrect encodings

print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")


# 'thal' should only contain values 1, 2, or 3
# 'ca' should only contain values 0, 1, 2, or 3

invalid_thal = df[~df['thal'].isin([1,2,3])]
invalid_ca = df[~df['ca'].isin([0,1,2,3])]
print("\nRows with invalid 'thal' values:")
print(len(invalid_thal))
print("\nRows with invalid 'ca' values:")
print(len(invalid_ca))

#Converting 'thal' and 'ca' to numeric, forcing errors to NaN

# Replacing encoded missing values / invalid category codes with NaN

df['thal'] = df['thal'].replace(0, np.nan)
df['ca'] = df['ca'].replace(4, np.nan)

print("\nMissing values AFTER converting invalid codes to NaN:")
print(df.isna().sum())


# Imputing missing values (thal and ca) using mode


df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
df['ca'] = df['ca'].fillna(df['ca'].mode()[0])

print("\nMissing values AFTER imputing mode values:")
print(df.isna().sum())