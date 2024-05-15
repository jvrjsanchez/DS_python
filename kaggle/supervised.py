import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Data Loading
employee_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Exploration
print("First few rows of the dataset:")
print(employee_data.head())

print("\nShape of the dataset:", employee_data.shape)

# Identify data types and missing values
print("\nData types of each column:")
print(employee_data.dtypes)

print("\nMissing values:")
print(employee_data.isnull().sum())

# Data Cleaning
cleaned_data = employee_data.dropna()

# Display the cleaned data
print("\nCleaned data:")
print(cleaned_data.head())


label_encoder = LabelEncoder()
cleaned_data['BusinessTravel_encoded'] = label_encoder.fit_transform(cleaned_data['BusinessTravel'])
cleaned_data['Department_encoded'] = label_encoder.fit_transform(cleaned_data['Department'])

# Split the dataset into training and testing sets
X = cleaned_data.drop(['Attrition'], axis=1)  # Features
y = cleaned_data['Attrition']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the preprocessed data and split sets
print("Cleaned Data after Preprocessing:")
print(cleaned_data.head())

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
