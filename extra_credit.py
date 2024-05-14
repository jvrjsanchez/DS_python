import pandas as pd
import matplotlib.pyplot as plt

existing_data = pd.read_csv("weather_data.csv")
date_data = pd.DataFrame({'date': pd.date_range(start='2024-01-01', end='2024-12-31')})
merged_data = pd.concat([existing_data, date_data], axis=1)
merged_data.to_csv("combined_data.csv", index=False)

# Data Loading
weather_data = pd.read_csv("combined_data.csv")

# Exploration
print("First few rows of the dataset:")
print(weather_data.head())

print("\nShape of the dataset:", weather_data.shape)

# Identify data types and missing values
print("\nData types of each column:")
print(weather_data.dtypes)

print("\nMissing values:")
print(weather_data.isnull().sum())

# Data Cleaning
cleaned_data = weather_data.dropna()

# Convert 'date' column to datetime format
cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])

# Display the cleaned data
print("\nCleaned data:")
print(cleaned_data.head())

print(existing_data.columns)

# Calculate average temperature and humidity for each month
monthly_avg = cleaned_data.groupby(cleaned_data['date'].dt.month).agg({'Outdoor Drybulb Temperature [C]': 'mean', 'Outdoor Relative Humidity [%]': 'mean'})

# Visualize monthly average temperature and humidity
plt.figure(figsize=(10, 6))
monthly_avg.plot(kind='bar', title='Monthly Average Temperature and Humidity')
plt.xlabel('Month')
plt.ylabel('Value')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# Determine month with highest and lowest average wind speed
wind_speed_monthly_avg = cleaned_data.groupby(cleaned_data['date'].dt.month)['wind_speed'].mean()
highest_wind_month = wind_speed_monthly_avg.idxmax()
lowest_wind_month = wind_speed_monthly_avg.idxmin()
print("Month with highest average wind speed:", highest_wind_month)
print("Month with lowest average wind speed:", lowest_wind_month)

# Visualize distribution of wind speed
plt.figure(figsize=(10, 6))
cleaned_data['wind_speed'].plot(kind='hist', bins=20, title='Distribution of Wind Speed')
plt.xlabel('Wind Speed')
plt.ylabel('Frequency')
plt.show()

# Count frequency of each weather condition and display using a bar chart
plt.figure(figsize=(10, 6))
weather_condition_counts = cleaned_data['weather_condition'].value_counts()
weather_condition_counts.plot(kind='bar', title='Frequency of Weather Conditions')
plt.xlabel('Weather Condition')
plt.ylabel('Frequency')
plt.show()
