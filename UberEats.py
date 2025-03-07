import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
import re

# Set non-interactive backend for Matplotlib
matplotlib.use('Agg')

# Load Data
file_path = 'uber-eats-deliveries.csv'  # Update with actual file path
data = pd.read_csv(file_path)

# Data Preprocessing
st.title("Uber Eats Delivery Analysis")
st.write("## Data Preview")
st.dataframe(data.head())

# Handling Missing Values
data.dropna(inplace=True)

# Strip column names to remove extra spaces
data.columns = data.columns.str.strip()

# Check available columns
st.write("## Available Columns in Dataset")
st.write(data.columns.tolist())

# Cleaning "Time_taken(min)" column
if 'Time_taken(min)' in data.columns:
    data['Time_taken(min)'] = data['Time_taken(min)'].astype(str).apply(lambda x: re.sub(r'[^0-9.]', '', x)).astype(float)
else:
    st.write("⚠ Warning: Column 'Time_taken(min)' not found in dataset.")

# Encoding Categorical Data
encoder = LabelEncoder()
categorical_cols = ['Weatherconditions', 'Road_traffic_density', 'Festival', 'Type_of_order', 'Type_of_vehicle']
for col in categorical_cols:
    if col in data.columns:  # Check if column exists
        data[col] = encoder.fit_transform(data[col])
    else:
        st.write(f"⚠ Warning: Column '{col}' not found in dataset.")

# Standardizing Numeric Features
scaler = StandardScaler()
data[['Delivery_person_Age', 'Delivery_person_Ratings', 'Time_taken(min)']] = scaler.fit_transform(
    data[['Delivery_person_Age', 'Delivery_person_Ratings', 'Time_taken(min)']]
)

# Exploratory Data Analysis (EDA)
st.write("## Exploratory Data Analysis")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

sns.histplot(data['Time_taken(min)'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Delivery Time Distribution")

if 'Road_traffic_density' in data.columns:
    sns.boxplot(x=data['Road_traffic_density'], y=data['Time_taken(min)'], ax=axes[0, 1])
    axes[0, 1].set_title("Delivery Time vs Traffic Density")
else:
    st.write("⚠ Warning: 'Road_traffic_density' column not found.")

if 'Weatherconditions' in data.columns:
    sns.scatterplot(x=data['Weatherconditions'], y=data['Time_taken(min)'], ax=axes[1, 0])
    axes[1, 0].set_title("Weather Conditions vs Delivery Time")
else:
    st.write("⚠ Warning: 'Weatherconditions' column not found.")

sns.barplot(x=data['Type_of_vehicle'], y=data['Time_taken(min)'], ax=axes[1, 1])
axes[1, 1].set_title("Vehicle Type vs Delivery Time")

sns.barplot(x=data['multiple_deliveries'], y=data['Time_taken(min)'], ax=axes[2, 0])
axes[2, 0].set_title("Impact of Multiple Deliveries on Time")

sns.barplot(x=data['Festival'], y=data['Time_taken(min)'], ax=axes[2, 1])
axes[2, 1].set_title("Delivery Time during Festivals")

plt.tight_layout()
st.pyplot(fig)

# Clustering Analysis
st.write("## Clustering Analysis")
if 'Weatherconditions' in data.columns and 'Road_traffic_density' in data.columns:
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Time_taken(min)', 'Road_traffic_density', 'Weatherconditions']])
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['Time_taken(min)'], y=data['Road_traffic_density'], hue=data['Cluster'], palette='viridis', ax=ax)
    ax.set_title("K-Means Clustering of Deliveries")
    st.pyplot(fig)
else:
    st.write("⚠ Skipping clustering: 'Weatherconditions' or 'Road_traffic_density' column not found.")

# Deployment
st.write("## Conclusion")
st.write("This analysis provides key insights into Uber Eats delivery patterns, highlighting the impact of weather, traffic, and vehicle type on delivery times. The clustering and association rule mining help optimize delivery strategies.")

# Run Streamlit App using: streamlit run <filename>.py
