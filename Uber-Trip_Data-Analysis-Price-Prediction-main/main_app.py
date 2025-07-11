import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Uber dataset with caching
@st.cache_data
def load_data():
    dataset = pd.read_csv("uber.csv")
    dataset.dropna(inplace=True)  # Remove missing values
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'], utc=True)
    dataset['hour'] = dataset['pickup_datetime'].dt.hour
    dataset['day_of_week'] = dataset['pickup_datetime'].dt.day_name()
    dataset['day'] = dataset['pickup_datetime'].dt.day
    dataset['month'] = dataset['pickup_datetime'].dt.month
    dataset['year'] = dataset['pickup_datetime'].dt.year
    dataset['day-night'] = pd.cut(dataset['hour'], bins=[0, 10, 15, 19, 24], labels=['Morning', 'Afternoon', 'Evening', 'Night'])
    return dataset.sample(n=5000, random_state=42)  # Limit dataset size for performance

dataset = load_data()

# Location mapping for easy selection
location_map = {
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Times Square": (40.7580, -73.9855),
    "Brooklyn": (40.6782, -73.9442),
    "Central Park": (40.7851, -73.9683)
}

# Train fare prediction model with optimized parameters
features = ['passenger_count', 'hour', 'month', 'year']  # Removed lat/lon

target = 'fare_amount'
X = dataset[features]
y = dataset[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)  # Reduced estimators for speed
model.fit(X_train, y_train)

# Batch predict fares
@st.cache_data
def batch_predict_fares(df):
    return model.predict(df[features])

dataset['Predicted Price'] = batch_predict_fares(dataset)

# Sidebar for filtering
st.sidebar.title("Filter Data")
selected_day_time = st.sidebar.multiselect("Select Time of Day", options=dataset['day-night'].dropna().unique(), default=dataset['day-night'].dropna().unique())

def filter_data(selected_day_time):
    if not selected_day_time:
        return dataset  # Return full dataset if no filter is selected
    return dataset[dataset['day-night'].isin(selected_day_time)]

filtered_data = filter_data(selected_day_time)

# Show trip data
st.title("Uber Trip Analysis Dashboard")
with st.expander("Trip Data Box"):
    st.write("Filtered Dataset:")
    st.dataframe(filtered_data[['pickup_datetime', 'day-night', 'fare_amount', 'Predicted Price']])

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    numeric_dataset = filtered_data.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True, ax=ax)
    st.pyplot(fig)

# Distribution of trips by hour
if st.checkbox("Show Trip Distribution"):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(x='hour', data=filtered_data, ax=ax)
    ax.set_title('Distribution of Trips by Hour')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Number of Trips')
    st.pyplot(fig)

# Fare Prediction Section
st.title("Uber Fare Prediction")
with st.expander("Price Prediction Box"):
    pickup_location = st.selectbox("Pickup Location", list(location_map.keys()))
    dropoff_location = st.selectbox("Dropoff Location", list(location_map.keys()))
    passengers = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    hour = st.slider("Hour of the Day", 0, 23, 12)
    month = st.slider("Month", 1, 12, 6)
    year = st.slider("Year", 2009, 2015, 2013)
    
    if st.button("Predict Fare"):
        fare = model.predict(np.array([[passengers, hour, month, year]]))[0]
        st.success(f"Predicted Fare: ${fare:.2f}")

# Model evaluation
if st.checkbox("Show Model Performance"):
    st.subheader("Model Performance")
    y_pred = model.predict(X_test[:500])  # Limit test set for faster evaluation
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test[:500], y_pred):.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test[:500], y_pred):.2f}")
