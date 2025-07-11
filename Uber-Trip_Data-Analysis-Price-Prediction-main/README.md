# 🚕 Uber Fare Prediction & Trip Analysis Dashboard

This interactive Streamlit application analyzes historical Uber trip data and predicts fare amounts using a trained machine learning model. It enables users to explore trip patterns, filter by time of day, and predict fares based on user inputs.

---

## 🔍 Features

- 📊 **Trip Data Exploration**: View and filter trip data by time of day (Morning, Afternoon, Evening, Night).
- 🔥 **Visualizations**:
  - Correlation heatmap of numerical features.
  - Trip distribution by hour of the day.
- 💡 **Fare Prediction**:
  - Predict fare amounts based on custom inputs like time, passengers, and date.
- 📈 **Model Evaluation**:
  - View key metrics including Mean Absolute Error and Mean Squared Error.

---

## 🧠 Technologies Used

- **Streamlit**: for building the interactive UI
- **Pandas / NumPy**: for data handling
- **Matplotlib / Seaborn**: for visualizations
- **Scikit-learn**: for model training and prediction (`RandomForestRegressor`)
- **Uber dataset**: with pickup times and fare information

---

## 📂 Dataset Details

- Input file: `uber.csv`
- Sample size used: 5,000 rows (for performance)
- Important features:
  - `passenger_count`
  - `hour`, `month`, `year` (derived from `pickup_datetime`)
  - `fare_amount` (target)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/uber-fare-prediction.git
cd uber-fare-prediction
