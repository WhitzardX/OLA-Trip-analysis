import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import zscore

# Streamlit Page configuration
st.set_page_config(page_title="Ola Trips Data Analysis", page_icon="🚗", layout="wide")

# Title and Introduction
st.title("Ola Trips Data Analysis and Insights")
st.write("This app analyzes the Ola Trips dataset to provide key insights on trip trends, ride categories, and cost structures, along with predictions using machine learning models.")

# File Upload Section
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    df['booking id'] = df['booking id'].astype('str')
    st.write("### Dataset Overview")
    st.write(df.head(5))
    st.write(f"Dataset Shape: {df.shape}")
    st.write(f"Column Names: {', '.join(df.columns)}")
    
    # Data Preprocessing and Feature Engineering
    st.write("### Feature Engineering and Data Preprocessing")
    df['booking_date_time'] = (pd.to_datetime("1899-12-30") + pd.to_timedelta(df['booking_date_time'], unit="D")).dt.year
    df['time_of_day'] = (pd.to_timedelta(df['time_of_day'], unit = 'D').dt.round('us')).astype(str).str.replace("0 days ", "")
    df['time_of_day'] = pd.to_timedelta(df['time_of_day']).dt.components.hours
    df['weekday_weekend'] = list(map(lambda x: 'Weekend' if x in ('Sat', 'Sun') else 'Weekday', df['day_of_week']))
    
    # Fill Missing Values
    reason_mode = df['reason'].mode()[0]
    df['reason'] = df['reason'].fillna(reason_mode)

    if 'month' in df.columns:  # Check if 'month' exists in the dataset
        # Defining month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June']
        df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

        # Grouping by month and day_of_week to get counts
        monthly_day_counts = df.groupby('month')[['day_of_week']].count()
        
        # Plotting
        st.write("### Monthly Day-of-Week Count")
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_day_counts.plot.line(ax=ax, color='blue')
        ax.set_title("Day of Week Count by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Day of Week Count")
        st.pyplot(fig)
    
    def am_pm(x):
        if x <=6:
            return 'Midnight'
        elif x<=11:
            return 'Morning'
        elif x<=18:
            return "Noon"
        else:
            return 'Night'

    df['am_or_pm'] = df['time_of_day'].apply(am_pm)

    # Removing unnecessary columns
    df.rename(columns={'booking_date_time': 'year'}, inplace=True)
    
    # Display Preprocessed Data
    st.write(df.head(3))

### Step 2: Interactive Metrics Section

    # Total Trips
    total_trips = df['booking id'].nunique()
    st.sidebar.write(f"Total Trips: {total_trips}")
    
    # Average Fare
    avg_fare = df['total_trip_cost'].mean()
    st.sidebar.write(f"Average Fare: ${avg_fare:.2f}")
    
    # Total Revenue
    total_revenue = df['total_trip_cost'].sum()
    st.sidebar.write(f"Total Revenue: ${total_revenue:.2f}")
    
    # Most Frequent Trip Reasons
    most_frequent_reasons = df['reason'].value_counts().nlargest(5)
    st.sidebar.write("Most Frequent Trip Reasons")
    st.sidebar.write(most_frequent_reasons)

### Step 3: Visualizations Section

    # Visualize the Most Frequent Trip Reasons
    st.write("### Most Frequent Trip Reasons")
    fig, ax = plt.subplots()
    most_frequent_reasons.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Most Frequent Trip Reasons')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Trip Reason')
    st.pyplot(fig)

    # Visualize Distribution of Fare
    st.write("### Distribution of Fare (Total Trip Cost)")
    fig, ax = plt.subplots()
    sns.histplot(df['total_trip_cost'], kde=True, ax=ax, color='blue')
    ax.set_title('Fare Distribution')
    ax.set_xlabel('Fare ($)')
    st.pyplot(fig)
    
    # Visualize Peak Ride Times
    st.write("### Peak Ride Times (Time of Day vs Day of Week)")
    heatmap_data = pd.pivot_table(df, values='booking id', columns='am_or_pm', index='weekday_weekend', aggfunc='count')
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title('Peak Ride Times')
    st.pyplot(fig)

    # Ride Category Distribution
    st.write("### Distribution of Ride Categories")
    ride_category_count = df['category'].value_counts()
    fig, ax = plt.subplots()
    ride_category_count.plot(kind='bar', ax=ax, color='orange')
    ax.set_title('Ride Categories Distribution')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Ride Category')
    st.pyplot(fig)

### Step 4: Machine Learning Model Section (Linear Regression and Random Forest Regressor)

    # Data Preprocessing for Machine Learning
    df_encoded = df.copy()
    cols_to_encode = ['month', 'gender', 'reason', 'category', 'day_of_week', 'weekday_weekend', 'am_or_pm']
    scaler = StandardScaler()
    encoder = LabelEncoder()

    for col in cols_to_encode:
        df_encoded[col] = encoder.fit_transform(df_encoded[col])

    X = df_encoded.drop(['total_trip_cost', 'booking id','commission_base_cost', 'driver_base_cost', 'total_tax', 'ratings'], axis=1)
    y = df_encoded['total_trip_cost']
    
    # Splitting the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions_lr = model.predict(X_test_scaled)

    lr_mse = MSE(y_test, predictions_lr)
    lr_mae = MAE(y_test, predictions_lr)
    lr_r2 = r2_score(y_test, predictions_lr)

    # Displaying Linear Regression Results
    st.write("### Linear Regression Model Results")
    st.write(f"Mean Squared Error (MSE): {lr_mse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {lr_mae:.2f}")
    st.write(f"R-Squared Score: {lr_r2:.2f}")

    # Plotting Predictions vs Actual
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions_lr)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    ax.set_title("Linear Regression: Actual vs Predicted")
    st.pyplot(fig)

    # Random Forest Regressor Model
    rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr_model.fit(X_train_scaled, y_train)
    predictions_rfr = rfr_model.predict(X_test_scaled)

    rfr_mse = MSE(y_test, predictions_rfr)
    rfr_mae = MAE(y_test, predictions_rfr)
    rfr_r2 = r2_score(y_test, predictions_rfr)

    # Displaying Random Forest Results
    st.write("### Random Forest Regressor Results")
    st.write(f"Mean Squared Error (MSE): {rfr_mse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {rfr_mae:.2f}")
    st.write(f"R-Squared Score: {rfr_r2:.2f}")

    # Plotting Random Forest Predictions vs Actual
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions_rfr)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    ax.set_title("Random Forest Regressor: Actual vs Predicted")
    st.pyplot(fig)

### Step 5: Insights and Recommendations

    st.write("### Key Insights")
    st.write("- **Peak Ride Times**: Most rides are booked between **7 PM and 11 PM**, indicating that people typically use rides after work hours or for social events.")
    st.write("- **Revenue Trends**: There is a significant increase in rides during **April**, likely due to seasonal factors.")
    st.write("- **Ride Categories**: **Micro** and **Mini** cars are the most commonly chosen ride types.")
    st.write("- **Fare Distribution**: The average fare is strongly correlated with the **distance travelled**.")
    
    st.write("### Recommendations")
    st.write("- **Peak Pricing**: Implement higher fares during peak hours (7 PM - 11 PM) to optimize revenue.")
    st.write("- **Optimal Driver Deployment**: Deploy more **Micro** and **Mini** cars as they are in higher demand.")
    st.write("- **Promotional Offers**: Consider offering discounts during off-peak periods to encourage more bookings.")
    st.write("- **Revenue Optimization**: Focus on **April** as a high-traffic month and ensure sufficient availability of cars during this period.")

    st.write("### Conclusion")
    st.write("By analyzing the data and building predictive models, we can gain valuable insights into ride trends and optimize pricing strategies and resource allocation.")

# Run the Streamlit app by executing the following in the terminal:
# streamlit run app.py
