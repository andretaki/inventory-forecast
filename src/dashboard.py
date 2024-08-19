import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import DataLoader
from predict import load_model, make_predictions
import numpy as np

# Load your model and data
@st.cache_resource
def load_data_and_model():
    data_loader = DataLoader()
    model = load_model("tft_model.ckpt")
    return data_loader, model

data_loader, model = load_data_and_model()

# Streamlit app
st.set_page_config(page_title="Inventory Forecast Dashboard", layout="wide")

st.title("Inventory Forecast Dashboard")

# Sidebar for user input
st.sidebar.header("Forecast Settings")
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
selected_products = st.sidebar.multiselect("Select Products", data_loader.get_product_list())

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Historical Data and Forecast")
    
    # Load historical data
    historical_data = data_loader.load_orders('2022-01-01', '2024-12-31')  # Adjust date range as needed
    historical_data = data_loader.preprocess_data(historical_data, data_loader.load_order_items())
    
    # Make predictions
    predictions = make_predictions(model, data_loader)
    
    # Prepare data for plotting
    for product in selected_products:
        fig = go.Figure()
        
        # Historical data
        historical = historical_data[historical_data['product'] == product]
        fig.add_trace(go.Scatter(x=historical['order_date'], y=historical['quantity'],
                                 mode='lines', name='Historical'))
        
        # Forecast
        forecast = predictions[predictions['product'] == product]
        fig.add_trace(go.Scatter(x=pd.date_range(start=historical['order_date'].max(), periods=forecast_horizon+1),
                                 y=forecast['prediction'].values[:forecast_horizon],
                                 mode='lines', name='Forecast', line=dict(dash='dash')))
        
        fig.update_layout(title=f"Product: {product}", xaxis_title="Date", yaxis_title="Quantity")
        st.plotly_chart(fig)

with col2:
    st.subheader("Inventory Metrics")
    
    # Calculate some example metrics
    for product in selected_products:
        historical = historical_data[historical_data['product'] == product]
        forecast = predictions[predictions['product'] == product]
        
        avg_demand = historical['quantity'].mean()
        forecast_demand = forecast['prediction'].mean()
        stock_on_hand = historical['quantity'].iloc[-1]  # Assume last historical point is current stock
        
        st.write(f"**{product}**")
        st.write(f"Average Historical Demand: {avg_demand:.2f}")
        st.write(f"Forecasted Average Demand: {forecast_demand:.2f}")
        st.write(f"Current Stock on Hand: {stock_on_hand:.2f}")
        
        # Days of Supply
        if forecast_demand > 0:
            days_of_supply = stock_on_hand / forecast_demand
            st.write(f"Estimated Days of Supply: {days_of_supply:.2f}")
        
        st.write("---")

# Additional Insights
st.subheader("Additional Insights")

# Overall forecast accuracy
mape = np.mean(np.abs((predictions['prediction'] - predictions['actuals']) / predictions['actuals'])) * 100
st.write(f"Overall Forecast Accuracy (MAPE): {mape:.2f}%")

# Top products by forecasted demand
top_products = predictions.groupby('product')['prediction'].sum().sort_values(ascending=False).head(5)
st.write("Top 5 Products by Forecasted Demand:")
st.write(top_products)

# You can add more visualizations or metrics here

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Dashboard created with Streamlit")
