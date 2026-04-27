import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

# Set page config
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Advertising Sales Prediction App")
st.markdown("""
### Project Description
This application uses a **Polynomial Regression Model** to predict future sales based on advertising budgets for **TV**, **Radio**, and **Newspaper**. 

The model has been trained on historical advertising data. By adjusting the budget allocations for each channel using the sliders in the sidebar, you can get instant predictions of the expected sales.

---
""")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "sales_model.pkl"
    if not os.path.exists(model_path):
        # Train the model if it doesn't exist
        data = pd.read_csv("advertising.csv")
        X = data[['TV', 'Radio', 'Newspaper']]
        y = data['Sales']
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
        model.fit(X, y)
        joblib.dump(model, model_path)
        return model
    return joblib.load(model_path)

model = load_model()

# Sidebar for user input
st.sidebar.header("Input Advertising Budgets")

# Sliders for budget inputs (based on the typical range in the dataset)
tv_budget = st.sidebar.slider("TV Budget", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
radio_budget = st.sidebar.slider("Radio Budget", min_value=0.0, max_value=50.0, value=25.0, step=1.0)
newspaper_budget = st.sidebar.slider("Newspaper Budget", min_value=0.0, max_value=120.0, value=50.0, step=1.0)

# Make prediction
input_data = pd.DataFrame({
    'TV': [tv_budget],
    'Radio': [radio_budget],
    'Newspaper': [newspaper_budget]
})

# Display current inputs and Predict button
st.subheader("Interactive Prediction")
st.write("Adjust the budgets in the sidebar to see the predicted sales instantly.")

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📺 TV Budget: **${tv_budget:.2f}**")
with col2:
    st.info(f"📻 Radio Budget: **${radio_budget:.2f}**")
with col3:
    st.info(f"📰 Newspaper Budget: **${newspaper_budget:.2f}**")

# Calculate prediction
prediction = model.predict(input_data)[0]

# Display result
st.markdown("### 📊 Predicted Sales")
st.success(f"## **{prediction:.2f}** units")
