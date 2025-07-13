import streamlit as st
import pickle
import numpy as np

# Load the crop recommendation model
@st.cache_resource
def load_crop_model():
    try:
        model = pickle.load(open('C:/Users/praja/Downloads/naive_bayes_model.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise
    return model

# Crop number to name mapping
crop_dict = {
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

model = load_crop_model()

# Streamlit app UI
st.title("🌱 Crop Recommendation System")

st.markdown("### Enter the following details:")

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorous (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Recommend Crop"):
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)[0]
        crop = crop_dict.get(prediction, "Unknown Crop")
        st.success(f"✅ Recommended Crop: **{crop.capitalize()}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
