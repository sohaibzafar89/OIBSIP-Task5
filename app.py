import streamlit as st
import pandas as pd
import pickle

# Load the trained model and scaler files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

# Create a Streamlit app and add input fields for TV, radio, and newspaper advertising
st.title('💰Sales Prediction')
st.write("*Developed for 🌎 with ❤️‍🔥 by Sohaib👨🏻‍💻🇵🇰*")
st.write('Enter the advertising budget below to predict sales.')

# Taking the input from user
new_tv = st.number_input('TV Advertising', min_value=0.0, max_value=1000.0, step=0.1)
new_radio = st.number_input('Radio Advertising', min_value=0.0, max_value=1000.0, step=0.1)
new_newspaper = st.number_input('Newspaper Advertising', min_value=0.0, max_value=1000.0, step=0.1)

# Button to trigger the prediction
if st.button('Predict'):
    new_value = pd.DataFrame([[new_tv, new_radio, new_newspaper]])
    new_value_scaled = scaler.transform(new_value)
    prediction = model.predict(new_value_scaled)
    st.markdown(f"Prediction result: **{prediction[0]}**")


